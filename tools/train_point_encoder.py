"""训练点式 3D encoder（Stage 5b）：逐点语义分割 + 实例嵌入（+ 可选开放词汇对齐）。

与当前几何流水线的关系：这一阶段产出的模型是 Stage 6–9（蒸馏 / 剪枝 / 量化 /
TensorRT）唯一可压缩的载体。评测口径直接复用已有工具，保证与几何跟踪器可比。

用法（在项目根目录）：

    # 先快速验证链路（几十步就停）
    python -m tools.train_point_encoder \
        --dataset-root outputs/point_dataset --output-directory outputs/stage5/run1 \
        --max-steps 20

    # 正式训练
    python -m tools.train_point_encoder \
        --dataset-root outputs/point_dataset --output-directory outputs/stage5/run1 \
        --epochs 60 --batch-size 4 --num-points 8192 --lr 2e-3

    # 加上开放词汇对齐（需先用 tools.build_text_embeddings 生成文本嵌入）
    python -m tools.train_point_encoder ... --text-embeddings outputs/stage5/text_embeddings.npz \
        --text-loss-weight 0.5
"""

import argparse
import contextlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.point_dataset import PointCloudPatchDataset
from src.models.losses import (
    compute_class_weights,
    discriminative_loss,
    semantic_loss,
    text_alignment_loss,
)
from src.models.point_encoder import PointEncoderConfig, PointEncoder, parameter_count


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", default="outputs/point_dataset")
    parser.add_argument("--output-directory", default="outputs/stage5/run1")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--patches-per-scene", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-alpha", type=float, default=0.5,
                        help="采样时类别均衡强度；0 表示不均衡")
    parser.add_argument("--class-weight-mode", default="inverse_sqrt",
                        choices=["none", "inverse_sqrt", "inverse"])
    parser.add_argument("--instance-loss-weight", type=float, default=0.5)
    parser.add_argument("--text-loss-weight", type=float, default=0.0)
    parser.add_argument("--text-embeddings", default=None,
                        help="tools.build_text_embeddings 产出的 npz")
    parser.add_argument("--stem-channels", type=int, default=64)
    parser.add_argument("--instance-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", help="混合精度（3090 上可省显存）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="每 epoch 最多多少步，用于快速冒烟验证")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--scene-limit", type=int, default=None)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #


def confusion_from_predictions(confusion, prediction, target, num_classes, ignore_index=-1):
    """把一批预测累加进混淆矩阵。"""

    valid = target != ignore_index
    if not valid.any():
        return confusion
    predicted = prediction[valid].reshape(-1).to(torch.int64)
    truth = target[valid].reshape(-1).to(torch.int64)
    flat = truth * num_classes + predicted
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return confusion + counts.reshape(num_classes, num_classes)


def metrics_from_confusion(confusion):
    """从混淆矩阵算逐点准确率与 mIoU（只统计出现过的类别）。"""

    total = confusion.sum().item()
    correct = torch.diagonal(confusion).sum().item()
    accuracy = correct / total if total else 0.0

    intersection = torch.diagonal(confusion).to(torch.float64)
    ground_truth = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    union = ground_truth + predicted - intersection

    present = ground_truth > 0
    if not present.any():
        return {"accuracy": accuracy, "miou": 0.0, "classes_present": 0}
    iou = torch.zeros_like(intersection)
    iou[present] = intersection[present] / union[present].clamp(min=1e-9)
    return {
        "accuracy": round(accuracy, 4),
        "miou": round(float(iou[present].mean()), 4),
        "classes_present": int(present.sum()),
        "per_class_iou": [round(float(value), 4) for value in iou.tolist()],
    }


@torch.no_grad()
def evaluate(model, loader, num_classes, device, max_steps=None, text_bank=None):
    """在验证集上跑一遍，返回逐点准确率与 mIoU。"""

    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    steps = 0
    for features, class_id, instance in loader:
        features = features.to(device, non_blocking=True)
        class_id = class_id.to(device, non_blocking=True)
        output = model(features)
        if text_bank is not None:
            prediction = (output.text_embedding @ text_bank.t()).argmax(dim=-1)
        else:
            prediction = output.semantic_logits.argmax(dim=-1)
        confusion = confusion_from_predictions(
            confusion, prediction.cpu(), class_id.cpu(), num_classes
        )
        steps += 1
        if max_steps and steps >= max_steps:
            break
    model.train()
    return metrics_from_confusion(confusion)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def make_grad_scaler(enabled):
    """兼容 torch 2.5（服务端）与较新版本（本地）。"""

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_text_bank(path, class_names):
    """读取文本嵌入 npz，并按 class_names 顺序对齐（顺序错位会静默毁掉训练）。"""

    data = np.load(path, allow_pickle=True)
    stored_names = [str(name) for name in data["class_names"]]
    if stored_names != list(class_names):
        raise SystemExit(
            "文本嵌入的类别顺序与数据集不一致，必须重新生成：\n"
            f"  嵌入：{stored_names[:5]}...\n  数据集：{list(class_names)[:5]}..."
        )
    embeddings = torch.from_numpy(data["embeddings"].astype(np.float32))
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("警告：没有可用的 CUDA，退回 CPU（会很慢）")

    train_dataset = PointCloudPatchDataset(
        args.dataset_root, split="train", num_points=args.num_points,
        patches_per_scene=args.patches_per_scene, class_alpha=args.class_alpha,
        augment=not args.no_augment, seed=args.seed, scene_limit=args.scene_limit,
    )
    val_dataset = PointCloudPatchDataset(
        args.dataset_root, split="val", num_points=args.num_points,
        patches_per_scene=max(1, args.patches_per_scene // 4), class_alpha=0.0,
        augment=False, seed=args.seed + 1, scene_limit=args.scene_limit,
    )
    class_names = train_dataset.class_names
    num_classes = len(class_names)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=(device.type == "cuda"), persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers), pin_memory=(device.type == "cuda"),
    )

    text_bank = None
    text_loss_weight = args.text_loss_weight
    if args.text_embeddings:
        text_bank = load_text_bank(args.text_embeddings, class_names).to(device)
        if text_bank.shape[0] != num_classes:
            raise SystemExit("文本嵌入的类别数与数据集不一致")
        print(f"已加载文本嵌入 {tuple(text_bank.shape)}，文本对齐权重 {text_loss_weight}")
    elif text_loss_weight > 0:
        raise SystemExit("--text-loss-weight > 0 时必须提供 --text-embeddings")

    config = PointEncoderConfig(
        num_classes=num_classes,
        stem_channels=args.stem_channels,
        instance_dim=args.instance_dim,
        dropout=args.dropout,
    )
    model = PointEncoder(config).to(device)
    total, trainable = parameter_count(model)
    print(f"模型参数 {total:,}（可训练 {trainable:,}）  类别数 {num_classes}")

    histogram = train_dataset.manifest["class_histogram"]
    class_weights = compute_class_weights(histogram, class_names, mode=args.class_weight_mode)
    class_weights = class_weights.to(device)
    if args.class_weight_mode != "none":
        heaviest = class_weights.argmax().item()
        print(f"类别权重：最大 {class_names[heaviest]} = {class_weights[heaviest]:.2f}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = len(train_loader) if not args.max_steps else min(len(train_loader), args.max_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def learning_rate_at(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_at)
    scaler = make_grad_scaler(args.amp and device.type == "cuda")

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    log_path = output_directory / "training_log.json"
    history = []
    best_miou = -1.0
    global_step = 0

    print(f"训练集 {len(train_dataset)} 块 / 验证集 {len(val_dataset)} 块  "
          f"每 epoch {steps_per_epoch} 步，共 {args.epochs} epoch")

    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        started = time.time()
        running = {"total": 0.0, "semantic": 0.0, "instance": 0.0, "text": 0.0}
        seen = 0

        for step, (features, class_id, instance) in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            features = features.to(device, non_blocking=True)
            class_id = class_id.to(device, non_blocking=True)
            instance = instance.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # 关闭 AMP 时用 nullcontext，避免在不支持 cuda autocast 的环境里构造它
            autocast = (
                torch.amp.autocast("cuda", enabled=True)
                if scaler.is_enabled()
                else contextlib.nullcontext()
            )
            with autocast:
                output = model(features)
                semantic = semantic_loss(output.semantic_logits, class_id, class_weights)
                loss = semantic
                instance_term = torch.zeros((), device=device)
                if args.instance_loss_weight > 0:
                    var, dist, reg = discriminative_loss(output.instance_embedding, instance)
                    instance_term = var + dist + reg
                    loss = loss + args.instance_loss_weight * instance_term
                text_term = torch.zeros((), device=device)
                if text_bank is not None and text_loss_weight > 0:
                    text_term = text_alignment_loss(output.text_embedding, class_id, text_bank)
                    loss = loss + text_loss_weight * text_term

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["total"] += loss.item()
            running["semantic"] += semantic.item()
            running["instance"] += float(instance_term.item())
            running["text"] += float(text_term.item())
            seen += 1
            global_step += 1

        if seen == 0:
            raise SystemExit("一个 epoch 都没跑到步数；检查数据集是否为空")

        train_metrics = {key: value / seen for key, value in running.items()}
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "seconds": round(time.time() - started, 1),
            "train": {key: round(value, 4) for key, value in train_metrics.items()},
        }

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate(
                model, val_loader, num_classes, device,
                max_steps=args.max_steps, text_bank=text_bank,
            )
            record["val"] = val_metrics
            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(config),
                        "class_names": list(class_names),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    output_directory / "best.pt",
                )
                record["best"] = True

        history.append(record)
        message = (f"epoch {epoch:3d}  总损失 {train_metrics['total']:.4f}"
                   f"  语义 {train_metrics['semantic']:.4f}")
        if args.instance_loss_weight > 0:
            message += f"  实例 {train_metrics['instance']:.4f}"
        if "val" in record:
            message += (f"  | 验证 mIoU {record['val']['miou']:.4f}"
                        f"  准确率 {record['val']['accuracy']:.4f}")
        message += f"  {record['seconds']}s"
        print(message, flush=True)

        log_path.write_text(
            json.dumps(
                {
                    "arguments": vars(args),
                    "model_config": asdict(config),
                    "parameter_count": total,
                    "class_names": list(class_names),
                    "best_val_miou": best_miou,
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"\n最佳验证 mIoU {best_miou:.4f}")
    print(f"检查点 {output_directory / 'best.pt'}")
    print(f"训练日志 {log_path}")


if __name__ == "__main__":
    main()
