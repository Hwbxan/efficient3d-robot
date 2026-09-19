"""评测训练好的点式 encoder（Stage 5b 的验收工具）。

两个数据源都能评：

1. `--dataset-root --split test` —— 网格采样的点云（S5a 产出），干净、均匀；
2. `--cloud path.npz` —— 任意同格式的带标签点云，**必须用它做域内评测**：
   在线融合出来的点云有噪声、遮挡、密度不均，分布和网格采样差很远，
   只看网格验证集会把指标看虚高（`stage5_design.md` §6 已写明这一点）。

指标分三层：

- **语义**：逐点准确率 + mIoU（忽略 class = -1），并给出逐类 IoU 与混淆矩阵；
- **实例（oracle 诊断）**：用 GT 实例均值做最近质心分类的纯度。衡量的是
  "嵌入空间有没有把实例分开"，需要 GT 质心，所以**不是可部署指标**；
- **实例（可部署）**：在嵌入的 kNN 图上按距离阈值取连通分量做聚类，
  再用与 `evaluate_against_gt.py` **完全相同的定义**算碎片率与纯度，
  这样数字能和几何跟踪器直接比。

若提供 `--text-embeddings`，还会用 `argmax(嵌入 @ 文本嵌入.T)` 做开放词汇预测，
报告其准确率/mIoU —— 这正是推理时会用的那条路径。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.models.point_encoder import PointEncoder, encode_cloud_in_blocks


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=None,
                        help="用数据集划分评测（与 --cloud 二选一）")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cloud", default=None,
                        help="用单个 npz 点云评测（键：xyz/normal/rgb/class/instance）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-embeddings", default=None)
    parser.add_argument("--num-points", type=int, default=8192,
                        help="分块推理的块大小，应与训练一致")
    parser.add_argument("--overlap", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ignore-class", type=int, default=-1)
    parser.add_argument("--ignore-instance", type=int, default=0)
    parser.add_argument("--min-instance-points", type=int, default=20,
                        help="参与实例指标的最小点数，避免极小实例主导统计")
    parser.add_argument("--cluster-k", type=int, default=8)
    parser.add_argument("--cluster-threshold", type=float, default=1.0,
                        help="嵌入空间的连线距离阈值")
    parser.add_argument("--max-instance-points", type=int, default=50000,
                        help="实例指标的最大点数（超出则等距抽稀，控制 O(N²) 邻域开销）")
    parser.add_argument("--scene-limit", type=int, default=None)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# 语义指标
# --------------------------------------------------------------------------- #


def semantic_metrics(prediction, target, num_classes, ignore_index=-1):
    """逐点准确率 + mIoU + 逐类 IoU。"""

    valid = target != ignore_index
    if not valid.any():
        return {"points": 0, "accuracy": None, "miou": None}

    predicted = prediction[valid]
    truth = target[valid]
    flat = truth * num_classes + predicted
    confusion = np.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )

    total = int(confusion.sum())
    correct = int(np.diagonal(confusion).sum())
    intersection = np.diagonal(confusion).astype(np.float64)
    ground_truth = confusion.sum(axis=1).astype(np.float64)
    predicted_totals = confusion.sum(axis=0).astype(np.float64)
    union = ground_truth + predicted_totals - intersection

    present = ground_truth > 0
    iou = np.zeros(num_classes, dtype=np.float64)
    iou[present] = intersection[present] / np.maximum(union[present], 1e-9)

    return {
        "points": total,
        "accuracy": round(correct / total, 4) if total else None,
        "miou": round(float(iou[present].mean()), 4) if present.any() else None,
        "classes_present": int(present.sum()),
        "per_class_iou": [round(float(value), 4) for value in iou.tolist()],
        "confusion": confusion.tolist(),
    }


# --------------------------------------------------------------------------- #
# 实例指标
# --------------------------------------------------------------------------- #


def nearest_centroid_purity(embedding, instance, ignore_instance=0):
    """oracle 诊断：把每个点判给最近的 GT 实例均值，看判对的比例。

    需要 GT 均值，所以**不是可部署指标**——它回答的是"嵌入空间本身
    有没有把实例分开"这个问题。纯度和随机基线一起给出，方便判断有没有信息量。
    """

    valid = instance != ignore_instance
    if valid.sum() < 2:
        return {"points": int(valid.sum()), "purity": None, "baseline": None}

    embedding = embedding[valid]
    labels = instance[valid]
    unique, inverse = np.unique(labels, return_inverse=True)

    sums = np.zeros((unique.shape[0], embedding.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, embedding)
    counts = np.bincount(inverse, minlength=unique.shape[0]).astype(np.float64)
    means = sums / counts[:, None]

    # 分块算距离，避免 (N, C) 全量驻留
    correct = 0
    for start in range(0, embedding.shape[0], 4096):
        block = embedding[start:start + 4096]
        distance = ((block[:, None, :] - means[None, :, :]) ** 2).sum(axis=-1)
        correct += int((distance.argmin(axis=1) == inverse[start:start + 4096]).sum())

    # 随机基线：按实例点数加权，猜中概率 = sum((n_i/N)^2)
    share = counts / counts.sum()
    return {
        "points": int(valid.sum()),
        "instances": int(unique.shape[0]),
        "purity": round(correct / embedding.shape[0], 4),
        "baseline": round(float((share ** 2).sum()), 4),
    }


def knn_graph_clusters(embedding, k=8, threshold=1.0, chunk=1024):
    """在嵌入的 kNN 图上按距离阈值取连通分量（无 GT、可部署）。

    用并查集实现连通分量，不依赖 scipy。只连接"互为 k 近邻且距离 < threshold"
    的点，避免长链把两个物体串起来。
    """

    total = embedding.shape[0]
    k = min(k, total - 1) if total > 1 else 0
    parent = np.arange(total)

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:          # 路径压缩
            parent[node], node = root, parent[node]
        return root

    for start in range(0, total, chunk):
        block = embedding[start:start + chunk]
        distance = ((block[:, None, :] - embedding[None, :, :]) ** 2).sum(axis=-1)
        if k:
            neighbours = np.argpartition(distance, k, axis=1)[:, :k + 1]
        else:
            neighbours = np.arange(total)[None, :]
        rows = np.repeat(np.arange(block.shape[0]), neighbours.shape[1]) + start
        cols = neighbours.reshape(-1)
        close = np.sqrt(distance[rows - start, cols]) < threshold
        for left, right in zip(rows[close], cols[close]):
            left_root, right_root = find(int(left)), find(int(right))
            if left_root != right_root:
                parent[right_root] = left_root

    roots = np.array([find(node) for node in range(total)])
    _, clusters = np.unique(roots, return_inverse=True)
    return clusters


def instance_metrics(clusters, instance, ignore_instance=0, min_points=20):
    """碎片率 / 纯度，定义与 evaluate_against_gt.py 保持一致。"""

    valid = instance != ignore_instance
    clusters = clusters[valid]
    labels = instance[valid]
    if labels.size == 0:
        return {"instances": 0}

    unique_instances, counts = np.unique(labels, return_counts=True)
    keep = counts >= min_points
    kept = set(unique_instances[keep].tolist())
    if not kept:
        return {"instances": 0, "min_points": min_points}

    mask = np.isin(labels, list(kept))
    clusters, labels = clusters[mask], labels[mask]

    instance_to_clusters = {}
    cluster_to_instances = {}
    for cluster, label in zip(clusters.tolist(), labels.tolist()):
        instance_to_clusters.setdefault(label, set()).add(cluster)
        cluster_to_instances.setdefault(cluster, set()).add(label)

    instance_count = len(instance_to_clusters)
    cluster_count = len(cluster_to_instances)
    single = sum(1 for value in instance_to_clusters.values() if len(value) == 1)
    pure = sum(1 for value in cluster_to_instances.values() if len(value) == 1)

    return {
        "instances": instance_count,
        "clusters": cluster_count,
        "single_cluster_instances": single,
        "single_cluster_ratio": round(single / instance_count, 4) if instance_count else None,
        "fragmentation_mean": round(
            sum(len(value) for value in instance_to_clusters.values()) / instance_count, 4
        ) if instance_count else None,
        "pure_clusters": pure,
        "pure_cluster_ratio": round(pure / cluster_count, 4) if cluster_count else None,
        "merged_instances": {
            str(cluster): sorted(values)
            for cluster, values in cluster_to_instances.items() if len(values) > 1
        },
        "fragmented_instances": {
            str(label): sorted(values)
            for label, values in instance_to_clusters.items() if len(values) > 1
        },
        "min_points": min_points,
    }


# --------------------------------------------------------------------------- #
# 数据加载与推理
# --------------------------------------------------------------------------- #


def load_cloud_arrays(source, dataset_root=None, split=None, scene_limit=None):
    """返回 {场景名: 数组字典}。"""

    if dataset_root:
        root = Path(dataset_root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if split not in manifest["splits"]:
            raise SystemExit(f"未知划分 {split}，可选 {sorted(manifest['splits'])}")
        scenes = list(manifest["splits"][split])
        if scene_limit:
            scenes = scenes[:scene_limit]
        if not scenes:
            raise SystemExit(f"划分 {split} 里没有场景")
        arrays = {}
        for scene in scenes:
            path = root / "scenes" / f"{scene}.npz"
            if not path.exists():
                raise SystemExit(f"缺少场景文件 {path}")
            arrays[scene] = np.load(path)
        return arrays

    if source:
        path = Path(source)
        if not path.exists():
            raise SystemExit(f"找不到点云 {path}")
        return {path.stem: np.load(path)}

    raise SystemExit("必须提供 --dataset-root 或 --cloud")


def load_text_bank(path, class_names):
    """读取文本嵌入并校验类别顺序。"""

    data = np.load(path, allow_pickle=True)
    stored = [str(name) for name in data["class_names"]]
    if stored != list(class_names):
        raise SystemExit(
            "文本嵌入的类别顺序与 checkpoint 不一致：\n"
            f"  嵌入：{stored[:5]}\n  模型：{list(class_names)[:5]}"
        )
    embeddings = torch.from_numpy(data["embeddings"].astype(np.float32))
    return torch.nn.functional.normalize(embeddings, dim=-1)


def build_features(arrays):
    """拼成 (N, 9) 的输入特征，与训练时的构造保持一致。"""

    xyz = np.asarray(arrays["xyz"], dtype=np.float32)
    normal = np.asarray(arrays["normal"], dtype=np.float32)
    rgb = np.asarray(arrays["rgb"], dtype=np.float32) / 255.0
    return np.concatenate([xyz, normal, rgb], axis=1).astype(np.float32)


def subsample_indices(total, cap, seed=0):
    """等距抽稀索引；实例指标是 O(N²) 邻域，必须给点数上限。"""

    if cap is None or total <= cap:
        return np.arange(total)
    return np.linspace(0, total - 1, cap).round().astype(np.int64)


def main():
    args = parse_arguments()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from src.models.point_encoder import PointEncoderConfig

    config = PointEncoderConfig(**checkpoint["config"])
    model = PointEncoder(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    class_names = list(checkpoint["class_names"])
    num_classes = len(class_names)
    print(f"载入 {args.checkpoint}（epoch {checkpoint.get('epoch')}）")
    print(f"类别 {num_classes}  设备 {device}")

    text_bank = None
    if args.text_embeddings:
        text_bank = load_text_bank(args.text_embeddings, class_names).to(device)
        print(f"文本嵌入 {tuple(text_bank.shape)}")

    clouds = load_cloud_arrays(args.cloud, args.dataset_root, args.split, args.scene_limit)
    print(f"评测 {len(clouds)} 个点云：{list(clouds)}")

    report = {
        "checkpoint": args.checkpoint,
        "epoch": checkpoint.get("epoch"),
        "dataset_root": args.dataset_root,
        "split": args.split if args.dataset_root else None,
        "cloud": args.cloud,
        "num_points": args.num_points,
        "overlap": args.overlap,
        "cluster_threshold": args.cluster_threshold,
        "per_scene": {},
    }

    total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_predictions, all_targets = [], []

    for scene, arrays in clouds.items():
        features = build_features(arrays)
        logits, instance_embedding, text_embedding = encode_cloud_in_blocks(
            model, features, num_points=args.num_points, overlap=args.overlap, device=device
        )
        logits = logits.cpu().numpy()
        instance_embedding = instance_embedding.cpu().numpy()

        target = np.asarray(arrays["class"], dtype=np.int64)
        instance = np.asarray(arrays["instance"], dtype=np.int64)

        prediction = logits.argmax(axis=-1).astype(np.int64)
        entry = {
            "points": int(target.shape[0]),
            "semantic": semantic_metrics(prediction, target, num_classes, args.ignore_class),
        }
        all_predictions.append(prediction)
        all_targets.append(target)

        if text_embedding is not None and text_bank is not None:
            open_vocab = (text_embedding @ text_bank.t()).argmax(dim=-1).cpu().numpy()
            entry["open_vocab"] = semantic_metrics(
                open_vocab.astype(np.int64), target, num_classes, args.ignore_class
            )

        # 实例指标：抽稀以控制邻域开销，但抽稀是等距的（Morton 排序后更均匀）
        picked = subsample_indices(instance_embedding.shape[0], args.max_instance_points)
        sub_embedding = instance_embedding[picked]
        sub_instance = instance[picked]

        entry["instance_oracle"] = nearest_centroid_purity(
            sub_embedding, sub_instance, args.ignore_instance
        )
        clusters = knn_graph_clusters(
            sub_embedding, k=args.cluster_k, threshold=args.cluster_threshold
        )
        entry["instance_clustering"] = instance_metrics(
            clusters, sub_instance, args.ignore_instance, args.min_instance_points
        )

        report["per_scene"][scene] = entry
        message = (f"  {scene}: {entry['points']:,} 点  准确率 {entry['semantic']['accuracy']}"
                   f"  mIoU {entry['semantic']['miou']}")
        if "open_vocab" in entry:
            message += f"  开放词汇 mIoU {entry['open_vocab']['miou']}"
        oracle = entry["instance_oracle"]["purity"]
        message += f"  实例纯度(oracle) {oracle}"
        print(message, flush=True)

    # 汇总：语义指标按点数加权（把所有场景的预测拼起来重算，避免场景大小不均）
    concatenated_prediction = np.concatenate(all_predictions)
    concatenated_target = np.concatenate(all_targets)
    report["overall"] = {
        "semantic": semantic_metrics(
            concatenated_prediction, concatenated_target, num_classes, args.ignore_class
        ),
    }
    # 逐场景 mIoU 的宏平均，和按点数加权一起给——两者差异大说明小场景被淹没
    miou_values = [entry["semantic"]["miou"] for entry in report["per_scene"].values()
                   if entry["semantic"]["miou"] is not None]
    report["overall"]["macro_miou"] = round(float(np.mean(miou_values)), 4) if miou_values else None
    report["overall"]["per_class_names"] = class_names

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    overall = report["overall"]["semantic"]
    print(f"\n总体：{overall['points']:,} 点  准确率 {overall['accuracy']}  "
          f"mIoU {overall['miou']}（宏平均 {report['overall']['macro_miou']}）")
    print(f"报告：{output_path}")


if __name__ == "__main__":
    main()
