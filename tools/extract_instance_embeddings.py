"""为融合后的 3D 实例提取 CLIP 语义嵌入。

做法（OpenMask3D 式）：对每一帧，按 tracking.json 的 local→global 映射
取出该实例的 2D 掩码，裁剪 RGB、抠掉背景，送入 CLIP 图像编码器；
再按掩码面积加权平均，得到每个 global_id 一个 L2 归一化向量。

输出：
- instance_embeddings.npz   global_ids + embeddings（已 L2 归一化）
- instance_embeddings.json  元数据：标签、观测数、使用的帧、累计像素

用法（在项目根目录）：
python -m tools.extract_instance_embeddings \
    --run-directory outputs/experiments/office0_fixed_detector_v3 \
    --scene-directory datasets/processed/Replica/office0 \
    --output-directory outputs/experiments/office0_fixed_detector_v3/embeddings
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.datasets.replica_sequence import ReplicaSequence
from src.perception.open_vocab_encoder import OpenVocabularyEncoder


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--scene-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--min-mask-area", type=int, default=200, help="小于该面积的掩码不参与聚合")
    parser.add_argument("--context-ratio", type=float, default=0.10, help="裁剪框外扩比例")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classes-file", default=None,
                        help="JSON 列表或 [[组...],...]，提供类别短语；"
                             "给出后启用逐 crop 投票与精炼嵌入")
    return parser.parse_args()


def load_tracking(path):
    data = json.load(open(path, encoding="utf-8"))
    frames = {}
    for frame in data["frames"]:
        index = frame["frame_index"]
        frames[index] = {
            item["local_instance_id"]: item.get("global_id")
            for item in frame["associations"]
        }
    return frames


def crop_with_mask(rgb, mask, context_ratio):
    """按掩码包围盒裁剪并抠掉背景，返回裁剪后的 RGB 与掩码面积。"""

    rows = np.any(mask, axis=1)
    columns = np.any(mask, axis=0)
    if not rows.any() or not columns.any():
        return None, 0

    top, bottom = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    left, right = int(np.argmax(columns)), int(len(columns) - np.argmax(columns[::-1]))

    height, width = mask.shape
    pad_y = int((bottom - top) * context_ratio)
    pad_x = int((right - left) * context_ratio)
    top = max(0, top - pad_y)
    bottom = min(height, bottom + pad_y)
    left = max(0, left - pad_x)
    right = min(width, right + pad_x)

    crop = rgb[top:bottom, left:right].copy()
    crop_mask = mask[top:bottom, left:right]
    crop[~crop_mask] = 0
    return crop, int(mask.sum())


def collect_crops(args, run_directory):
    """扫描所有帧，收集 (global_id, PIL 裁剪图, 面积) 三元组。"""

    tracking = load_tracking(run_directory / "association" / "tracking.json")
    instance_map = json.load(
        open(run_directory / "fusion_attempt_01" / "instance_map.json", encoding="utf-8")
    )
    fused_labels = {
        item["global_id"]: item.get("label", "unknown") for item in instance_map["instances"]
    }

    sequence = ReplicaSequence(args.scene_directory)
    index_by_frame = {int(sequence.rgb_paths[i].stem[5:]): i for i in range(len(sequence))}
    segmentation_root = run_directory / "segmentation"

    records = []
    used_frames = {}
    skipped_small = 0

    for frame_index in sorted(tracking):
        instances_path = segmentation_root / ("frame_%06d_instances.json" % frame_index)
        if not instances_path.exists() or frame_index not in index_by_frame:
            continue

        rgb = sequence[index_by_frame[frame_index]]["rgb"]
        for item in json.load(open(instances_path, encoding="utf-8")):
            global_id = tracking[frame_index].get(item["local_instance_id"])
            if global_id is None:
                continue
            mask = cv2.imread(item["mask_path"], cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            mask = mask > 0

            crop, area = crop_with_mask(rgb, mask, args.context_ratio)
            if crop is None or area < args.min_mask_area:
                skipped_small += 1
                continue

            records.append((global_id, Image.fromarray(crop), area))
            used_frames.setdefault(global_id, []).append(frame_index)

    return records, used_frames, fused_labels, skipped_small


def main():
    args = parse_arguments()

    run_directory = Path(args.run_directory)
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # PATCH_CLIP_VOTE_V1：加载类别短语表
    class_names = []
    if args.classes_file:
        data = json.loads(Path(args.classes_file).read_text(encoding="utf-8"))
        if data and isinstance(data[0], list):
            for group in data:
                for c in group:
                    if c not in class_names:
                        class_names.append(c)
        else:
            class_names = list(data)
        print("类别短语 %d 个，启用逐 crop 投票" % len(class_names))

    records, used_frames, fused_labels, skipped_small = collect_crops(args, run_directory)
    if not records:
        raise RuntimeError("没有收集到任何实例裁剪图")
    print("收集到 %d 张裁剪图，覆盖 %d 个实例" % (len(records), len(used_frames)))

    encoder = OpenVocabularyEncoder(args.clip_model, device=args.device)
    print("CLIP 模型：%s  设备：%s  维度：%d"
          % (args.clip_model, encoder.device, encoder.embedding_dim))

    features = encoder.encode_images([r[1] for r in records], batch_size=args.batch_size).numpy()

    accumulator = {}
    pixel_area = {}
    for (global_id, _, area), vector in zip(records, features):
        if global_id not in accumulator:
            accumulator[global_id] = np.zeros_like(vector)
            pixel_area[global_id] = 0
        accumulator[global_id] += vector * area
        pixel_area[global_id] += area

    # PATCH_CLIP_VOTE_V1：逐 crop 投票 + 只聚合投给多数类的 crop
    vote_label = {}
    vote_hist = {}
    refined = {}
    refined_area = {}
    if class_names:
        text = encoder.encode_texts(["a photo of a %s" % c for c in class_names])
        text = np.asarray(text, dtype=np.float32).reshape(len(class_names), -1)
        text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-8)
        feats = np.asarray(features, dtype=np.float32)
        feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
        sims = feats @ text.T                       # (n_crop, n_class)
        top = np.argmax(sims, axis=1)
        for (global_id, _, area), k in zip(records, top):
            hist = vote_hist.setdefault(global_id, {})
            hist[class_names[int(k)]] = hist.get(class_names[int(k)], 0.0) + area
        for gid, hist in vote_hist.items():
            vote_label[gid] = max(hist.items(), key=lambda kv: kv[1])[0]
        for (global_id, _, area), vector, k in zip(records, features, top):
            if class_names[int(k)] != vote_label.get(global_id):
                continue
            if global_id not in refined:
                refined[global_id] = np.zeros_like(vector)
                refined_area[global_id] = 0.0
            refined[global_id] += vector * area
            refined_area[global_id] += area
        print("投票完成：%d 个实例；精炼嵌入覆盖 %d 个"
              % (len(vote_label), len(refined)))

    global_ids = sorted(accumulator)
    embeddings = np.stack([accumulator[g] for g in global_ids])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    # PATCH_CLIP_VOTE_V1：原始嵌入 + 精炼嵌入一起存
    save_kwargs = dict(
        global_ids=np.asarray(global_ids, dtype=np.int64),
        embeddings=embeddings.astype(np.float32),
    )
    if refined:
        ref = np.stack([refined.get(g, accumulator[g]) for g in global_ids])
        ref = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-8)
        save_kwargs["embeddings_refined"] = ref.astype(np.float32)
    np.savez(output_directory / "instance_embeddings.npz", **save_kwargs)

    metadata = {
        "clip_model": args.clip_model,
        "min_mask_area": args.min_mask_area,
        "context_ratio": args.context_ratio,
        "skipped_small_masks": skipped_small,
        "crops_used": len(records),
        "instance_count": len(global_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "instances": [
            {
                "global_id": int(global_id),
                "label": fused_labels.get(global_id, "unknown"),
                "crops_used": len(used_frames[global_id]),
                "pixel_area": int(pixel_area[global_id]),
                "frames": used_frames[global_id],
                # PATCH_CLIP_VOTE_V1
                "clip_label": vote_label.get(global_id),
                "clip_votes": {k: int(v) for k, v in sorted(
                    (vote_hist.get(global_id) or {}).items(),
                    key=lambda kv: -kv[1])[:5]},
            }
            for global_id in global_ids
        ],
    }
    with open(output_directory / "instance_embeddings.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("完成：%d 个实例，维度 %d，跳过小掩码 %d"
          % (len(global_ids), embeddings.shape[1], skipped_small))
    print("输出：%s" % (output_directory / "instance_embeddings.npz"))


if __name__ == "__main__":
    main()
