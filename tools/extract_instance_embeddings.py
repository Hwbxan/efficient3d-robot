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

    global_ids = sorted(accumulator)
    embeddings = np.stack([accumulator[g] for g in global_ids])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    np.savez(
        output_directory / "instance_embeddings.npz",
        global_ids=np.asarray(global_ids, dtype=np.int64),
        embeddings=embeddings.astype(np.float32),
    )

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
