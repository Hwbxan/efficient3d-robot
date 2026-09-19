"""离线扫描「实例置信度公式」对 AP 的影响（不重跑流水线）。

AP 是排序指标：即使检测集合完全不变，只要把真阳性排到前面就能提高 AP。
当前流水线用 Grounding DINO 的 detection_score 作为实例分数，而 DINO 分数
只反映"框里像不像这个类别"，不反映掩码质量——SAM2 自带的
sam_predicted_iou（predicted IoU head）正好补上这一维。

本脚本对同一份分割结果，用不同打分公式重算 AP，找出最合理的组合。

用法：
python -m tools.sweep_score_formula \
    --tracking-json <run>/association/tracking.json \
    --segmentation-root <run>/segmentation \
    --gt-root outputs/gt/office0 \
    --gt-classes bin,chair,door,sofa,table,tv-screen \
    --max-area-fraction 0.4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate_against_gt import average_precision, load_tracking


def load_instances(frame_index, segmentation_root, local_to_global,
                   score_threshold, max_area_fraction, image_area):
    """返回 {global_id: {"mask", "dino", "sam_iou", "area", "labels"}}。"""

    path = segmentation_root / f"frame_{frame_index:06d}_instances.json"
    if not path.exists():
        return {}

    merged = {}
    for item in json.load(path.open(encoding="utf-8")):
        if item.get("detection_score", 1.0) < score_threshold:
            continue
        if image_area:
            box = item.get("box_xyxy")
            if box:
                width = max(0.0, box[2] - box[0])
                height = max(0.0, box[3] - box[1])
                if (width * height) / image_area > max_area_fraction:
                    continue
        global_id = local_to_global.get(item["local_instance_id"])
        if global_id is None:
            continue
        mask = cv2.imread(item["mask_path"], cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        mask = mask > 0
        if not mask.any():
            continue

        dino = float(item.get("detection_score", 0.0))
        sam_iou = float(item.get("sam_predicted_iou", 1.0) or 1.0)
        label = item.get("label", "")

        if global_id in merged:
            merged[global_id]["mask"] |= mask
            merged[global_id]["dino"] = max(merged[global_id]["dino"], dino)
            # 同一 global_id 的多个观测取 SAM 预测 IoU 的中位，避免单个离群值
            merged[global_id]["sam_list"].append(sam_iou)
            merged[global_id]["labels"].add(label)
        else:
            merged[global_id] = {
                "mask": mask,
                "dino": dino,
                "sam_list": [sam_iou],
                "labels": {label},
            }

    for value in merged.values():
        value["sam"] = float(np.median(value["sam_list"]))
        value["area"] = int(value["mask"].sum())
    return merged


FORMULAS = {
    "dino": lambda v: v["dino"],
    "sam": lambda v: v["sam"],
    "dino*sam": lambda v: v["dino"] * v["sam"],
    "dino*sqrt(sam)": lambda v: v["dino"] * np.sqrt(max(v["sam"], 0.0)),
    "dino*sam^2": lambda v: v["dino"] * v["sam"] ** 2,
    "dino^0.5*sam": lambda v: np.sqrt(max(v["dino"], 0.0)) * v["sam"],
    "dino*sam*area": lambda v: v["dino"] * v["sam"] * min(1.0, v["area"] / 20000.0),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-json", required=True)
    parser.add_argument("--segmentation-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--gt-classes", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-gt-area", type=int, default=100)
    parser.add_argument("--max-area-fraction", type=float, default=0.4)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--image-width", type=int, default=1200)
    parser.add_argument("--image-height", type=int, default=680)
    args = parser.parse_args()

    image_area = args.image_width * args.image_height
    gt_root = Path(args.gt_root)
    manifest = json.load((gt_root / "gt_manifest.json").open(encoding="utf-8"))
    object_to_class = {int(k): v for k, v in manifest["object_id_to_class"].items()}
    class_filter = (
        {item.strip() for item in args.gt_classes.split(",") if item.strip()}
        if args.gt_classes
        else None
    )

    tracking = load_tracking(Path(args.tracking_json))
    frames = sorted(tracking)

    # 先缓存每帧的 (预测, GT, 配对 IoU 矩阵)
    cache = []
    total_gt = 0
    for frame_index in frames:
        gt_image = cv2.imread(
            str(gt_root / "instance_masks" / f"instance{frame_index:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if gt_image is None:
            continue
        gt_ids = [
            int(value)
            for value in np.unique(gt_image)
            if value != 0
            and (gt_image == value).sum() >= args.min_gt_area
            and (class_filter is None or object_to_class.get(int(value)) in class_filter)
        ]
        total_gt += len(gt_ids)
        predictions = load_instances(
            frame_index,
            Path(args.segmentation_root),
            tracking[frame_index],
            args.min_score,
            args.max_area_fraction,
            image_area,
        )
        cache.append((frame_index, predictions, gt_image, gt_ids))

    print(f"帧数 {len(cache)}  GT 实例 {total_gt}\n")

    results = {}
    print(f"{'公式':<18}" + "".join(f"  AP@{t:<5}" for t in args.iou_thresholds))
    for name, formula in FORMULAS.items():
        records = {t: [] for t in args.iou_thresholds}
        for _, predictions, gt_image, gt_ids in cache:
            gt_masks = {gt_id: gt_image == gt_id for gt_id in gt_ids}
            # 贪心匹配（与 evaluate_against_gt 一致）建立 global_id → best_iou
            best = {}
            for global_id, value in predictions.items():
                best_iou = 0.0
                for gt_id, gt_mask in gt_masks.items():
                    intersection = int(np.logical_and(value["mask"], gt_mask).sum())
                    union = value["area"] + int(gt_mask.sum()) - intersection
                    if union:
                        best_iou = max(best_iou, intersection / union)
                best[global_id] = best_iou
            for global_id, value in predictions.items():
                score = float(formula(value))
                for threshold in args.iou_thresholds:
                    records[threshold].append((score, best[global_id]))

        row = {}
        for threshold in args.iou_thresholds:
            stats = average_precision(records[threshold], total_gt, threshold)
            row[f"ap_{threshold}"] = stats["ap"]
        results[name] = row
        line = f"{name:<18}" + "".join(
            f"  {row[f'ap_{t}']:<9.4f}" for t in args.iou_thresholds
        )
        print(line)

    best_name = max(results, key=lambda k: sum(results[k].values()))
    print(f"\n最佳（AP 之和）: {best_name} -> {results[best_name]}")

    # ---- 第二阶段：归一化线性融合 ----
    # dino 与 sam 的量纲/方差差异极大（dino 约 0.3~0.9，sam 约 0.8~0.99），
    # 直接相乘时 dino 会完全主导排序。先把两者做 min-max 归一化再加权求和，
    # 才能真正体现"类别可信度 x 掩码可信度"的互补。
    pool = []
    for _, predictions, _, _ in cache:
        for value in predictions.values():
            pool.append((value["dino"], value["sam"]))

    print("\n【归一化融合】score = w * norm(dino) + (1-w) * norm(sam)")
    print(f"{'w':<8}{'AP@0.25':<12}{'AP@0.5':<12}{'之和':<10}")

    dino_values = np.array([item[0] for item in pool])
    sam_values = np.array([item[1] for item in pool])
    print(
        f"  dino 范围 [{dino_values.min():.3f}, {dino_values.max():.3f}]  "
        f"sam 范围 [{sam_values.min():.3f}, {sam_values.max():.3f}]"
    )

    d_low, d_high = float(dino_values.min()), float(dino_values.max())
    s_low, s_high = float(sam_values.min()), float(sam_values.max())
    d_span = max(d_high - d_low, 1e-6)
    s_span = max(s_high - s_low, 1e-6)

    blend_results = {}
    for weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        records = {t: [] for t in args.iou_thresholds}
        for _, predictions, gt_image, gt_ids in cache:
            gt_masks = {gt_id: gt_image == gt_id for gt_id in gt_ids}
            best = {}
            for global_id, value in predictions.items():
                best_iou = 0.0
                for gt_id, gt_mask in gt_masks.items():
                    intersection = int(np.logical_and(value["mask"], gt_mask).sum())
                    union = value["area"] + int(gt_mask.sum()) - intersection
                    if union:
                        best_iou = max(best_iou, intersection / union)
                best[global_id] = best_iou
            for global_id, value in predictions.items():
                score = (
                    weight * (value["dino"] - d_low) / d_span
                    + (1.0 - weight) * (value["sam"] - s_low) / s_span
                )
                for threshold in args.iou_thresholds:
                    records[threshold].append((float(score), best[global_id]))

        row = {}
        for threshold in args.iou_thresholds:
            row[f"ap_{threshold}"] = average_precision(
                records[threshold], total_gt, threshold
            )["ap"]
        blend_results[f"w={weight}"] = row
        total = sum(row.values())
        print(
            f"{weight:<8.1f}{row[f'ap_{args.iou_thresholds[0]}']:<12.4f}"
            f"{row[f'ap_{args.iou_thresholds[-1]}']:<12.4f}{total:<10.4f}"
        )

    best_blend = max(blend_results, key=lambda k: sum(blend_results[k].values()))
    print(f"\n最佳融合: {best_blend} -> {blend_results[best_blend]}")
    results.update(blend_results)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump({"total_gt": total_gt, "results": results,
                       "best": best_name, "best_blend": best_blend,
                       "normalization": {
                           "dino": [d_low, d_high], "sam": [s_low, s_high]},
                       }, file, indent=2, ensure_ascii=False)
        print(f"报告: {args.output}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump({"total_gt": total_gt, "results": results,
                       "best": best_name}, file, indent=2, ensure_ascii=False)
        print(f"报告: {args.output}")


if __name__ == "__main__":
    main()
