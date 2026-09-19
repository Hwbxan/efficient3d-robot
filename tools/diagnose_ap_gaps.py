"""定位 AP 损失的构成：漏检 / 掩码不准 / 虚假检测。

对每一帧把预测掩码（按 tracking 分配的 global_id 合并）与 GT 实例掩码做
全配对 IoU（不做贪心去重），统计：

1. 漏检（FN）：GT 实例的最佳 IoU 分布 → 区分「完全没检测到」与
   「检测到了但掩码重叠太低」。
2. 虚警（FP）：预测实例的最佳 IoU 分布 → 区分「纯背景误检」与
   「框到了物体但掩码质量差」。
3. 掩码尺度：已匹配配对的 mask/GT 面积比分布 → 判断系统性过覆盖还是欠覆盖。

用法（项目根目录）：
python -m tools.diagnose_ap_gaps \
    --tracking-json <run>/association/tracking.json \
    --segmentation-root <run>/segmentation \
    --gt-root outputs/gt/office0 \
    --gt-classes bin,chair,door,sofa,table,tv-screen \
    --max-area-fraction 0.4
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate_against_gt import load_predictions, load_tracking


def percentile(values, fraction):
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def bucketize(values, edges):
    counts = Counter()
    for value in values:
        for edge in edges:
            if value < edge:
                counts[edge] += 1
                break
        else:
            counts[float("inf")] += 1
    buckets = []
    lower = 0.0
    for edge in edges:
        buckets.append([lower, edge, int(counts[edge])])
        lower = edge
    buckets.append([lower, float("inf"), int(counts[float("inf")])])
    return buckets


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
    parser.add_argument("--image-width", type=int, default=1200)
    parser.add_argument("--image-height", type=int, default=680)
    args = parser.parse_args()

    image_area = args.image_width * args.image_height
    gt_root = Path(args.gt_root)
    manifest = json.load(
        (gt_root / "gt_manifest.json").open(encoding="utf-8")
    )
    object_to_class = {
        int(key): value for key, value in manifest["object_id_to_class"].items()
    }
    class_filter = None
    if args.gt_classes:
        class_filter = {
            item.strip() for item in args.gt_classes.split(",") if item.strip()
        }

    tracking = load_tracking(Path(args.tracking_json))
    frames = sorted(tracking)

    gt_best = []
    pred_best = []
    matched_ratios = []
    matched_ious = []
    fp_labels = Counter()
    fn_labels = Counter()
    per_frame = {}

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
            and (
                class_filter is None
                or object_to_class.get(int(value)) in class_filter
            )
        ]

        predictions = load_predictions(
            frame_index,
            Path(args.segmentation_root),
            tracking[frame_index],
            score_threshold=args.min_score,
            max_area_fraction=args.max_area_fraction,
            image_area=image_area,
        )

        pred_keys = list(predictions)
        gt_masks = [gt_image == gt_id for gt_id in gt_ids]
        gt_areas = [int(mask.sum()) for mask in gt_masks]
        pred_masks = [predictions[key][0] for key in pred_keys]
        pred_areas = [int(mask.sum()) for mask in pred_masks]

        if not pred_keys or not gt_ids:
            per_frame[frame_index] = {
                "predictions": len(pred_keys),
                "gt": len(gt_ids),
            }
            for mask, key in zip(pred_masks, pred_keys):
                pred_best.append(0.0)
                labels = sorted(predictions[key][2])
                fp_labels[labels[0] if labels else "?"] += 1
            for gt_id, area in zip(gt_ids, gt_areas):
                gt_best.append(0.0)
                fn_labels[object_to_class.get(int(gt_id), "?")] += 1
            continue

        matrix = np.zeros((len(pred_keys), len(gt_ids)))
        for row, pred_mask in enumerate(pred_masks):
            for column, gt_mask in enumerate(gt_masks):
                intersection = int(np.logical_and(pred_mask, gt_mask).sum())
                union = pred_areas[row] + gt_areas[column] - intersection
                matrix[row, column] = intersection / union if union else 0.0

        for column, gt_id in enumerate(gt_ids):
            best = float(matrix[:, column].max())
            gt_best.append(best)
            if best < 0.25:
                fn_labels[object_to_class.get(int(gt_id), "?")] += 1

        for row, key in enumerate(pred_keys):
            best = float(matrix[row].max())
            pred_best.append(best)
            column = int(np.argmax(matrix[row]))
            if best >= 0.25:
                matched_ratios.append(pred_areas[row] / max(1, gt_areas[column]))
                matched_ious.append(best)
            else:
                labels = sorted(predictions[key][2])
                fp_labels[labels[0] if labels else "?"] += 1

        per_frame[frame_index] = {
            "predictions": len(pred_keys),
            "gt": len(gt_ids),
        }

    total_gt = len(gt_best)
    total_pred = len(pred_best)

    report = {
        "frames": len(frames),
        "gt_instances": total_gt,
        "predictions": total_pred,
        "gt_best_iou": {
            "buckets": bucketize(gt_best, [0.01, 0.1, 0.25, 0.5, 0.75]),
            "recall_at_0.25": sum(1 for v in gt_best if v >= 0.25) / max(1, total_gt),
            "recall_at_0.5": sum(1 for v in gt_best if v >= 0.5) / max(1, total_gt),
        },
        "pred_best_iou": {
            "buckets": bucketize(pred_best, [0.01, 0.1, 0.25, 0.5, 0.75]),
            "precision_at_0.25": sum(1 for v in pred_best if v >= 0.25) / max(1, total_pred),
        },
        "mask_scale": {
            "n_matched": len(matched_ratios),
            "ratio_p10": percentile(matched_ratios, 0.10),
            "ratio_p50": percentile(matched_ratios, 0.50),
            "ratio_p90": percentile(matched_ratios, 0.90),
            "frac_gt_1.5": sum(1 for v in matched_ratios if v > 1.5) / max(1, len(matched_ratios)),
            "frac_lt_0.67": sum(1 for v in matched_ratios if v < 0.67) / max(1, len(matched_ratios)),
            "matched_iou_median": percentile(matched_ious, 0.5),
            "matched_iou_p90": percentile(matched_ious, 0.9),
        },
        "fp_labels": dict(fp_labels.most_common()),
        "fn_labels": dict(fn_labels.most_common()),
        "per_frame": per_frame,
    }

    print(f"帧数 {len(frames)}  GT {total_gt}  预测 {total_pred}\n")

    print("【漏检 FN】GT 实例的最佳 IoU 分布")
    for lower, upper, count in report["gt_best_iou"]["buckets"]:
        upper_text = "inf" if upper == float("inf") else f"{upper:g}"
        print(f"  [{lower:g}, {upper_text})  {count:4d}  ({count / max(1, total_gt):.1%})")
    print(f"  Recall@0.25 = {report['gt_best_iou']['recall_at_0.25']:.3f}   "
          f"Recall@0.5 = {report['gt_best_iou']['recall_at_0.5']:.3f}")
    print(f"  按类别: {report['fn_labels']}\n")

    print("【虚警 FP】预测实例的最佳 IoU 分布")
    for lower, upper, count in report["pred_best_iou"]["buckets"]:
        upper_text = "inf" if upper == float("inf") else f"{upper:g}"
        print(f"  [{lower:g}, {upper_text})  {count:4d}  ({count / max(1, total_pred):.1%})")
    print(f"  Precision@0.25 = {report['pred_best_iou']['precision_at_0.25']:.3f}")
    print(f"  按类别: {report['fp_labels']}\n")

    scale = report["mask_scale"]
    print("【掩码尺度】已匹配(IoU>=0.25)配对的 mask/GT 面积比")
    print(f"  样本 {scale['n_matched']}  中位 {scale['ratio_p50']:.2f}  "
          f"p10 {scale['ratio_p10']:.2f}  p90 {scale['ratio_p90']:.2f}")
    print(f"  过覆盖 >1.5x: {scale['frac_gt_1.5']:.1%}   "
          f"欠覆盖 <0.67x: {scale['frac_lt_0.67']:.1%}")
    print(f"  已匹配配对 IoU: 中位 {scale['matched_iou_median']:.3f}  "
          f"p90 {scale['matched_iou_p90']:.3f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        print(f"\n报告: {args.output}")


if __name__ == "__main__":
    main()
