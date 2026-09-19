"""用 GT 实例掩码评测关联结果。

对每一帧：把检测掩码按 tracking.json 分配的 global_id 合并成
"预测实例"，与该帧 GT 实例掩码做贪心 IoU 匹配（按 IoU 降序）。

指标：
- AP@IoU：按检测分数排序的帧级实例检测精度（类别无关，
  VOC 式逐步插值）；同时给出各阈值的 Precision / Recall。
- 关联质量（在 --association-iou 阈值下统计）：
  fragmentation：一个 GT 物体被多少个不同 track id 覆盖（1 为完美）；
  purity：一个 track id 吸收了多少个不同 GT 物体（1 为完美）。

用法（在项目根目录）：
python -m tools.evaluate_against_gt \
    --tracking-json outputs/experiments/xxx/tracking.json \
    --segmentation-root outputs/experiments/xxx/segmentation \
    --gt-root outputs/gt/office0 \
    --output outputs/experiments/xxx/gt_evaluation.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LABEL_SYNONYMS = {
    "trash can": {"bin", "tissue-paper"},
    "computer monitor": {"tv-screen", "tablet"},
    "chair": {"chair", "sofa"},
    "desk": {"table", "desk-organizer", "panel"},
    "door": {"door"},
    "sofa": {"sofa"},
    "couch": {"sofa"},
    "table": {"table"},
    "plant": {"indoor-plant", "plant-stand"},
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-json", required=True)
    parser.add_argument("--segmentation-root", required=True, help="含 frame_*_instances.json 与 frame_*_masks/")
    parser.add_argument("--gt-root", required=True, help="GT 根目录（instance_masks/ 与 gt_manifest.json）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--association-iou", type=float, default=0.5)
    parser.add_argument("--min-gt-area", type=int, default=100, help="GT 物体计入评测的最小像素面积")
    parser.add_argument(
        "--gt-classes",
        default=None,
        help="逗号分隔的 GT 类别白名单；只评测这些类别（默认全部，含墙/地板等结构性物体）",
    )
    return parser.parse_args()


def load_tracking(path):
    """返回 {frame_index: {local_instance_id: global_id}}。"""

    data = json.load(open(path, encoding="utf-8"))
    frames = {}
    for frame in data["frames"]:
        index = frame["frame_index"]
        frames[index] = {
            item["local_instance_id"]: item.get("global_id")
            for item in frame["associations"]
        }
    return frames


def load_predictions(frame_index, segmentation_root, local_to_global):
    """合并同 global_id 的检测掩码，返回 {global_id: (mask, score, labels)}。"""

    instances_path = segmentation_root / f"frame_{frame_index:06d}_instances.json"
    if not instances_path.exists():
        return {}

    instances = json.load(open(instances_path, encoding="utf-8"))
    merged = {}
    for item in instances:
        global_id = local_to_global.get(item["local_instance_id"])
        if global_id is None:
            continue
        mask = cv2.imread(item["mask_path"], cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        mask = mask > 0
        if not mask.any():
            continue
        if global_id in merged:
            previous, score, labels = merged[global_id]
            merged[global_id] = (
                previous | mask,
                max(score, item.get("detection_score", 0.0)),
                labels | {item.get("label", "")},
            )
        else:
            merged[global_id] = (
                mask,
                item.get("detection_score", 0.0),
                {item.get("label", "")},
            )
    return merged


def greedy_match(predictions, gt_ids, gt_image):
    """贪心 IoU 匹配，返回 [(iou, global_id, gt_id)]。

    只考虑传入的 gt_ids（已按面积与类别白名单过滤），
    否则会出现"匹配到未计入分母的 GT 物体"导致 TP > 分母。
    """

    allowed = set(int(value) for value in gt_ids)
    candidates = []
    for global_id, (mask, _, _) in predictions.items():
        overlapping = np.unique(gt_image[mask])
        for gt_id in overlapping:
            gt_id = int(gt_id)
            if gt_id not in allowed:
                continue
            gt_mask = gt_image == gt_id
            union = np.logical_or(mask, gt_mask).sum()
            if union == 0:
                continue
            iou = np.logical_and(mask, gt_mask).sum() / union
            if iou > 0:
                candidates.append((float(iou), global_id, gt_id))

    candidates.sort(reverse=True)
    used_predictions = set()
    used_gt = set()
    matches = []
    for iou, global_id, gt_id in candidates:
        if global_id in used_predictions or gt_id in used_gt:
            continue
        used_predictions.add(global_id)
        used_gt.add(gt_id)
        matches.append((iou, global_id, gt_id))
    return matches


def average_precision(records, total_gt, threshold):
    """records: [(score, matched_iou, ...)]，未匹配 matched_iou=0；VOC 式 AP。"""

    ordered = sorted(records, key=lambda item: -item[0])
    true_positives = 0
    false_positives = 0
    precision_sum = 0.0
    for record in ordered:
        matched_iou = record[1]
        if matched_iou >= threshold:
            true_positives += 1
            precision_sum += true_positives / (true_positives + false_positives)
        else:
            false_positives += 1
    ap = precision_sum / total_gt if total_gt else 0.0

    matched = sum(1 for record in ordered if record[1] >= threshold)
    precision = matched / len(ordered) if ordered else 0.0
    recall = matched / total_gt if total_gt else 0.0
    return {
        "ap": round(ap, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": matched,
        "predictions": len(ordered),
        "gt_instances": total_gt,
    }


def main():
    args = parse_arguments()

    tracking = load_tracking(args.tracking_json)
    segmentation_root = Path(args.segmentation_root)
    gt_root = Path(args.gt_root)
    manifest = json.load(open(gt_root / "gt_manifest.json", encoding="utf-8"))
    object_to_class = {int(k): v for k, v in manifest["object_id_to_class"].items()}
    class_filter = None
    if args.gt_classes:
        class_filter = {name.strip() for name in args.gt_classes.split(",") if name.strip()}

    end_frame = args.end_frame if args.end_frame is not None else max(tracking)
    frames = sorted(
        index for index in tracking
        if args.start_frame <= index <= end_frame
        and (gt_root / "instance_masks" / f"instance{index:06d}.png").exists()
    )

    all_records = []  # (score, matched_iou, frame, global_id, gt_id, labels)
    total_gt = 0
    unassigned = 0
    gt_to_tracks = {}
    track_to_gts = {}
    label_consistent = 0
    label_total = 0
    frame_summaries = {}

    for frame_index in frames:
        gt_image = cv2.imread(
            str(gt_root / "instance_masks" / f"instance{frame_index:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if gt_image is None:
            continue

        gt_ids = [
            int(value) for value in np.unique(gt_image)
            if value != 0
            and (gt_image == value).sum() >= args.min_gt_area
            and (
                class_filter is None
                or object_to_class.get(int(value)) in class_filter
            )
        ]
        total_gt += len(gt_ids)

        local_to_global = tracking[frame_index]
        predictions = load_predictions(frame_index, segmentation_root, local_to_global)
        unassigned += sum(1 for gid in local_to_global.values() if gid is None)

        matches = greedy_match(predictions, gt_ids, gt_image)
        matched_globals = {global_id for _, global_id, _ in matches}

        for global_id, (mask, score, labels) in predictions.items():
            matched = next(((iou, gt_id) for iou, g, gt_id in matches if g == global_id), None)
            matched_iou = matched[0] if matched else 0.0
            gt_id = matched[1] if matched else None
            all_records.append((score, matched_iou, frame_index, global_id, gt_id, labels))

            if matched and matched_iou >= args.association_iou:
                gt_to_tracks.setdefault(gt_id, set()).add(global_id)
                track_to_gts.setdefault(global_id, set()).add(gt_id)
                label_total += 1
                expected = set()
                for label in labels:
                    expected |= LABEL_SYNONYMS.get(label, {label})
                if object_to_class.get(gt_id) in expected:
                    label_consistent += 1

        frame_summaries[str(frame_index)] = {
            "gt_instances": len(gt_ids),
            "predictions": len(predictions),
            "matched_at_half": sum(1 for iou, _, _ in matches if iou >= 0.5),
        }

    detection = {
        f"iou_{threshold:g}": average_precision(all_records, total_gt, threshold)
        for threshold in args.iou_thresholds
    }

    fragmentation = {
        gt_id: sorted(tracks) for gt_id, tracks in gt_to_tracks.items() if len(tracks) > 1
    }
    purity_issues = {
        global_id: sorted(gts) for global_id, gts in track_to_gts.items() if len(gts) > 1
    }
    matched_gt_objects = len(gt_to_tracks)
    matched_tracks = len(track_to_gts)
    single_track_gt = sum(1 for tracks in gt_to_tracks.values() if len(tracks) == 1)
    pure_tracks = sum(1 for gts in track_to_gts.values() if len(gts) == 1)

    report = {
        "tracking_json": args.tracking_json,
        "segmentation_root": str(segmentation_root),
        "gt_root": str(gt_root),
        "evaluated_frames": len(frames),
        "frame_range": [frames[0], frames[-1]] if frames else [],
        "gt_class_filter": sorted(class_filter) if class_filter else None,
        "min_gt_area": args.min_gt_area,
        "unassigned_observations": unassigned,
        "detection": detection,
        "association": {
            "iou_threshold": args.association_iou,
            "matched_gt_objects": matched_gt_objects,
            "matched_tracks": matched_tracks,
            "gt_objects_with_single_track": single_track_gt,
            "single_track_ratio": round(single_track_gt / matched_gt_objects, 4) if matched_gt_objects else None,
            "pure_track_ratio": round(pure_tracks / matched_tracks, 4) if matched_tracks else None,
            "fragmentation_mean": round(
                sum(len(t) for t in gt_to_tracks.values()) / matched_gt_objects, 4
            ) if matched_gt_objects else None,
            "fragmented_gt_objects": {
                object_to_class.get(int(gt), str(gt)): {"gt_id": int(gt), "track_ids": tracks}
                for gt, tracks in sorted(fragmentation.items())
            },
            "impure_tracks": {
                str(global_id): {
                    "gt_classes": [object_to_class.get(int(g), str(g)) for g in gts]
                }
                for global_id, gts in sorted(purity_issues.items())
            },
        },
        "label_consistency": {
            "consistent": label_consistent,
            "total": label_total,
            "ratio": round(label_consistent / label_total, 4) if label_total else None,
        },
        "frames": frame_summaries,
        "note": "AP 为类别无关的帧级实例检测精度；关联指标在 association.iou_threshold 下统计。",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"帧数 {len(frames)}（{frames[0]}–{frames[-1]}）  GT 实例 {total_gt}  未分配观测 {unassigned}")
    for key, value in detection.items():
        print(
            f"{key}: AP={value['ap']:.4f}  P={value['precision']:.4f}  R={value['recall']:.4f}"
            f"  (TP={value['tp']}/{value['predictions']})"
        )
    print(
        f"关联: 匹配 GT 物体 {matched_gt_objects} 个，单轨覆盖 {single_track_gt} 个"
        f"（{report['association']['single_track_ratio']}），"
        f"纯轨道比例 {report['association']['pure_track_ratio']}"
    )
    if label_total:
        print(f"标签一致性: {label_consistent}/{label_total} = {report['label_consistency']['ratio']}")
    print(f"报告: {output_path}")


if __name__ == "__main__":
    main()
