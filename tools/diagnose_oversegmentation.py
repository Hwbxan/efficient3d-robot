"""过分割 / 误合并根因诊断。

**不重跑流水线**，只吃三份已有产物：

- `tracking.json` —— 每次观测的 `global_id` / `association_cost` / `decision`
- `segmentation/frame_*_instances.json` + `mask_path` —— 2D 检测掩码
- `outputs/gt/<scene>/instance_masks/instance%06d.png` —— GT 实例掩码

要回答的问题是：**一个 GT 物体为什么被拆成多条轨道？**

对每个 GT 物体逐帧找出"最负责"的那条预测（与它 IoU 最高的掩码），
沿时间轴把负责的 global_id 切成若干段，然后给每次"换轨"归因：

| 归因 | 判据 | 指向的修复方向 |
|---|---|---|
| `detection_gap` | 换轨前有若干帧**该物体仍有 GT 标注但没检到** | 2D 漏检 / SAM2 漏分割（可修） |
| `occlusion_gap` | 换轨前有若干帧**GT 根本没标注该物体** | 物体不在视野内，属合理断档；只能靠轨迹记忆 / 运动模型 |
| `association_no_candidate` | 无断档，新轨以 `new_tentative` 起步且 `association_cost` 为空 | 3D 几何代价 / 阈值（`unmatched_cost`） |
| `association_cost_high` | 无断档，新轨起步时带着接近阈值的代价 | 同上，属"差一点匹配上" |
| `association_conflict` | 新轨起步决策是 `deferred_conflict` | 关联冲突仲裁 |
| `other` | 其余 | 待人工看 |

另外输出掩码质量分布与轨道纯度，用于判断"是不是 2D 掩码本身就烂"。

用法（在项目根目录）：
python -m tools.diagnose_oversegmentation \
    --tracking-json outputs/experiments/office0_filtered_v4/tracking.json \
    --segmentation-root outputs/experiments/office0_filtered_v4/segmentation \
    --gt-root outputs/gt/office0 \
    --output outputs/experiments/office0_filtered_v4/overseg_diagnosis.json \
    --gt-classes chair,sofa,table,tv-screen,bin,door
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# 与 evaluate_against_gt.py 保持一致：结构性物体（墙/地板/天花板）不计入。
DEFAULT_GT_CLASSES = "chair,sofa,table,tv-screen,bin,door"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-json", required=True)
    parser.add_argument("--segmentation-root", required=True,
                        help="含 frame_*_instances.json 与掩码文件")
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--min-gt-area", type=int, default=100,
                        help="GT 物体计入诊断的最小像素面积")
    parser.add_argument("--iou-threshold", type=float, default=0.25,
                        help="认定「该帧检到了这个物体」的 IoU 下限")
    parser.add_argument("--gt-classes", default=DEFAULT_GT_CLASSES,
                        help="逗号分隔的 GT 类别白名单")
    parser.add_argument("--near-threshold-cost", type=float, default=0.65,
                        help="关联代价的 unmatched_cost 阈值，用于判断'差一点'")
    return parser.parse_args()


def load_tracking(path):
    """返回 (per_frame, tracks)。

    per_frame: {frame_index: {local_instance_id: {global_id, cost, decision, label}}}
    tracks:    {global_id: {...}} 便于查轨道存活区间
    """

    data = json.load(open(path, encoding="utf-8"))
    per_frame = {}
    for frame in data["frames"]:
        index = int(frame["frame_index"])
        per_frame[index] = {
            int(item["local_instance_id"]): {
                "global_id": item.get("global_id"),
                "cost": item.get("association_cost"),
                "decision": item.get("decision"),
                "label": item.get("raw_label"),
            }
            for item in frame["associations"]
        }
    tracks = {int(t["global_id"]): t for t in data.get("tracks", [])}
    return per_frame, tracks


def load_frame_masks(frame_index, segmentation_root):
    """返回 {local_instance_id: (mask_bool, score, label)}。"""

    instances_path = Path(segmentation_root) / f"frame_{frame_index:06d}_instances.json"
    if not instances_path.exists():
        return None
    instances = json.load(open(instances_path, encoding="utf-8"))
    masks = {}
    for item in instances:
        mask = cv2.imread(item["mask_path"], cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        mask = mask > 0
        if not mask.any():
            continue
        masks[int(item["local_instance_id"])] = (
            mask,
            float(item.get("detection_score", 0.0)),
            item.get("label", ""),
        )
    return masks


def best_overlapping(gt_mask, masks):
    """找出与该 GT 掩码最重合的预测掩码。

    返回 (best_iou, best_local_id, best_mask_area, n_overlapping)。
    `n_overlapping` 统计所有 IoU > 0 的预测掩码数量——大于 1 说明
    同一个物体被切成了多个检测。
    """

    gt_area = int(gt_mask.sum())
    best_iou = 0.0
    best_local = None
    best_area = 0
    overlapping = 0
    for local_id, (mask, _, _) in masks.items():
        intersection = int(np.logical_and(mask, gt_mask).sum())
        if intersection == 0:
            continue
        overlapping += 1
        union = int(mask.sum()) + gt_area - intersection
        if union == 0:
            continue
        iou = intersection / union
        if iou > best_iou:
            best_iou = iou
            best_local = local_id
            best_area = int(mask.sum())
    return best_iou, best_local, best_area, overlapping


def trace_gt_objects(args, tracking, segmentation_root, object_to_class, class_filter, frames):
    """逐帧扫描，为每个 GT 物体建立一条时间序列。"""

    traces = {}
    for frame_index in frames:
        gt_path = Path(args.gt_root) / "instance_masks" / f"instance{frame_index:06d}.png"
        gt_image = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if gt_image is None:
            continue
        masks = load_frame_masks(frame_index, segmentation_root)
        if masks is None:
            continue
        frame_assoc = tracking.get(frame_index, {})

        gt_ids = [
            int(value) for value in np.unique(gt_image)
            if value != 0
            and int((gt_image == value).sum()) >= args.min_gt_area
            and (class_filter is None or object_to_class.get(int(value)) in class_filter)
        ]

        for gt_id in gt_ids:
            gt_mask = gt_image == gt_id
            iou, local_id, mask_area, overlapping = best_overlapping(gt_mask, masks)

            assoc = frame_assoc.get(local_id) if local_id is not None else None
            record = {
                "frame": frame_index,
                "gt_area": int(gt_mask.sum()),
                "best_iou": round(iou, 4),
                "best_local_id": local_id,
                "best_global_id": assoc["global_id"] if assoc else None,
                "best_mask_area": mask_area,
                "n_overlapping": overlapping,
                "decision": assoc["decision"] if assoc else None,
                "cost": assoc["cost"] if assoc else None,
                "label": assoc["label"] if assoc else None,
            }
            trace = traces.setdefault(
                gt_id, {"gt_id": gt_id, "class": object_to_class.get(gt_id, "?"), "frames": []}
            )
            trace["frames"].append(record)

    for trace in traces.values():
        trace["frames"].sort(key=lambda item: item["frame"])
    return traces


def classify_transition(current, args, gap_missed, gap_absent):
    """给一次换轨归因。

    顺序很重要：**先看换轨前有没有观测断档**（断档是 2D 侧的问题，
    再好的关联逻辑也救不回来），只有在没有断档时才去怪 3D 关联。

    `gap_missed`：断档期间 GT 仍标注该物体 → 检测器/SAM2 漏了（可修）
    `gap_absent`：断档期间 GT 根本没标注该物体 → 物体本就不在该帧视野内
                 （被遮挡或出了视锥，属"合理的看不见"，只能靠轨迹记忆/运动模型）
    """

    decision = current.get("entry_decision")
    cost = current.get("entry_cost")

    if gap_missed > 0:
        return {"cause": "detection_gap"}
    if gap_absent > 0:
        return {"cause": "occlusion_gap"}
    if decision == "deferred_conflict":
        return {"cause": "association_conflict"}
    if decision == "new_tentative" and cost is None:
        return {"cause": "association_no_candidate"}
    if cost is not None and cost >= args.near_threshold_cost * 0.8:
        return {"cause": "association_cost_high", "cost": round(cost, 4)}
    return {"cause": "other", "decision": decision, "cost": cost}


def analyse_trace(trace, args, frame_order):
    """把一条 GT 时间序列切成「段」，并对每次换轨归因。"""

    by_frame = {item["frame"]: item for item in trace["frames"]}
    accepted = [item for item in trace["frames"] if item["best_iou"] >= args.iou_threshold]
    trace["visible_frames"] = len(trace["frames"])
    trace["accepted_frames"] = len(accepted)
    trace["track_ids"] = sorted({item["best_global_id"] for item in accepted
                                 if item["best_global_id"] is not None})

    # 沿 frame_order 走，把"检到同一 global_id 的连续帧"并成一段。
    # 必须按全局帧序走，才能把"该帧 GT 未标注"也算成断档。
    segments = []
    for frame in frame_order:
        item = by_frame.get(frame)
        if item is None or item["best_iou"] < args.iou_threshold:
            continue
        gid = item["best_global_id"]
        if segments and segments[-1]["global_id"] == gid:
            segments[-1]["last_frame"] = frame
            segments[-1]["frames"] += 1
            continue
        segments.append({
            "global_id": gid,
            "first_frame": frame,
            "last_frame": frame,
            "frames": 1,
            "entry_decision": item["decision"],
            "entry_cost": item["cost"],
            "entry_iou": item["best_iou"],
        })

    transitions = []
    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]
        gap = [frame for frame in frame_order
               if previous["last_frame"] < frame < current["first_frame"]]
        # 断档里 GT 仍标注 → 检测器漏了；GT 没标注 → 物体本就不在视野内
        missed = sum(1 for frame in gap if frame in by_frame)
        absent = len(gap) - missed
        cause = classify_transition(current, args, missed, absent)
        transitions.append({
            "from_track": previous["global_id"],
            "to_track": current["global_id"],
            "from_last_frame": previous["last_frame"],
            "to_first_frame": current["first_frame"],
            "gap_frames": len(gap),
            "gap_detector_missed": missed,
            "gap_not_visible": absent,
            "entry_decision": current["entry_decision"],
            "entry_cost": current["entry_cost"],
            "entry_iou": current["entry_iou"],
            **cause,
        })

    trace["segments"] = segments
    trace["transitions"] = transitions
    trace["fragmented"] = len(trace["track_ids"]) > 1
    return trace


def summarise_mask_quality(traces, args):
    """所有 (GT 物体, 帧) 对的掩码质量分布。"""

    buckets = {"iou_ge_0.5": 0, "iou_0.25_0.5": 0, "iou_0_0.25": 0, "iou_eq_0": 0}
    ratios = []
    for trace in traces.values():
        for item in trace["frames"]:
            iou = item["best_iou"]
            if iou >= 0.5:
                buckets["iou_ge_0.5"] += 1
            elif iou >= 0.25:
                buckets["iou_0.25_0.5"] += 1
            elif iou > 0:
                buckets["iou_0_0.25"] += 1
            else:
                buckets["iou_eq_0"] += 1
            if iou > 0 and item["best_mask_area"] > 0:
                ratios.append(item["best_mask_area"] / item["gt_area"])

    total = sum(buckets.values())
    return {
        "pairs": total,
        "buckets": buckets,
        "fraction": {key: (round(value / total, 4) if total else None)
                     for key, value in buckets.items()},
        "mask_to_gt_area_ratio": {
            "median": round(float(np.median(ratios)), 4) if ratios else None,
            "p10": round(float(np.percentile(ratios, 10)), 4) if ratios else None,
            "p90": round(float(np.percentile(ratios, 90)), 4) if ratios else None,
        },
    }


def summarise_purity(traces, args):
    """轨道纯度：一条轨道吸了几个 GT 物体 / 几个 GT 类别。"""

    track_to_gts = {}
    for trace in traces.values():
        for item in trace["frames"]:
            if item["best_iou"] < args.iou_threshold:
                continue
            gid = item["best_global_id"]
            if gid is None:
                continue
            track_to_gts.setdefault(gid, {}).setdefault(trace["gt_id"], trace["class"])

    impure = {
        str(gid): {
            "gt_objects": len(entries),
            "gt_classes": sorted(set(entries.values())),
        }
        for gid, entries in track_to_gts.items() if len(entries) > 1
    }
    return {
        "tracks_with_gt": len(track_to_gts),
        "pure_tracks": sum(1 for entries in track_to_gts.values() if len(entries) == 1),
        "impure_tracks": impure,
    }


def main():
    args = parse_arguments()

    tracking, tracks = load_tracking(args.tracking_json)
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

    traces = trace_gt_objects(args, tracking, args.segmentation_root,
                              object_to_class, class_filter, frames)
    for trace in traces.values():
        analyse_trace(trace, args, frames)

    # 归因计数
    cause_counts = {}
    gap_missed_total = 0
    gap_absent_total = 0
    for trace in traces.values():
        for transition in trace["transitions"]:
            cause_counts[transition["cause"]] = cause_counts.get(transition["cause"], 0) + 1
            gap_missed_total += transition["gap_detector_missed"]
            gap_absent_total += transition["gap_not_visible"]

    fragmented = {gt_id: trace for gt_id, trace in traces.items() if trace["fragmented"]}
    mask_quality = summarise_mask_quality(traces, args)
    purity = summarise_purity(traces, args)

    total_transitions = sum(cause_counts.values())
    report = {
        "tracking_json": args.tracking_json,
        "segmentation_root": args.segmentation_root,
        "gt_root": str(gt_root),
        "evaluated_frames": len(frames),
        "frame_range": [frames[0], frames[-1]] if frames else [],
        "iou_threshold": args.iou_threshold,
        "min_gt_area": args.min_gt_area,
        "gt_class_filter": sorted(class_filter) if class_filter else None,
        "gt_objects": len(traces),
        "fragmented_gt_objects": len(fragmented),
        "fragmentation_ratio": round(len(fragmented) / len(traces), 4) if traces else None,
        "transitions": total_transitions,
        "transition_causes": dict(sorted(cause_counts.items(), key=lambda kv: -kv[1])),
        "transition_cause_share": {
            key: round(value / total_transitions, 4)
            for key, value in cause_counts.items()
        } if total_transitions else {},
        "gap_breakdown": {
            "frames_detector_missed": gap_missed_total,
            "frames_not_visible": gap_absent_total,
        },
        "mask_quality": mask_quality,
        "track_purity": purity,
        "per_gt_object": {
            str(gt_id): {
                "class": trace["class"],
                "visible_frames": trace["visible_frames"],
                "accepted_frames": trace["accepted_frames"],
                "track_ids": trace["track_ids"],
                "segments": trace["segments"],
                "transitions": trace["transitions"],
            }
            for gt_id, trace in sorted(traces.items())
        },
        "note": (
            "换轨归因基于已有产物，不重跑流水线。gap_frames 是两次观测之间的评测帧数；"
            "gap_detector_missed 表示那些帧 GT 仍标注该物体（检测器漏检），"
            "gap_not_visible 表示 GT 本就没标注（物体不在视野内，属合理断档）。"
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"帧数 {len(frames)}（{frames[0]}–{frames[-1]}）  "
          f"GT 物体 {len(traces)} 个，其中 {len(fragmented)} 个被拆成多轨"
          f"（{report['fragmentation_ratio']}）")
    print(f"换轨事件 {total_transitions} 次，归因分布：")
    for cause, count in sorted(cause_counts.items(), key=lambda kv: -kv[1]):
        share = count / total_transitions if total_transitions else 0.0
        print(f"  {cause:28s} {count:4d}  ({share:.1%})")

    print("\n掩码质量（GT 物体 × 帧，共 %d 对）：" % mask_quality["pairs"])
    for key, value in mask_quality["fraction"].items():
        print(f"  {key:14s} {mask_quality['buckets'][key]:5d}  ({value:.1%})" if value is not None
              else f"  {key:14s} {mask_quality['buckets'][key]:5d}")
    ratio = mask_quality["mask_to_gt_area_ratio"]
    print(f"  预测掩码/GT 面积比：中位 {ratio['median']}  p10 {ratio['p10']}  p90 {ratio['p90']}")

    print(f"\n轨道纯度：{purity['pure_tracks']}/{purity['tracks_with_gt']} 条轨道只对应 1 个 GT 物体"
          f"，{len(purity['impure_tracks'])} 条误合并")

    if fragmented:
        print("\n被拆分的 GT 物体（前 15 个）：")
        print("%6s %-16s %7s %7s  %s" % ("gt_id", "class", "可见帧", "轨道数", "换轨归因"))
        for gt_id, trace in sorted(fragmented.items(),
                                   key=lambda kv: -len(kv[1]["track_ids"]))[:15]:
            causes = ",".join(t["cause"] for t in trace["transitions"]) or "-"
            print("%6d %-16s %7d %7d  %s" % (gt_id, trace["class"], trace["visible_frames"],
                                             len(trace["track_ids"]), causes))
    print(f"\n报告: {output_path}")


if __name__ == "__main__":
    main()
