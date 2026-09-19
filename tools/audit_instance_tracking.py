"""使用原关联器重放缓存，核对旧结果并导出每个候选的拒绝原因。

不调用模型、不修改原始关联结果、不合并实例。
只有回放结果与基线完全一致时，才写出诊断文件。
"""

import argparse
import json
import math
from pathlib import Path

from src.mapping.geometric_instance_tracker import (
    KNOWN_LABELS,
    GeometricInstanceTracker,
    geometry_distances,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_details(tracker, observation, track):
    """在更新前读取历史状态，不能使用后续帧融合得到的包围盒。"""
    center_distance, bbox_gap = geometry_distances(observation, track.latest_observation)
    label = observation["label"].strip().lower()
    if label not in KNOWN_LABELS or track.label == "unknown":
        label_penalty = 0.5
    elif label == track.label:
        label_penalty = 0.0
    else:
        label_penalty = 1.0

    center_term = 0.70 * center_distance / tracker.max_center_distance
    gap_term = 0.20 * bbox_gap / tracker.max_bbox_gap
    label_term = 0.10 * label_penalty
    raw_cost = center_term + gap_term + label_term
    actual_cost = float(tracker._association_cost(observation, track))

    reasons = []
    if center_distance > tracker.max_center_distance:
        reasons.append("center_distance_gate")
    if bbox_gap > tracker.max_bbox_gap:
        reasons.append("bbox_gap_gate")
    if not reasons and raw_cost >= tracker.unmatched_cost:
        reasons.append("cost_threshold")

    expected_cost = math.inf if any(reason.endswith("_gate") for reason in reasons) else raw_cost
    if not (math.isinf(actual_cost) and math.isinf(expected_cost)):
        if not math.isclose(actual_cost, expected_cost, abs_tol=1e-8, rel_tol=1e-8):
            raise ValueError("当前关联器的代价公式与诊断器不一致，请勿继续解释该审计结果")

    return {
        "global_id": track.global_id,
        "track_label": track.label,
        "track_last_seen_frame": track.last_seen_frame,
        "center_distance_m": center_distance,
        "bbox_gap_m": bbox_gap,
        "center_cost_term": center_term,
        "bbox_cost_term": gap_term,
        "label_cost_term": label_term,
        "raw_cost_before_gates": raw_cost,
        "association_cost": actual_cost if math.isfinite(actual_cost) else None,
        "eligible": actual_cost < tracker.unmatched_cost,
        "rejection_reasons": reasons,
    }


def verify_result(expected, actual):
    fields = ["frame_index", "local_instance_id", "raw_label", "global_id", "decision"]
    for field in fields:
        if expected[field] != actual[field]:
            raise ValueError(
                f"回放与基线不一致：帧 {expected['frame_index']}，"
                f"Local {expected['local_instance_id']}，字段 {field}。"
                "请检查关联器版本、参数和单帧缓存；诊断文件未生成。"
            )
    old_cost, new_cost = expected["association_cost"], actual["association_cost"]
    if old_cost is None or new_cost is None:
        equal = old_cost is None and new_cost is None
    else:
        equal = math.isclose(old_cost, new_cost, abs_tol=1e-7, rel_tol=1e-7)
    if not equal:
        raise ValueError("回放代价与基线不一致，诊断文件未生成")


def audit(tracking, instances_root, focus_frames):
    tracker = GeometricInstanceTracker()
    events = []
    replayed_count = 0
    for frame in tracking["frames"]:
        frame_index = frame["frame_index"]
        source = instances_root / f"frame_{frame_index:06d}" / "instances_3d.json"
        records = read_json(source)
        by_id = {item["local_instance_id"]: item for item in records}
        if len(by_id) != len(records):
            raise ValueError(f"单帧实例 ID 重复：{source}")

        # 精确沿用基线中参与关联的局部 ID 及顺序，避免引入新的过滤规则。
        expected_results = frame["associations"]
        local_ids = [item["local_instance_id"] for item in expected_results]
        if len(local_ids) != len(set(local_ids)):
            raise ValueError(f"基线关联记录局部 ID 重复：帧 {frame_index}")
        observations = [by_id[local_id] for local_id in local_ids]
        frame_events = []

        for expected, observation in zip(expected_results, observations):
            if expected["decision"] == "matched" and frame_index not in focus_frames:
                continue
            candidates = [candidate_details(tracker, observation, track) for track in tracker.tracks.values()]
            candidates.sort(key=lambda item: (not item["eligible"], item["raw_cost_before_gates"]))
            frame_events.append({
                "frame_index": frame_index,
                "local_instance_id": observation["local_instance_id"],
                "raw_label": observation["label"],
                "observation_centroid_world": observation["centroid_world"],
                "baseline_decision": expected["decision"],
                "baseline_global_id": expected["global_id"],
                "candidates": candidates,
            })

        actual_results = tracker.update(observations, frame_index)
        if len(actual_results) != len(expected_results):
            raise ValueError("回放结果数量与基线不一致")
        for expected, actual in zip(expected_results, actual_results):
            verify_result(expected, actual)
        events.extend(frame_events)
        replayed_count += len(actual_results)

    return {
        "baseline_reproduced": True,
        "replayed_observations": replayed_count,
        "processed_frames": [frame["frame_index"] for frame in tracking["frames"]],
        "parameters": {
            "max_center_distance": tracker.max_center_distance,
            "max_bbox_gap": tracker.max_bbox_gap,
            "unmatched_cost": tracker.unmatched_cost,
            "ambiguity_margin": tracker.ambiguity_margin,
        },
        "note": "eligible 仅表示通过候选筛选，不保证被最终指派；仍有歧义检查及一对一分配。",
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", default="outputs/experiments/office0_0000_0300_v1")
    parser.add_argument("--focus-frames", type=int, nargs="+", default=[80, 120, 130, 140, 160, 200, 250, 300])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_directory = Path(args.run_directory)
    if not run_directory.is_absolute():
        run_directory = ROOT / run_directory
    output = Path(args.output) if args.output else run_directory / "association/candidate_audit.json"
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        parser.error("诊断输出已存在，请用 --output 指定新文件，避免覆盖")

    tracking = read_json(run_directory / "association/tracking.json")
    report = audit(tracking, run_directory / "instances_3d", set(args.focus_frames))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"基线复现成功：{len(report['processed_frames'])} 帧，{report['replayed_observations']} 条观测")
    for event in report["events"]:
        if event["baseline_decision"] == "matched" or event["frame_index"] == 0:
            continue
        print(
            f"\nFrame {event['frame_index']:06d} Local {event['local_instance_id']:02d} "
            f"{event['raw_label']}：{event['baseline_decision']}"
        )
        for candidate in event["candidates"][:3]:
            reason = ", ".join(candidate["rejection_reasons"]) or "eligible_before_assignment"
            print(
                f"  候选 G{candidate['global_id']:03d} | 距离={candidate['center_distance_m']:.3f} m | "
                f"间距={candidate['bbox_gap_m']:.3f} m | 原始代价={candidate['raw_cost_before_gates']:.3f} | {reason}"
            )
    print(f"\n完整候选诊断：{output}")


if __name__ == "__main__":
    main()
