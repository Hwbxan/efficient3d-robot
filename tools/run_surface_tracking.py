"""复用基线单帧缓存，对比原关联与历史表面补充关联；不调用检测模型。"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.mapping.geometric_instance_tracker import GeometricInstanceTracker
from src.mapping.surface_instance_tracker import SurfaceInstanceTracker
from tools.audit_instance_tracking import verify_result


ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_world_points(observation):
    import open3d as o3d

    path = project_path(observation["point_cloud_path"])
    if not path.is_file():
        raise FileNotFoundError(f"找不到基线实例点云：{path}")
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points).copy()
    if len(points) == 0:
        raise ValueError(f"基线实例点云为空：{path}")
    # 已经是世界坐标，不能再次乘相机位姿。
    return points


def origin_of(result, origins):
    global_id = result["global_id"]
    if global_id is None:
        return None
    if result["decision"] == "new_tentative":
        origins[global_id] = [result["frame_index"], result["local_instance_id"]]
    return origins[global_id]


def compare_sequence(baseline, instances_root, surface_tracker):
    baseline_tracker = GeometricInstanceTracker()
    frames_v2 = []
    original_births, new_births = {}, {}
    correspondences = defaultdict(Counter)
    changed = []
    fallback_events = []

    for frame in baseline["frames"]:
        frame_index = frame["frame_index"]
        records = read_json(instances_root / f"frame_{frame_index:06d}" / "instances_3d.json")
        by_local_id = {item["local_instance_id"]: item for item in records}
        expected = frame["associations"]
        local_ids = [item["local_instance_id"] for item in expected]
        if len(by_local_id) != len(records) or len(set(local_ids)) != len(local_ids):
            raise ValueError(f"帧 {frame_index} 含重复局部 ID")
        # 采用与原基线一致的有效观测集合和顺序，不改变前处理。
        observations = [by_local_id[local_id] for local_id in local_ids]
        replayed = baseline_tracker.update(observations, frame_index)
        if len(replayed) != len(expected):
            raise ValueError("基线复现失败：返回数量不同")
        for old, actual in zip(expected, replayed):
            verify_result(old, actual)

        updated = surface_tracker.update(observations, frame_index)
        frames_v2.append({"frame_index": frame_index, "associations": updated})
        print(f"\nFrame {frame_index:06d}", flush=True)

        for old, new in zip(expected, updated):
            old_origin = origin_of(old, original_births)
            new_origin = origin_of(new, new_births)
            if old["global_id"] is not None:
                target = f"G{new['global_id']:03d}" if new["global_id"] is not None else "unassigned"
                correspondences[old["global_id"]][target] += 1
            event = {
                "frame_index": frame_index,
                "local_instance_id": new["local_instance_id"],
                "raw_label": new["raw_label"],
                "baseline_global_id": old["global_id"],
                "v2_global_id": new["global_id"],
                "baseline_track_birth": old_origin,
                "v2_track_birth": new_origin,
                "baseline_decision": old["decision"],
                "v2_decision": new["decision"],
                "association_source": new["association_source"],
            }
            selected = next((
                item for item in new["surface_candidates"]
                if item["global_id"] == new["global_id"] and item["surface_passed"]
            ), None)
            if selected:
                event["surface_coverage"] = selected["coverage"]
                event["reference_last_frame"] = selected["reference_last_frame"]
            if old_origin != new_origin or old["decision"] != new["decision"]:
                changed.append(event)
            if new["association_source"].startswith("surface"):
                fallback_events.append(event)

            old_name = f"G{old['global_id']:03d}" if old["global_id"] is not None else "-"
            new_name = f"G{new['global_id']:03d}" if new["global_id"] is not None else "-"
            print(
                f"  Local {new['local_instance_id']:02d} {new['raw_label']:<18} | "
                f"基线 {old_name} → V2 {new_name} | {new['decision']} | {new['association_source']}",
                flush=True,
            )

    original_results = [item for frame in baseline["frames"] for item in frame["associations"]]
    new_results = [item for frame in frames_v2 for item in frame["associations"]]
    report = {
        "baseline_reproduced": True,
        "processed_frames": [frame["frame_index"] for frame in baseline["frames"]],
        "observation_count": len(new_results),
        "baseline_track_count": len(baseline_tracker.tracks),
        "v2_track_count": len(surface_tracker.tracks),
        "baseline_decisions": dict(Counter(item["decision"] for item in original_results)),
        "v2_decisions": dict(Counter(item["decision"] for item in new_results)),
        "v2_sources": dict(Counter(item["association_source"] for item in new_results)),
        "v2_track_states": dict(Counter(track.status for track in surface_tracker.tracks.values())),
        "surface_parameters": surface_tracker.surface_parameters(),
        "baseline_to_v2_observation_mapping": [
            {"baseline_global_id": global_id, "v2_observation_counts": dict(counts)}
            for global_id, counts in sorted(correspondences.items())
        ],
        "changed_assignment_events": changed,
        "surface_events": fallback_events,
        "notes": [
            "减少建档会使后续 ID 重新编号，编号不同本身不是 ID 切换。",
            "track_birth 以首次建档的 [帧编号, 局部 ID] 表示，仅用于比较回放，不是真值身份。",
            "表面覆盖率是几何证据，不是关联准确率；ID 数减少也不证明精度提升。",
            "保留所有 tentative 实例；本实验不改变类别投票、掩码去重或地图发布策略。",
        ],
    }
    return {"frames": frames_v2, "tracks": surface_tracker.export_tracks()}, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", default="outputs/experiments/office0_0000_0300_v1")
    parser.add_argument("--output-directory", default="outputs/experiments/office0_0000_0300_surface_v2")
    parser.add_argument("--surface-distance", type=float, default=0.04)
    parser.add_argument("--minimum-coverage", type=float, default=0.80)
    args = parser.parse_args()
    baseline_directory = project_path(args.baseline_run)
    output_directory = project_path(args.output_directory)
    if output_directory.exists():
        parser.error("输出目录已存在，请使用新的 --output-directory，避免覆盖")
    if baseline_directory == output_directory or baseline_directory in output_directory.parents:
        parser.error("请把 V2 输出放在基线目录之外")
    source_path = baseline_directory / "association/tracking.json"
    baseline = read_json(source_path)
    if not baseline["frames"]:
        parser.error("基线帧列表为空")
    tracker = SurfaceInstanceTracker(
        point_loader=load_world_points,
        surface_distance=args.surface_distance,
        minimum_coverage=args.minimum_coverage,
    )

    # 先复现并比较全部帧，成功后才新建结果目录；失败时不产生半套结果。
    tracking, report = compare_sequence(baseline, baseline_directory / "instances_3d", tracker)
    report["baseline_run"] = str(baseline_directory)
    report["input_instances_root"] = str(baseline_directory / "instances_3d")
    report["input_mask_directory"] = str(baseline_directory / "segmentation")
    report["source_sha256"] = {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            source_path,
            ROOT / "src/mapping/geometric_instance_tracker.py",
            ROOT / "src/mapping/surface_instance_tracker.py",
            Path(__file__).resolve(),
        ]
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    write_json(output_directory / "tracking.json", tracking)
    write_json(output_directory / "comparison.json", report)
    print("\n基线完整复现，V2 回放完成。", flush=True)
    print(f"实例记录：基线 {report['baseline_track_count']} → V2 {report['v2_track_count']}")
    print("V2 决策：", report["v2_decisions"])
    print("关联来源：", report["v2_sources"])
    print(f"结果目录：{output_directory}")


if __name__ == "__main__":
    main()
