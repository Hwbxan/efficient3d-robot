"""诊断轨道碎片化的真实形态。

问题：同一个 GT 物体常常被切成多条 track。有两种可能成因，修复方式完全不同：

1. **断裂（break）**——两条 track 在时间上前后相接（A 结束后 B 才开始），
   说明物体中途跟丢了又被当成新物体。修法是放宽记忆/重捕获门限。
2. **同帧重复（duplicate）**——两条 track 的时间区间重叠，说明同一时刻
   同一物体产生了两个观测（2D 过分割），或者同一帧内被分配了两个 id。
   修法是在关联层合并，跟放宽门限无关，放宽只会更糟。

本脚本把每条 track 的帧区间、空间位置打出来，并对同标签的 track 两两
比较时间重叠度与体素 IoU，直接判断属于哪一种。

用法（项目根目录）：
    python -m tools.diagnose_tracks \
        --scene room_2 --run-root /tmp/eval_voxel --gt-root outputs/gt
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.check_instance_pairs import load_observations  # noqa: E402
from tools.sweep_tracker_params import load_all_observations  # noqa: E402
from src.mapping.geometric_instance_tracker import (  # noqa: E402
    GeometricInstanceTracker,
)


def track_summary(track, observations_by_frame):
    """汇总一条 track 的时间区间与体素集合。"""

    frame_indices = sorted(observations_by_frame.keys())
    voxel_set = set()
    centroids = []
    for frame_index in frame_indices:
        observation = observations_by_frame[frame_index]
        for key in observation.get("voxel_keys", []) or []:
            voxel_set.add(tuple(key))
        centroid = observation.get("centroid_world")
        if centroid is not None:
            centroids.append([float(value) for value in centroid])
    return {
        "frames": frame_indices,
        "first": frame_indices[0] if frame_indices else None,
        "last": frame_indices[-1] if frame_indices else None,
        "voxels": voxel_set,
        "centroids": centroids,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gt-root", default="outputs/gt")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--end-frame", type=int, default=1990)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    run_directory = Path(args.run_root) / args.scene
    instances_root = run_directory / "instances_3d"
    frames = list(range(0, args.end_frame + 1, args.frame_stride))
    per_frame = load_all_observations(instances_root, frames)
    print(f"场景 {args.scene}：载入 {len(per_frame)} 帧观测")

    tracker = GeometricInstanceTracker()
    observations_of_track = {}
    label_of = {}
    for frame_index, observations in per_frame:
        by_local = {
            observation["local_instance_id"]: observation
            for observation in observations
        }
        for result in tracker.update(observations, frame_index):
            global_id = result["global_id"]
            if global_id is None:
                continue
            observation = by_local[result["local_instance_id"]]
            observations_of_track.setdefault(global_id, {})[frame_index] = (
                observation
            )
            label_of.setdefault(global_id, result["raw_label"])
    tracks = [{"global_id": key} for key in sorted(observations_of_track)]
    print(f"导出 {len(tracks)} 条 track\n")

    # ---- 1. 每条 track 的基本信息 ----
    summaries = {}
    for track in tracks:
        track_id = track["global_id"]
        summaries[track_id] = track_summary(
            track, observations_of_track[track_id]
        )

    print(f"{'id':>4}{'label':>14}{'帧数':>6}{'首帧':>7}{'末帧':>7}"
          f"{'体素数':>8}{'质心':>26}")
    for track_id in sorted(summaries):
        summary = summaries[track_id]
        label = label_of[track_id]
        center = ""
        if summary["centroids"]:
            mean = [sum(axis) / len(summary["centroids"])
                    for axis in zip(*summary["centroids"])]
            center = "[" + ", ".join(f"{value:6.2f}" for value in mean) + "]"
        print(f"{track_id:>4}{str(label):>14}{len(summary['frames']):>6}"
              f"{summary['first'] or -1:>7}{summary['last'] or -1:>7}"
              f"{len(summary['voxels']):>8}{center:>26}")

    # ---- 2. 同标签 track 两两比较 ----
    print("\n同标签 track 对（只列出体素 IoU > 0.02 的）：")
    print(f"{'A':>4}{'B':>4}{'label':>12}{'时间重叠':>9}{'帧距':>7}"
          f"{'并集IoU':>9}{'质心距':>8}  形态")
    pairs = []
    identifiers = sorted(summaries)
    for i, first_id in enumerate(identifiers):
        for second_id in identifiers[i + 1:]:
            if label_of[first_id] != label_of[second_id]:
                continue
            first = summaries[first_id]
            second = summaries[second_id]
            if not first["voxels"] or not second["voxels"]:
                continue
            union = len(first["voxels"] | second["voxels"])
            iou = len(first["voxels"] & second["voxels"]) / union if union else 0.0
            if iou <= 0.02:
                continue
            # 时间重叠：取交集帧数占较小区间的比例
            first_set = set(first["frames"])
            second_set = set(second["frames"])
            overlap = len(first_set & second_set)
            shorter = min(len(first_set), len(second_set))
            overlap_ratio = overlap / shorter if shorter else 0.0
            if overlap_ratio > 0.3:
                shape = "同帧重复"
            elif second["first"] > first["last"]:
                shape = "断裂(前→后)"
            elif first["first"] > second["last"]:
                shape = "断裂(后→前)"
            else:
                shape = "部分重叠"
            gap = min(abs(a - b) for a in [first["first"], first["last"]]
                      for b in [second["first"], second["last"]])
            center_distance = 0.0
            if first["centroids"] and second["centroids"]:
                mean_first = [sum(axis) / len(first["centroids"])
                              for axis in zip(*first["centroids"])]
                mean_second = [sum(axis) / len(second["centroids"])
                               for axis in zip(*second["centroids"])]
                center_distance = sum(
                    (a - b) ** 2 for a, b in zip(mean_first, mean_second)
                ) ** 0.5
            pairs.append((iou, first_id, second_id, label_of[first_id],
                          overlap_ratio, gap, center_distance, shape))

    pairs.sort(reverse=True)
    for iou, a, b, label, overlap_ratio, gap, distance, shape in pairs[:args.top]:
        print(f"{a:>4}{b:>4}{str(label):>12}{overlap_ratio:>9.2f}{gap:>7}"
              f"{iou:>9.3f}{distance:>8.2f}  {shape}")

    # ---- 3. 单帧内同标签观测的过分割检查 ----
    print("\n单帧内同标签多个观测（疑似 2D 过分割）：")
    print(f"{'帧':>7}{'label':>12}{'数量':>5}{'彼此IoU':>26}{'质心距':>9}")
    duplicate_frames = 0
    for frame_index, observations in per_frame:
        by_label = {}
        for observation in observations:
            by_label.setdefault(observation.get("label", ""), []).append(
                observation
            )
        for label, group in by_label.items():
            if len(group) < 2:
                continue
            keys = []
            centers = []
            for observation in group:
                keys.append({tuple(item) for item in
                             (observation.get("voxel_keys") or [])})
                centers.append([float(value) for value in
                                observation.get("centroid_world", [0, 0, 0])])
            ious = []
            distances = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    union = len(keys[i] | keys[j])
                    ious.append(len(keys[i] & keys[j]) / union if union else 0.0)
                    distances.append(sum(
                        (a - b) ** 2 for a, b in zip(centers[i], centers[j])
                    ) ** 0.5)
            if max(ious) > 0.05:
                duplicate_frames += 1
                if duplicate_frames <= 12:
                    text = " ".join(f"{value:.2f}" for value in ious)
                    print(f"{frame_index:>7}{str(label):>12}{len(group):>5}"
                          f"{text:>26}{max(distances):>9.2f}")
    print(f"  → 共 {duplicate_frames} 帧存在同标签高 IoU 重复观测")


if __name__ == "__main__":
    main()
