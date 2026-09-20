"""打印「新建轨道」那一刻所有候选旧轨道的判定量。

用途：跟踪器把同一物体切成新轨时，光看最终结果猜不出是门限卡住、还是
体素 IoU 没算出来、还是被别的观测抢占。这里在建轨的瞬间把所有候选轨道
的 gap / 体素 IoU / 几何代价 / 质心距打出来，一眼就能看出该连的为什么没连。

用法（项目根目录）：
    python -m tools.debug_new_tracks --scene room_2 \
        --run-root /tmp/eval_merge --from-frame 1800
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.sweep_tracker_params import load_all_observations  # noqa: E402
from src.mapping import geometric_instance_tracker as git  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--end-frame", type=int, default=1990)
    parser.add_argument("--from-frame", type=int, default=0,
                        help="只打印该帧之后新建的轨道")
    args = parser.parse_args()

    instances_root = Path(args.run_root) / args.scene / "instances_3d"
    frames = list(range(0, args.end_frame + 1, args.frame_stride))
    per_frame = load_all_observations(instances_root, frames)

    tracker = git.GeometricInstanceTracker()
    print(f"memory_frames={tracker.memory_frames} "
          f"voxel_memory_frames={tracker.voxel_memory_frames} "
          f"reacquire_min_coverage={tracker.reacquire_min_coverage} "
          f"voxel_min_coverage={tracker.voxel_min_coverage} "
          f"voxel_iou_weight={tracker.voxel_iou_weight} "
          f"unmatched_cost={tracker.unmatched_cost}")

    original_create = tracker._create_track

    def traced_create(observation, frame_index):
        if frame_index >= args.from_frame:
            center = np.asarray(observation["centroid_world"])
            keys = observation.get("voxel_keys")
            print(f"\n--- 帧 {frame_index} 新建轨道 "
                  f"label={observation['label']} "
                  f"质心={np.round(center, 2).tolist()} "
                  f"体素数={0 if not keys else len(keys)}")
            rows = []
            for track in tracker.tracks.values():
                iou, coverage = git.voxel_scores(
                    observation, track.voxel_window
                )
                gap = frame_index - track.last_seen_frame
                cost = tracker._association_cost(
                    observation, track, frame_index
                )
                distance = float(np.linalg.norm(
                    center
                    - np.asarray(track.latest_observation["centroid_world"])
                ))
                rows.append((-1.0 if coverage is None else coverage,
                             track.global_id, track.label, gap, distance, cost,
                             -1.0 if iou is None else iou,
                             git.voxel_size_of(track.voxel_window)))
            rows.sort(reverse=True)
            for (coverage, gid, label, gap, distance, cost, iou,
                 union_size) in rows[:6]:
                def fmt(value):
                    return "None" if value < 0 else f"{value:.3f}"
                cost_text = "inf" if not np.isfinite(cost) else f"{cost:.3f}"
                print(f"    track {gid:>3} {label:<8} gap={gap:>5} "
                      f"覆盖={fmt(coverage):>6} IoU={fmt(iou):>6} "
                      f"窗口体素={union_size:>5} 质心距={distance:5.2f} "
                      f"代价={cost_text:>6}")
        return original_create(observation, frame_index)

    tracker._create_track = traced_create

    for frame_index, observations in per_frame:
        tracker.update(observations, frame_index)


if __name__ == "__main__":
    main()
