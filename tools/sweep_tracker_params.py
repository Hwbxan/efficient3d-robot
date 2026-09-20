"""离线扫描跟踪器参数：只对已保存的 3D 实例重做关联 + 评估。

在线推理里 2D 前端（Grounding DINO + SAM2）占了 95% 以上的时间，而关联本身
只要几秒。调跟踪参数时没必要反复重跑 2D，因此这里复用现成的
`instances_3d/frame_*/instances_3d.json`，只重跑关联与 GT 评估，
单组耗时约 15 秒，可以在几分钟内扫完几十组参数。

用法（项目根目录）：
    python -m tools.sweep_tracker_params \
        --scene room_2 \
        --run-root /tmp/eval_tickfix \
        --gt-root outputs/gt \
        --output /tmp/sweep_room2.json

产出：每组参数的 AP@.25 / AP@.5 / 单轨率 / 纯轨率 / 碎片度，按 AP@.25 排序。
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.check_instance_pairs import load_observations  # noqa: E402
from src.mapping.geometric_instance_tracker import (  # noqa: E402
    GeometricInstanceTracker,
)

GT_CLASSES = [
    "bin", "basket", "tissue-paper",
    "chair", "sofa", "stool", "armchair", "couch",
    "door",
    "table", "desk", "desk-organizer",
    "tv-screen", "tablet", "monitor",
]


def load_all_observations(instances_root: Path, frames):
    """按顺序加载各帧观测；缺失的帧会被跳过。"""

    per_frame = []
    for frame_index in frames:
        path = instances_root / f"frame_{frame_index:06d}" / "instances_3d.json"
        if not path.is_file():
            continue
        observations = load_observations(path)
        observations.sort(key=lambda item: item["local_instance_id"])
        per_frame.append((frame_index, observations))
    return per_frame


def run_tracking(per_frame, params):
    tracker = GeometricInstanceTracker(**params)
    frame_results = []
    for frame_index, observations in per_frame:
        results = tracker.update(observations, frame_index)
        frame_results.append(
            {"frame_index": frame_index, "associations": results}
        )
    return {"frames": frame_results, "tracks": tracker.export_tracks()}


def evaluate(tracking, run_directory: Path, gt_directory: Path, workdir: Path):
    """把 tracking 写到临时目录，调用 evaluate_against_gt 取回指标。"""

    tracking_path = workdir / "tracking.json"
    with tracking_path.open("w", encoding="utf-8") as handle:
        json.dump(tracking, handle)

    evaluation_path = workdir / "gt_eval.json"
    command = [
        sys.executable, "-m", "tools.evaluate_against_gt",
        "--tracking-json", str(tracking_path),
        "--segmentation-root", str(run_directory / "segmentation"),
        "--gt-root", str(gt_directory),
        "--output", str(evaluation_path),
        "--gt-classes", ",".join(GT_CLASSES),
        "--max-area-fraction", "0.4",
    ]
    process = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"评估失败：{process.stdout[-2000:]}\n{process.stderr[-2000:]}"
        )
    return json.loads(evaluation_path.read_text(encoding="utf-8"))


def build_grid():
    """待扫描的参数组合。"""

    base = dict(max_center_distance=0.50, max_bbox_gap=0.10,
                memory_frames=30, reacquire_max_center_distance=1.5)
    # 关掉体素通道即回到纯几何基线，用来隔离「体素特征」这一改动本身的收益。
    geometric_only = dict(base, voxel_iou_weight=0.0, voxel_min_coverage=1.1,
                          reacquire_min_coverage=1.1)
    grid = [
        {"name": "纯几何基线（无体素）",
         "params": geometric_only},
        {"name": "默认",
         "params": dict(base)},
        {"name": "余量0 + 几何0.65/0.15",
         "params": dict(base, ambiguity_margin=0.0,
                        max_center_distance=0.65, max_bbox_gap=0.15)},
        {"name": "余量0 + 几何0.80/0.20",
         "params": dict(base, ambiguity_margin=0.0,
                        max_center_distance=0.80, max_bbox_gap=0.20)},
        {"name": "余量0 + 几何0.65 + 代价0.80",
         "params": dict(base, ambiguity_margin=0.0, unmatched_cost=0.80,
                        max_center_distance=0.65, max_bbox_gap=0.15)},
        {"name": "余量0 + 几何0.65 + min.50/req.65",
         "params": dict(base, ambiguity_margin=0.0,
                        max_center_distance=0.65, max_bbox_gap=0.15,
                        voxel_min_coverage=0.50,
                        reacquire_min_coverage=0.65)},
        {"name": "余量0 + 几何0.65 + 窗口8",
         "params": dict(base, ambiguity_margin=0.0, voxel_window_size=8,
                        max_center_distance=0.65, max_bbox_gap=0.15)},
        {"name": "余量0 + 几何0.65 + 重捕获1.0/0.20",
         "params": dict(base, ambiguity_margin=0.0,
                        max_center_distance=0.65, max_bbox_gap=0.15,
                        reacquire_max_center_distance=1.0,
                        reacquire_max_bbox_gap=0.20)},
        {"name": "歧义余量 0.02",
         "params": dict(base, ambiguity_margin=0.02)},
        {"name": "歧义余量 0.05",
         "params": dict(base, ambiguity_margin=0.05)},
        {"name": "歧义余量 0（永不歧义）",
         "params": dict(base, ambiguity_margin=0.0)},
        {"name": "不匹配代价 0.80",
         "params": dict(base, unmatched_cost=0.80)},
        {"name": "不匹配代价 0.80 + 余量0.02",
         "params": dict(base, unmatched_cost=0.80, ambiguity_margin=0.02)},
        {"name": "不匹配代价 0.95",
         "params": dict(base, unmatched_cost=0.95)},
        {"name": "几何放宽 0.65/0.15",
         "params": dict(base, max_center_distance=0.65, max_bbox_gap=0.15)},
        {"name": "几何放宽 + 代价0.80",
         "params": dict(base, max_center_distance=0.65, max_bbox_gap=0.15,
                        unmatched_cost=0.80)},
        {"name": "min.50/req.65",
         "params": dict(base, voxel_min_coverage=0.50,
                        reacquire_min_coverage=0.65)},
        {"name": "min.30/req.45",
         "params": dict(base, voxel_min_coverage=0.30,
                        reacquire_min_coverage=0.45)},
        {"name": "min.25/req.40",
         "params": dict(base, voxel_min_coverage=0.25,
                        reacquire_min_coverage=0.40)},
        {"name": "窗口 4",
         "params": dict(base, voxel_window_size=4)},
        {"name": "窗口 12",
         "params": dict(base, voxel_window_size=12)},
        {"name": "窗口 20",
         "params": dict(base, voxel_window_size=20)},
        {"name": "重捕获收紧 0.8/0.15",
         "params": dict(base, reacquire_max_center_distance=0.8,
                        reacquire_max_bbox_gap=0.15)},
        {"name": "重捕获收紧 + min.30",
         "params": dict(base, reacquire_max_center_distance=0.8,
                        reacquire_max_bbox_gap=0.15,
                        voxel_min_coverage=0.30,
                        reacquire_min_coverage=0.45)},
        {"name": "记忆 2000（近乎全程）",
         "params": dict(base, voxel_memory_frames=2000)},
        {"name": "记忆 200",
         "params": dict(base, voxel_memory_frames=200)},
        {"name": "窗口12 + min.30/req.45 + 记忆2000",
         "params": dict(base, voxel_window_size=12,
                        voxel_min_coverage=0.30,
                        reacquire_min_coverage=0.45,
                        voxel_memory_frames=2000)},
    ]
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True, help="raw 风格场景名，如 room_2")
    parser.add_argument("--run-root", required=True,
                        help="某次运行输出根目录（含 <scene>/instances_3d 与 segmentation）")
    parser.add_argument("--gt-root", default="outputs/gt")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--end-frame", type=int, default=1990)
    parser.add_argument("--output", default=None, help="结果 JSON 落盘路径")
    args = parser.parse_args()

    run_directory = Path(args.run_root) / args.scene
    gt_directory = Path(args.gt_root) / args.scene
    instances_root = run_directory / "instances_3d"

    frames = list(range(0, args.end_frame + 1, args.frame_stride))
    per_frame = load_all_observations(instances_root, frames)
    print(f"场景 {args.scene}：载入 {len(per_frame)} 帧观测")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for entry in build_grid():
            tracking = run_tracking(per_frame, entry["params"])
            evaluation = evaluate(
                tracking, run_directory, gt_directory, workdir,
            )
            detection = evaluation["detection"]
            association = evaluation["association"]
            rows.append({
                "name": entry["name"],
                "params": entry["params"],
                "ap25": detection["iou_0.25"]["ap"],
                "ap50": detection["iou_0.5"]["ap"],
                "precision": detection["iou_0.25"]["precision"],
                "recall": detection["iou_0.25"]["recall"],
                "single": association["single_track_ratio"],
                "pure": association["pure_track_ratio"],
                "frag": association["fragmentation_mean"],
                "tracks": association["matched_tracks"],
                "gt_objects": association["matched_gt_objects"],
            })
            print(
                f"  {entry['name']:<28} AP@.25={rows[-1]['ap25']:.4f} "
                f"AP@.5={rows[-1]['ap50']:.4f} 单轨={rows[-1]['single']:.3f} "
                f"纯轨={rows[-1]['pure']:.3f} 碎片={rows[-1]['frag']:.3f}"
            )

    print(f"\n{'参数组':<28}{'AP@.25':>9}{'AP@.5':>9}{'单轨':>8}{'纯轨':>8}{'碎片':>8}{'轨道/GT':>10}")
    for row in sorted(rows, key=lambda item: -item["ap25"]):
        print(f"{row['name']:<28}{row['ap25']:>9.4f}{row['ap50']:>9.4f}"
              f"{row['single']:>8.3f}{row['pure']:>8.3f}{row['frag']:>8.3f}"
              f"{str(row['tracks']) + '/' + str(row['gt_objects']):>10}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n结果：{args.output}")


if __name__ == "__main__":
    main()
