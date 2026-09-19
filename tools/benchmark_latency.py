"""单进程延迟基准：测量在线流水线的**真实**每帧耗时。

为什么需要这个脚本
------------------
`tools/run_instance_sequence.py` 默认**逐帧启动子进程**（每帧一次
`python -m tools.inspect_grounded_sam2` 和一次 `python -m tools.inspect_3d_instances`）。
每启动一次就要重新 import torch / open3d / matplotlib，并**重新加载 DINO 与 SAM2 的权重**，
再加上每帧写 PLY 和渲染 matplotlib 3D 预览。

结果：61 帧实测 18m32s（≈30.7 s/帧），而 DINO+SAM2 自报推理只有 1.89 s。
**约 90% 的挂钟时间不是推理。**

在修掉这个之前，任何"压缩了 X 倍"或"比 Y 快"的说法都没有基线。

本脚本做什么
------------
- **模型只加载一次**（常驻进程）
- 每帧在内存里完成 检测 → 分割 → 3D 提升，不做子进程调度
- 默认**不写任何文件**（`--with-write` 可打开，用来量化 I/O 与预览渲染的代价）
- 每个阶段前后 `torch.cuda.synchronize()`，避免异步 GPU 让计时失真
- 输出分阶段耗时 + 均值/中位/p95

用法
----
    # 纯推理基线（推荐）
    python -m tools.benchmark_latency --frames 0 10 20 30 40 50 60

    # 加上 I/O 与预览渲染，量化它们的代价
    python -m tools.benchmark_latency --frames 0 10 20 30 --with-write
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.datasets.replica_sequence import ReplicaSequence  # noqa: E402
from src.geometry.instance_lifting import lift_mask_to_world  # noqa: E402
from src.perception.grounding_dino_detector import GroundingDinoDetector  # noqa: E402
from src.perception.sam2_box_segmenter import Sam2BoxSegmenter  # noqa: E402


def sync():
    """GPU 是异步的，不同步就测不到真实耗时。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(stage, sink, function, *args, **kwargs):
    """跑一个阶段并记录耗时（毫秒）。"""
    sync()
    start = time.perf_counter()
    result = function(*args, **kwargs)
    sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    sink[stage] = sink.get(stage, 0.0) + elapsed_ms
    return result


def summarise(values):
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
        "sum": sum(ordered),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-directory", default="datasets/processed/Replica/office0")
    parser.add_argument("--dino-model", default="checkpoints/grounding-dino-tiny")
    parser.add_argument("--sam-model", default="checkpoints/sam2.1-hiera-tiny")
    parser.add_argument("--frames", type=int, nargs="+", default=list(range(0, 101, 10)))
    parser.add_argument("--classes", nargs="+",
                        default=["computer monitor", "chair", "desk", "trash can", "door", "sofa"])
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-box-area-fraction", type=float, default=0.40)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--erosion-iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3,
                        help="前 N 帧不计入统计（首次调用有 CUDA 初始化开销）")
    parser.add_argument("--with-write", action="store_true",
                        help="额外执行 PLY / PNG / 预览渲染，用于量化 I/O 代价")
    parser.add_argument("--write-directory", default="outputs/benchmark_latency_io")
    parser.add_argument("--output", default="outputs/benchmark_latency.json")
    parser.add_argument("--label", default="single_process_no_write")
    return parser.parse_args()


def main():
    args = parse_arguments()

    scene_directory = Path(args.scene_directory)
    if not scene_directory.is_absolute():
        scene_directory = (ROOT / scene_directory).resolve()

    print("=" * 78)
    print("单进程延迟基准")
    print("=" * 78)
    print(f"场景      {scene_directory}")
    print(f"帧        {args.frames[0]}..{args.frames[-1]}（{len(args.frames)} 帧）")
    print(f"预热      {args.warmup} 帧不计入统计")
    print(f"写文件    {'是（含预览渲染）' if args.with_write else '否'}")
    print(f"设备      {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU       {torch.cuda.get_device_name(0)}")

    # ---------------- 加载（不计入每帧耗时） ----------------
    print("\n加载数据与模型……")
    sync()
    load_start = time.perf_counter()
    sequence = ReplicaSequence(scene_directory)
    detector = GroundingDinoDetector(model_id=args.dino_model)
    segmenter = Sam2BoxSegmenter(model_id_or_path=args.sam_model)
    sync()
    setup_seconds = time.perf_counter() - load_start
    print(f"  数据 + 两个模型加载完成：{setup_seconds:.2f} s（一次性，不计入每帧）")

    write_directory = None
    if args.with_write:
        write_directory = Path(args.write_directory)
        if not write_directory.is_absolute():
            write_directory = (ROOT / write_directory).resolve()
        write_directory.mkdir(parents=True, exist_ok=True)

    # ---------------- 逐帧 ----------------
    per_frame = []
    print(f"\n开始逐帧计时……\n")
    header = f"{'帧':>6} {'载入':>8} {'DINO':>9} {'SAM2':>9} {'提升':>9} {'写盘':>9} {'合计':>9}  {'实例':>4}"
    print(header)
    print("-" * len(header))

    for index, frame_index in enumerate(args.frames):
        stages = {}

        frame = timed("load", stages, lambda: sequence[frame_index])
        rgb = frame["rgb"]

        detections = timed("detect", stages, detector.predict,
                           rgb=rgb, text_queries=args.classes,
                           box_threshold=args.box_threshold,
                           text_threshold=args.text_threshold,
                           max_box_area_fraction=args.max_box_area_fraction)

        if detections:
            boxes = np.stack([d.box_xyxy for d in detections])
            masks = timed("segment", stages, segmenter.predict, rgb=rgb, boxes_xyxy=boxes)
        else:
            masks = []
            stages["segment"] = 0.0

        def do_lift():
            geometries = []
            for instance_mask in masks:
                geometries.append(lift_mask_to_world(
                    rgb=rgb, depth_m=frame["depth_m"], mask=instance_mask.mask,
                    camera_matrix=frame["camera_matrix"],
                    camera_to_world=frame["camera_to_world"],
                    pixel_stride=args.pixel_stride,
                    erosion_iterations=args.erosion_iterations,
                ))
            return geometries

        geometries = timed("lift", stages, do_lift) if masks else []

        if args.with_write:
            def do_write():
                import cv2
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import open3d as o3d

                frame_directory = write_directory / f"frame_{frame_index:06d}"
                frame_directory.mkdir(parents=True, exist_ok=True)
                for instance_index, geometry in enumerate(geometries):
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(
                        geometry.points_world.astype(np.float64))
                    cloud.colors = o3d.utility.Vector3dVector(
                        geometry.colors.astype(np.float64))
                    o3d.io.write_point_cloud(
                        str(frame_directory / f"{instance_index:02d}.ply"), cloud,
                        write_ascii=False)
                if geometries:
                    figure = plt.figure(figsize=(11, 9))
                    axes = figure.add_subplot(111, projection="3d")
                    for geometry in geometries:
                        points = geometry.points_world
                        axes.scatter(points[:, 0], points[:, 2], points[:, 1], s=1.0)
                    figure.tight_layout()
                    figure.savefig(frame_directory / "preview.png", dpi=180)
                    plt.close(figure)

            timed("write", stages, do_write)

        total = sum(stages.values())
        per_frame.append({"frame": frame_index, "stages": stages, "total_ms": total,
                          "instances": len(geometries)})

        measured = index >= args.warmup
        marker = " " if measured else "*"
        print(f"{marker}{frame_index:>5} "
              f"{stages.get('load', 0):>8.1f} "
              f"{stages.get('detect', 0):>9.1f} "
              f"{stages.get('segment', 0):>9.1f} "
              f"{stages.get('lift', 0):>9.1f} "
              f"{stages.get('write', 0):>9.1f} "
              f"{total:>9.1f}  {len(geometries):>4}")

    # ---------------- 汇总 ----------------
    measured_frames = per_frame[args.warmup:]
    stage_names = ["load", "detect", "segment", "lift", "write"]
    stage_stats = {
        stage: summarise([f["stages"].get(stage, 0.0) for f in measured_frames])
        for stage in stage_names
    }
    stage_stats["total"] = summarise([f["total_ms"] for f in measured_frames])

    print(f"\n{'=' * 78}")
    print(f"汇总（排除前 {args.warmup} 帧预热，n={len(measured_frames)}）")
    print(f"{'=' * 78}")
    print(f"{'阶段':<12} {'均值':>10} {'中位':>10} {'p95':>10} {'最大':>10}")
    print("-" * 56)
    labels = {"load": "载入帧", "detect": "Grounding DINO", "segment": "SAM2",
              "lift": "3D 提升", "write": "写盘+预览", "total": "单帧合计"}
    for stage in stage_names + ["total"]:
        stats = stage_stats[stage]
        if not stats:
            continue
        print(f"{labels[stage]:<12} {stats['mean']:>9.1f}ms {stats['median']:>9.1f}ms "
              f"{stats['p95']:>9.1f}ms {stats['max']:>9.1f}ms")

    total_mean = stage_stats["total"]["mean"]
    print(f"\n吞吐      {1000.0 / total_mean:.2f} FPS（单帧 {total_mean / 1000:.3f} s）")
    inference = stage_stats["detect"]["mean"] + stage_stats["segment"]["mean"]
    print(f"纯 2D 推理 {inference:.1f} ms —— 占单帧 {100.0 * inference / total_mean:.1f}%")
    if args.with_write:
        write_share = 100.0 * stage_stats["write"]["mean"] / total_mean
        print(f"写盘+预览  {stage_stats['write']['mean']:.1f} ms —— 占单帧 {write_share:.1f}%")

    payload = {
        "label": args.label,
        "scene_directory": str(scene_directory),
        "frames": args.frames,
        "warmup": args.warmup,
        "with_write": args.with_write,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "setup_seconds": setup_seconds,
        "classes": args.classes,
        "stage_stats": stage_stats,
        "per_frame": per_frame,
        "note": ("单进程、模型只加载一次、每阶段前后 cuda.synchronize()。"
                 "这是与 run_instance_sequence.py 的关键区别 —— "
                 "后者逐帧 spawn 子进程并重复加载权重，其总耗时不可作为推理性能。"),
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {output_path}")


if __name__ == "__main__":
    main()
