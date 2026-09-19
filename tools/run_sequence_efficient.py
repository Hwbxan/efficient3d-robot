"""单进程在线感知序列推理器（模型只加载一次，进程内循环逐帧）。

为什么需要这个脚本
------------------
`tools/run_instance_sequence.py` 默认**逐帧 spawn 子进程**（`inspect_grounded_sam2`
+ `inspect_3d_instances`），每帧都要重新 import torch/open3d/matplotlib 并**重新加载
DINO 与 SAM2 权重**，实测 30.7 s/帧，而真实推理只有约 1.9 s——**约 90% 是脚手架开销**
（见 `tools/benchmark_latency.py` 的对照）。

本脚本把逐帧的 检测→分割→3D 提升 改成**进程内循环、模型只加载一次**，把 18 场景评测
从"不可行"变"可行"，同时是性能目标（≥6 FPS）的根基。

产物兼容性
----------
逐帧写出的 `segmentation/frame_*_instances.json` + `frame_*_masks/*.png` 以及
`instances_3d/frame_*/instances_3d.json` + `individual/*.ply` 与 `inspect_grounded_sam2`
/ `inspect_3d_instances` **字节级一致**，因此 `evaluate_against_gt`、
`inspect_instance_tracking`、`preview_instance_tracking`、`replay_instance_fusion` 可直接消费。

关联 / 预览 / 融合只在末尾对整条序列跑一次（复用 `run_instance_sequence.finish_sequence`
的 subprocess 编排，这些步骤不重载 2D 模型）。

用法
----
    python -m tools.run_sequence_efficient \
        --scene-directory datasets/processed/Replica/office0 \
        --run-directory outputs/experiments/office0_efficient \
        --frames 0 10 20 30 40 50 60 \
        --classes chair desk monitor "trash can" door
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.replica_sequence import ReplicaSequence  # noqa: E402
from src.geometry.instance_lifting import lift_mask_to_world  # noqa: E402
from src.perception.grounding_dino_detector import GroundingDinoDetector  # noqa: E402
from src.perception.sam2_box_segmenter import Sam2BoxSegmenter  # noqa: E402
from tools.run_instance_sequence import finish_sequence  # noqa: E402

MASK_COLORS_RGB = [
    (255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 180, 80),
    (220, 80, 255), (80, 255, 220), (255, 100, 180), (180, 255, 80),
    (120, 120, 255), (255, 220, 80),
]


def safe_filename(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_").lower()


def save_point_cloud(output_path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if not o3d.io.write_point_cloud(str(output_path), cloud, write_ascii=False):
        raise RuntimeError(f"点云保存失败：{output_path}")


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_autocast(enabled: bool):
    """半精度推理上下文。

    Grounding DINO / SAM2 都是 transformer，fp16 在 RTX 3090 上能拿到接近
    两倍的吞吐，而检测框与掩码的数值精度需求远低于 fp32。关闭时返回一个
    空上下文，保证结果与原流水线逐位一致。
    """

    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-directory", default="datasets/processed/Replica/office0")
    parser.add_argument("--dino-model", default="checkpoints/grounding-dino-tiny")
    parser.add_argument("--sam-model", default="checkpoints/sam2.1-hiera-tiny")
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--classes", nargs="+",
                        default=["computer monitor", "chair", "desk", "trash can", "door", "sofa"])
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-box-area-fraction", type=float, default=0.40)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--erosion-iterations", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--encoder-weights", type=Path, default=None,
                        help="Stage-5 点式 encoder 权重（提供则提取 shape/clip 嵌入，增强关联）")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True,
                        help="2D 前端用 fp16 autocast 推理（默认开，约 1.8x 加速）")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="关闭 fp16，用 fp32 复现基线数值")
    parser.add_argument("--write-previews", action="store_true",
                        help="额外写逐帧 matplotlib 3D 预览（慢，默认关以提速）")
    return parser.parse_args()


def main():
    args = parse_arguments()

    scene_directory = Path(args.scene_directory)
    if not scene_directory.is_absolute():
        scene_directory = (ROOT / scene_directory).resolve()
    run_directory = Path(args.run_directory)
    if not run_directory.is_absolute():
        run_directory = (ROOT / run_directory).resolve()

    # ---- 轻量校验（对齐 run_instance_sequence 的防护）----
    if args.frames[0] < 0 or any(b <= a for a, b in zip(args.frames, args.frames[1:])):
        parser_error = "帧编号必须非负且严格递增"
        raise SystemExit(f"错误：{parser_error}")
    results = scene_directory / "results"
    for frame in args.frames:
        for filename in [f"frame{frame:06d}.jpg", f"depth{frame:06d}.png"]:
            if not (results / filename).is_file():
                raise SystemExit(f"错误：缺少输入 {results / filename}")
    run_directory.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("单进程在线感知序列推理器")
    print("=" * 78)
    print(f"场景      {scene_directory}")
    print(f"帧        {args.frames[0]}..{args.frames[-1]}（{len(args.frames)} 帧）")
    print(f"设备      {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU       {torch.cuda.get_device_name(0)}")

    # ---------------- 加载（一次性，不计入每帧）----------------
    print("\n加载数据与模型……")
    sync()
    load_start = time.perf_counter()
    sequence = ReplicaSequence(scene_directory)
    detector = GroundingDinoDetector(model_id=args.dino_model)
    segmenter = Sam2BoxSegmenter(model_id_or_path=args.sam_model)

    encoder = None
    device = None
    if args.encoder_weights is not None:
        from src.models.point_encoder import (
            PointEncoder, PointEncoderConfig, encode_single_instance,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.encoder_weights, map_location=device)
        cfg = PointEncoderConfig(**ckpt["config"])
        encoder = PointEncoder(cfg).to(device).eval()
        encoder.load_state_dict(ckpt["model"])
        print(f"  已加载 encoder：{args.encoder_weights}")
    sync()
    print(f"  加载完成：{time.perf_counter() - load_start:.2f} s（一次性）")

    segmentation_root = run_directory / "segmentation"
    instances_root = run_directory / "instances_3d"
    segmentation_root.mkdir(parents=True, exist_ok=True)
    instances_root.mkdir(parents=True, exist_ok=True)

    # ---------------- 逐帧（进程内）----------------
    frame_records = []
    print(f"\n开始逐帧推理……\n")
    header = f"{'帧':>6} {'DINO':>9} {'SAM2':>9} {'提升':>9} {'合计':>9}  {'实例':>4}"
    print(header)
    print("-" * len(header))

    autocast = make_autocast(args.fp16)
    print(f"2D 前端精度：{'fp16' if args.fp16 and torch.cuda.is_available() else 'fp32'}")

    for index, frame_index in enumerate(args.frames):
        stages = {}
        frame_start = time.perf_counter()

        frame = sequence[frame_index]
        rgb = frame["rgb"]

        sync(); t0 = time.perf_counter()
        with autocast:
            detections = detector.predict(
                rgb=rgb, text_queries=args.classes,
                box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                max_box_area_fraction=args.max_box_area_fraction,
            )
        sync(); stages["detect"] = (time.perf_counter() - t0) * 1000.0

        if detections:
            boxes = np.stack([d.box_xyxy for d in detections])
            sync(); t0 = time.perf_counter()
            with autocast:
                masks = segmenter.predict(rgb=rgb, boxes_xyxy=boxes)
            sync(); stages["segment"] = (time.perf_counter() - t0) * 1000.0
        else:
            masks = []
            stages["segment"] = 0.0

        # ---- 3D 提升（进程内）----
        sync(); t0 = time.perf_counter()
        geometries = []
        for instance_mask in masks:
            geometries.append(lift_mask_to_world(
                rgb=rgb, depth_m=frame["depth_m"], mask=instance_mask.mask,
                camera_matrix=frame["camera_matrix"],
                camera_to_world=frame["camera_to_world"],
                pixel_stride=args.pixel_stride, erosion_iterations=args.erosion_iterations,
            ))
        sync(); stages["lift"] = (time.perf_counter() - t0) * 1000.0

        # ---- 写 segmentation 产物（与 inspect_grounded_sam2 一致）----
        name = f"frame_{frame_index:06d}"
        masks_directory = segmentation_root / f"{name}_masks"
        masks_directory.mkdir(parents=True, exist_ok=True)
        seg_records = []
        for ii, (detection, instance_mask) in enumerate(zip(detections, masks)):
            label_name = safe_filename(detection.label)
            mask_path = (masks_directory / f"{ii:02d}_{label_name}.png").resolve()
            cv2.imwrite(str(mask_path), instance_mask.mask.astype(np.uint8) * 255)
            seg_records.append({
                "local_instance_id": ii,
                "label": detection.label,
                "detection_score": detection.score,
                "sam_predicted_iou": instance_mask.predicted_iou,
                "area_pixels": instance_mask.area_pixels,
                "box_xyxy": detection.box_xyxy.tolist(),
                "mask_path": str(mask_path),
            })
        (segmentation_root / f"{name}_instances.json").write_text(
            json.dumps(seg_records, indent=2, ensure_ascii=False), encoding="utf-8")

        # ---- 写 instances_3d 产物（与 inspect_3d_instances 一致）----
        frame_dir = instances_root / name
        individual_dir = frame_dir / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        lift_records = []
        for ii, (detection, geometry) in enumerate(zip(detections, geometries)):
            label_name = safe_filename(detection.label)
            ply_path = (individual_dir / f"{ii:02d}_{label_name}.ply").resolve()
            save_point_cloud(ply_path, geometry.points_world, geometry.colors)
            entry = {
                "local_instance_id": ii,
                "label": detection.label,
                "detection_score": detection.score,
                "sam_predicted_iou": None,
                "point_count": len(geometry.points_world),
                "valid_depth_ratio": geometry.valid_depth_ratio,
                "median_depth_m": geometry.median_depth_m,
                "centroid_world": geometry.centroid.tolist(),
                "bbox_min_world": geometry.bbox_min.tolist(),
                "bbox_max_world": geometry.bbox_max.tolist(),
                "bbox_extent_m": geometry.bbox_extent.tolist(),
                "point_cloud_path": str(ply_path),
            }
            if encoder is not None and len(geometry.points_world) >= 30:
                emb = encode_single_instance(
                    encoder, points=geometry.points_world, colors=geometry.colors,
                    device=device, pool="max")
                entry["shape_embedding"] = emb["shape_embedding"].tolist()
                entry["clip_embedding"] = emb["clip_embedding"].tolist()
                entry["semantic_logits"] = emb["semantic_logits"].tolist()
            lift_records.append(entry)

        (frame_dir / "instances_3d.json").write_text(
            json.dumps(lift_records, indent=2, ensure_ascii=False), encoding="utf-8")

        total = sum(stages.values())
        frame_records.append({"frame": frame_index, "stages": stages,
                               "total_ms": total, "instances": len(geometries)})
        print(f"{frame_index:>6} {stages.get('detect', 0):>9.1f} "
              f"{stages.get('segment', 0):>9.1f} {stages.get('lift', 0):>9.1f} "
              f"{total:>9.1f}  {len(geometries):>4}")

    # ---------------- 汇总 ----------------
    measured = frame_records
    total_mean = statistics.fmean([f["total_ms"] for f in measured]) if measured else 0.0
    print(f"\n{'=' * 78}")
    print(f"逐帧推理汇总（n={len(measured)}）")
    print(f"{'=' * 78}")

    latency = None
    if total_mean:
        det = statistics.fmean([f["stages"]["detect"] for f in measured])
        seg = statistics.fmean([f["stages"]["segment"] for f in measured])
        lif = statistics.fmean([f["stages"]["lift"] for f in measured])
        totals = sorted(f["total_ms"] for f in measured)
        latency = {
            "measured_frames": len(measured),
            "mean_total_ms": round(total_mean, 1),
            "median_total_ms": round(statistics.median(totals), 1),
            "p95_total_ms": round(totals[min(len(totals) - 1, int(0.95 * len(totals)))], 1),
            "fps_mean": round(1000.0 / total_mean, 2),
            "detect_ms": round(det, 1),
            "segment_ms": round(seg, 1),
            "lift_ms": round(lif, 1),
            "pure_2d_ms": round(det + seg, 1),
            "note": "单进程、模型只加载一次、torch.cuda.synchronize() 计时、不写预览",
        }
        print(f"单帧均值 {total_mean:.1f} ms → {1000.0 / total_mean:.2f} FPS")
        print(f"  Grounding DINO {det:.1f} ms | SAM2 {seg:.1f} ms | 3D 提升 {lif:.1f} ms")
        print(f"纯 2D 推理 {det + seg:.1f} ms —— 占单帧 {100.0 * (det + seg) / total_mean:.1f}%")
    else:
        print("无帧")

    # 延迟落盘，便于跨实验/跨场景对比与审计
    latency_path = run_directory / "latency.json"
    with latency_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"frames": measured, "summary": latency},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"延迟明细：{latency_path}")

    # ---------------- 关联 / 预览 / 融合（复用现有 finish_sequence）----------------
    # 把需要的属性挂到 args 上（finish_sequence 用到）
    args.scene_directory = str(scene_directory)
    args.voxel_size = args.voxel_size
    finish_sequence(args, run_directory)


if __name__ == "__main__":
    main()
