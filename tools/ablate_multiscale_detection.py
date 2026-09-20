"""多尺度检测消融实验（Path C：改善显示器等小物体召回）。

对比两组配置的检测召回与单帧耗时：
  baseline   : 只用 processor 默认分辨率（shortest_edge=800）
  multiscale : 默认分辨率 + 若干额外高分辨率 pass，跨尺度按类别 NMS 融合

之所以要覆盖 processor 的 size 而不是「先放大输入图」：processor 默认会把
任意尺寸输入统一上采样到短边 800，单纯放大 PIL 图像会被再次归一化而完全无效。

用法：
    python -m tools.ablate_multiscale_detection \
        --scene office_1 --frames 60 --extra "1200:2000"
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.datasets.replica_sequence import ReplicaSequence
from src.perception.grounding_dino_detector import GroundingDinoDetector

CATEGORIES = ["computer monitor", "chair", "desk", "trash can", "door", "sofa"]


def parse_sizes(spec):
    """把 "1200:2000,1600:2600" 解析成 size 字典列表。"""
    sizes = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        shortest, longest = part.split(":")
        sizes.append({"shortest_edge": int(shortest), "longest_edge": int(longest)})
    return sizes


def run_config(detector, sequence, indices, box_threshold, max_area, sizes, label):
    per_class = Counter()
    monitor_per_frame = []
    dets_per_frame = []

    # 预热一次，避免首帧的 CUDA 初始化耗时污染计时
    detector.predict(sequence[indices[0]]["rgb"], CATEGORIES,
                     box_threshold=box_threshold,
                     max_box_area_fraction=max_area,
                     multi_scale_sizes=sizes or None)

    start = time.perf_counter()
    for index in indices:
        rgb = sequence[index]["rgb"]
        detections = detector.predict(
            rgb, CATEGORIES,
            box_threshold=box_threshold,
            max_box_area_fraction=max_area,
            multi_scale_sizes=sizes or None,
        )
        counts = Counter(d.label for d in detections)
        per_class.update(counts)
        monitor_per_frame.append(counts.get("computer monitor", 0))
        dets_per_frame.append(len(detections))
    elapsed = time.perf_counter() - start

    n = len(indices)
    return {
        "label": label,
        "extra_sizes": sizes,
        "frames": n,
        "per_class": dict(per_class),
        "monitor_total": per_class.get("computer monitor", 0),
        "monitor_frames_with_any": sum(1 for m in monitor_per_frame if m > 0),
        "monitor_mean_per_frame": float(np.mean(monitor_per_frame)),
        "dets_mean_per_frame": float(np.mean(dets_per_frame)),
        "ms_per_frame": elapsed / n * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", default="datasets/processed/Replica_rendered")
    parser.add_argument("--scene", default="office_1")
    parser.add_argument("--frames", type=int, default=60, help="均匀采样多少帧")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--max-area", type=float, default=0.40)
    parser.add_argument("--extra", default="1200:2000",
                        help="额外 pass 的尺寸，格式 shortest:longest，逗号分隔")
    parser.add_argument("--out", default="outputs/multiscale_ablation.json")
    args = parser.parse_args()

    scene_directory = Path(args.scene_root) / args.scene
    sequence = ReplicaSequence(scene_directory)
    total = len(sequence)
    indices = list(np.linspace(0, total - 1, min(args.frames, total)).astype(int))
    print(f"场景 {args.scene}：共 {total} 帧，采样 {len(indices)} 帧")

    detector = GroundingDinoDetector(model_id="checkpoints/grounding-dino-tiny")
    extra_sizes = parse_sizes(args.extra)

    baseline = run_config(detector, sequence, indices, args.box_threshold,
                          args.max_area, [], "baseline(800)")
    multiscale = run_config(detector, sequence, indices, args.box_threshold,
                            args.max_area, extra_sizes, f"multiscale(+{args.extra})")

    print("\n=== 各类别检测总数 ===")
    print(f"{'类别':<18}{'baseline':>12}{'multiscale':>14}{'变化':>10}")
    for category in CATEGORIES:
        b = baseline["per_class"].get(category, 0)
        m = multiscale["per_class"].get(category, 0)
        print(f"{category:<18}{b:>12}{m:>14}{m - b:>+10}")

    print("\n=== 显示器（computer monitor）专项 ===")
    print(f"  检测总数        : {baseline['monitor_total']} -> {multiscale['monitor_total']}")
    print(f"  有检出的帧数    : {baseline['monitor_frames_with_any']} -> "
          f"{multiscale['monitor_frames_with_any']} / {len(indices)}")
    print(f"  每帧平均个数    : {baseline['monitor_mean_per_frame']:.3f} -> "
          f"{multiscale['monitor_mean_per_frame']:.3f}")

    print("\n=== 代价 ===")
    print(f"  每帧检测数      : {baseline['dets_mean_per_frame']:.2f} -> "
          f"{multiscale['dets_mean_per_frame']:.2f}")
    print(f"  单帧耗时(ms)    : {baseline['ms_per_frame']:.1f} -> "
          f"{multiscale['ms_per_frame']:.1f} "
          f"(x{multiscale['ms_per_frame'] / max(baseline['ms_per_frame'], 1e-6):.2f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "scene": args.scene, "frames": len(indices),
        "box_threshold": args.box_threshold,
        "baseline": baseline, "multiscale": multiscale,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out_path}")


if __name__ == "__main__":
    main()
