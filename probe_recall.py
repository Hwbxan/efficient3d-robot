"""小物体召回探针：为什么 lamp / book / bottle / plate 从来没被检出？

网格协议下这些类的平均 IoU 只有 1.6%~13%，而它们的可见率高达 87%~99%
——不是没看见，是检测器压根没给框。这里逐配置量：
  · 降 box 阈值有没有用
  · 加高分辨率第二遍（processor 默认上采样到短边 800，小物体在那里就没了）有没有用
  · 各自的时间代价

用法：probe_recall.py --scene office_2 --frames 40
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/data/efficient3d_robot")
sys.path.insert(0, str(ROOT))
from src.datasets.replica_sequence import ReplicaSequence  # noqa: E402
from src.perception.grounding_dino_detector import GroundingDinoDetector  # noqa: E402

CLASSES = [
    "tv screen", "monitor", "tablet", "chair", "stool", "sofa",
    "desk", "table", "desk organizer", "trash can", "basket",
    "tissue box", "door",
    "lamp", "cushion", "book", "pillow", "blanket", "vase",
    "bottle", "plate", "clock", "camera", "bench",
    "potted plant", "plant stand", "cabinet", "picture frame", "nightstand",
    "sculpture", "cloth", "tv stand", "candle", "pot",
    "comforter", "bed", "bowl", "shelf",
]

# 我们 IoU 最低、数量最多的类，重点看它们
WATCH = ["lamp", "book", "bottle", "plate", "tissue box", "tablet",
         "camera", "desk organizer", "clock", "bowl", "box"]

CONFIGS = [
    ("base 0.30", 0.30, None),
    ("base 0.20", 0.20, None),
    ("ms 0.30", 0.30, [{"shortest_edge": 1200, "longest_edge": 2000}]),
    ("ms 0.20", 0.20, [{"shortest_edge": 1200, "longest_edge": 2000}]),
    ("ms2 0.20", 0.20, [{"shortest_edge": 1200, "longest_edge": 2000},
                        {"shortest_edge": 1600, "longest_edge": 2600}]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="office2")
    ap.add_argument("--frames", type=int, default=40, help="采样多少个检测帧")
    ap.add_argument("--dino", default="checkpoints/grounding-dino-base")
    ap.add_argument("--out", default="probe_recall.json")
    args = ap.parse_args()

    seq = ReplicaSequence(ROOT / "datasets/processed/Replica" / args.scene.replace("_", ""))
    det = GroundingDinoDetector(args.dino, device="cuda")
    stride = max(1, len(seq) // args.frames)
    indices = list(range(0, len(seq), stride))[:args.frames]
    print(f"{args.scene}: 采样 {len(indices)} 帧（步长 {stride}）")
    print("=" * 96)

    out = []
    for name, bt, sizes in CONFIGS:
        # 预热
        det.predict(seq[indices[0]]["rgb"], CLASSES, box_threshold=bt,
                    multi_scale_sizes=sizes)
        per_class = Counter()
        n_det = []
        t0 = time.perf_counter()
        for i in indices:
            ds = det.predict(seq[i]["rgb"], CLASSES, box_threshold=bt,
                             multi_scale_sizes=sizes)
            per_class.update(d.label for d in ds)
            n_det.append(len(ds))
        ms = (time.perf_counter() - t0) / len(indices) * 1000.0
        watch = {w: per_class.get(w, 0) for w in WATCH}
        tot_watch = sum(watch.values())
        print(f"{name:<10} 帧均耗时 {ms:7.1f} ms   帧均框数 {np.mean(n_det):5.1f}   "
              f"重点类总命中 {tot_watch:>4}  " +
              "  ".join(f"{w.split()[0]}={watch[w]}" for w in WATCH))
        out.append({"cfg": name, "box_threshold": bt, "sizes": sizes,
                    "ms_per_frame": ms, "mean_dets": float(np.mean(n_det)),
                    "per_class": dict(per_class), "watch": watch})

    (ROOT / args.out).write_text(json.dumps(
        {"scene": args.scene, "frames": len(indices), "configs": out},
        ensure_ascii=False, indent=2, default=float))
    print(f"\n写入 {ROOT / args.out}")


if __name__ == "__main__":
    main()
