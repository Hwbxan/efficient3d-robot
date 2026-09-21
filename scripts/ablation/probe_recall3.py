"""措辞探针：lamp / clock / camera / book 一个都检不出，是不是提示词说法不对？

Replica 的 "lamp" 多是吸顶灯具，说 "lamp" 未必触发；换成
"ceiling light" / "light fixture" 试试。同一个开放词汇系统，用户本来就会换说法，
这不属于针对测试集调参。
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

# (我们原来的说法, 候选说法)
GROUPS = {
    "lamp": ["lamp", "ceiling light", "light fixture", "ceiling lamp",
             "recessed light", "fluorescent light", "light"],
    "clock": ["clock", "wall clock"],
    "camera": ["camera", "security camera", "webcam"],
    "book": ["book", "books", "stack of books"],
    "bottle": ["bottle", "water bottle", "plastic bottle"],
    "plate": ["plate", "paper plate"],
    "tissue-paper": ["tissue box", "tissue paper", "box of tissues", "napkin box"],
    "tablet": ["tablet", "ipad", "electronic tablet"],
    "bowl": ["bowl", "fruit bowl"],
    "box": ["box", "cardboard box", "package"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="office_2")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--dino", default="checkpoints/grounding-dino-base")
    ap.add_argument("--out", default="probe_recall3.json")
    args = ap.parse_args()

    seq = ReplicaSequence(ROOT / "datasets/processed/Replica" / args.scene.replace("_", ""))
    det = GroundingDinoDetector(args.dino, device="cuda")
    stride = max(1, len(seq) // args.frames)
    indices = list(range(0, len(seq), stride))[:args.frames]
    print(f"{args.scene}: 采样 {len(indices)} 帧   探测措辞对目标类召回的影响")
    print("=" * 92)

    out = []
    for gt_name, phrases in GROUPS.items():
        for bt in (0.20, 0.10):
            det.predict(seq[indices[0]]["rgb"], phrases, box_threshold=bt)
            per = Counter()
            t0 = time.perf_counter()
            for i in indices:
                ds = det.predict(seq[i]["rgb"], phrases, box_threshold=bt)
                per.update(d.label for d in ds)
            ms = (time.perf_counter() - t0) / len(indices) * 1000.0
            s = "  ".join(f"{k}={v}" for k, v in sorted(per.items(), key=lambda x: -x[1]))
            print(f"  {gt_name:<14} @box{bt:<5} {ms:6.0f}ms  {s if s else '(零命中)'}")
            out.append({"gt": gt_name, "phrases": phrases, "box_threshold": bt,
                        "per_class": dict(per), "ms": ms})

    (ROOT / args.out).write_text(json.dumps(
        {"scene": args.scene, "frames": len(indices), "results": out},
        ensure_ascii=False, indent=2, default=float))
    print(f"\n写入 {ROOT / args.out}")


if __name__ == "__main__":
    main()
