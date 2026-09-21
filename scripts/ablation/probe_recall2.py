"""提示词竞争探针：小物体检不出来，是"被 38 个提示词挤掉了"还是"模型看不见"？

Grounding DINO 对每个框在全部短语上做 softmax，常见大物体（chair / table /
cushion）分数高，小物体即使有响应也会被压到阈值以下。
验证办法：只把小物体那一组提示词单独喂进去，看它们出不出来。
如果出来了，就能靠"分组多遍查询"低成本提升召回。
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

BIG = [
    "tv screen", "monitor", "chair", "stool", "sofa", "desk", "table",
    "trash can", "basket", "door", "cushion", "pillow", "blanket",
    "potted plant", "plant stand", "cabinet", "nightstand", "shelf",
    "comforter", "bed", "bench", "vase", "picture frame",
]

SMALL = [
    "lamp", "book", "bottle", "plate", "clock", "camera", "tablet",
    "tissue box", "bowl", "box", "candle", "sculpture", "desk organizer",
    "cloth", "tv stand", "pot",
]

CONFIGS = [
    ("38合1 @0.20", BIG + SMALL, 0.20, None),
    ("仅小物体 @0.20", SMALL, 0.20, None),
    ("仅小物体 @0.15", SMALL, 0.15, None),
    ("仅小物体 @0.10", SMALL, 0.10, None),
    ("仅小物体 @0.15+ms", SMALL, 0.15,
     [{"shortest_edge": 1200, "longest_edge": 2000}]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="office_2")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--dino", default="checkpoints/grounding-dino-base")
    ap.add_argument("--out", default="probe_recall2.json")
    args = ap.parse_args()

    seq = ReplicaSequence(ROOT / "datasets/processed/Replica" / args.scene.replace("_", ""))
    det = GroundingDinoDetector(args.dino, device="cuda")
    stride = max(1, len(seq) // args.frames)
    indices = list(range(0, len(seq), stride))[:args.frames]
    print(f"{args.scene}: 采样 {len(indices)} 帧")
    print("=" * 92)

    out = []
    for name, prompts, bt, sizes in CONFIGS:
        det.predict(seq[indices[0]]["rgb"], prompts, box_threshold=bt,
                    multi_scale_sizes=sizes)
        per = Counter()
        n = []
        t0 = time.perf_counter()
        for i in indices:
            ds = det.predict(seq[i]["rgb"], prompts, box_threshold=bt,
                             multi_scale_sizes=sizes)
            per.update(d.label for d in ds)
            n.append(len(ds))
        ms = (time.perf_counter() - t0) / len(indices) * 1000.0
        hit_small = {w: per.get(w, 0) for w in SMALL}
        print(f"{name:<18} {ms:7.1f} ms  框数/帧 {np.mean(n):5.1f}  小物体总命中 "
              f"{sum(hit_small.values()):>4}  | " +
              " ".join(f"{k}={v}" for k, v in hit_small.items() if v))
        out.append({"cfg": name, "n_prompts": len(prompts), "box_threshold": bt,
                    "sizes": sizes, "ms_per_frame": ms,
                    "mean_dets": float(np.mean(n)), "per_class": dict(per)})

    (ROOT / args.out).write_text(json.dumps(
        {"scene": args.scene, "frames": len(indices), "configs": out},
        ensure_ascii=False, indent=2, default=float))
    print(f"\n写入 {ROOT / args.out}")


if __name__ == "__main__":
    main()
