"""汇总 8 个场景的实时性能指标（latency.json summary）。"""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
scenes = "office0 office1 office2 office3 office4 room0 room1 room2".split()

keys = [
    "fps_mean", "mean_total_ms", "median_total_ms", "p95_total_ms",
    "detect_frame_mean_ms", "propagate_frame_mean_ms",
    "detect_frames", "propagate_frames", "detect_interval",
]

print(f"{'scene':>9} " + " ".join(f"{k:>22}" for k in keys))
vals = {k: [] for k in keys if k not in ("detect_frames", "propagate_frames", "detect_interval")}
for s in scenes:
    p = root / s / "latency.json"
    if not p.is_file():
        print(f"{s:>9} 缺 latency.json")
        continue
    sm = json.loads(p.read_text())["summary"]
    print(f"{s:>9} " + " ".join(
        f"{sm.get(k, float('nan')):>22.2f}" if isinstance(sm.get(k), (int, float))
        else f"{str(sm.get(k)):>22}" for k in keys))
    for k in vals:
        v = sm.get(k)
        if isinstance(v, (int, float)):
            vals[k].append(v)

print(f"\n{'均值':>9} " + " ".join(
    f"{sum(vals[k])/len(vals[k]):>22.2f}" if vals[k] else f"{'-':>22}"
    for k in keys if k in vals))
