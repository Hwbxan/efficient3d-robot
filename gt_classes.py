"""每个场景的 GT 物体类别清单 + 我们对应的 IoU，定位"哪些类我们完全没抓住"。

读 ap_ceiling.json（已含逐物体的可见率与 IoU），按 (平均IoU 升序) 输出，
并单列"场景里真的有、但我们 IoU < 5%"的类 —— 那才是纯漏检。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/efficient3d_robot")
data = json.loads((ROOT / "ap_ceiling.json").read_text())

per_cls = defaultdict(lambda: {"n": 0, "iou": 0.0, "vis": 0.0, "zero": 0})
per_scene_cls = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
for rec in data:
    sc = rec["scene"]
    for r in rec["rows"]:
        c = r["cls"]
        per_cls[c]["n"] += 1
        per_cls[c]["iou"] += r["iou"]
        per_cls[c]["vis"] += r["f"]
        if r["iou"] < 0.05:
            per_cls[c]["zero"] += 1
        s = per_scene_cls[sc][c]
        s[0] += 1
        s[1] += r["iou"]

print(f"{'类别':<20} {'物体数':>6} {'平均可见率':>10} {'平均IoU':>9} {'几乎没抓住(<5%)':>16}")
items = sorted(per_cls.items(), key=lambda kv: kv[1]["iou"] / kv[1]["n"])
for c, v in items:
    n = v["n"]
    print(f"{c:<20} {n:>6} {v['vis']/n*100:>9.1f}% {v['iou']/n*100:>8.1f}% "
          f"{v['zero']:>13}/{n}")

tot = sum(v["n"] for v in per_cls.values())
w = sum(v["iou"] for v in per_cls.values()) / tot
print(f"\n合计物体 {tot}，全体平均 IoU {w*100:.1f}%")

if len(sys.argv) > 1:
    sc = sys.argv[1]
    print(f"\n{sc} 的 GT 类别明细：")
    for c, (n, s) in sorted(per_scene_cls[sc].items(), key=lambda kv: kv[1][1] / kv[1][0]):
        print(f"  {c:<20} {n:>3} 个   平均IoU {s/n*100:5.1f}%")
