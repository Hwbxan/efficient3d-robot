import json, sys
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/efficient3d_robot/outputs/rt8_v3")
tot = []
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    lp = d / "latency.json"
    if not lp.is_file():
        continue
    f = json.loads(lp.read_text())["frames"]
    m = sum(x["total_ms"] for x in f) / len(f)
    det = [x for x in f if x.get("mode", "detect") == "detect"]
    pro = [x for x in f if x.get("mode") == "propagate"]
    dm = sum(x["total_ms"] for x in det) / max(len(det), 1)
    pm = sum(x["total_ms"] for x in pro) / max(len(pro), 1)
    tot.append((d.name, 1000 / m, m, dm, pm, len(f)))
    print(f"{d.name:<10} {1000/m:>6.2f} FPS  {m:>6.1f} ms  检测 {dm:>6.0f} ms  传播 {pm:>5.1f} ms  {len(f)} 帧")
if tot:
    print(f"\n宏平均 FPS = {sum(t[1] for t in tot)/len(tot):.2f}")
