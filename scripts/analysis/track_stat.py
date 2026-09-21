"""统计 tracking.json 里的轨道：标签、观测帧数、体素数，看小物体是否被切成碎片。"""
import collections
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
frames = d["frames"] if isinstance(d, dict) else d
agg = {}
for fr in frames:
    fi = fr.get("frame_index")
    for a in fr.get("associations", []):
        gid = a.get("global_id")
        lab = str(a.get("label", "?")).strip().lower()
        e = agg.setdefault(gid, {"label": lab, "frames": 0, "labels": collections.Counter()})
        e["frames"] += 1
        e["labels"][lab] += 1

print(f"轨道数 {len(agg)}")
by_lab = collections.defaultdict(list)
for gid, e in agg.items():
    top = e["labels"].most_common(1)[0][0]
    by_lab[top].append((gid, e["frames"]))
print(f"{'标签':<18} {'轨道数':>6} {'各轨道观测帧数':>40}")
for lab, items in sorted(by_lab.items(), key=lambda kv: -len(kv[1])):
    fr = sorted([f for _, f in items], reverse=True)
    show = ",".join(str(x) for x in fr[:12])
    print(f"{lab:<18} {len(items):>6}   {show}")
tot = sum(len(v) for v in by_lab.values())
frag = sum(1 for v in by_lab.values() for _, f in v if f <= 2)
print(f"\n总轨道 {tot}，其中观测 <=2 帧的碎片轨道 {frag}（{frag/max(tot,1)*100:.0f}%）")
