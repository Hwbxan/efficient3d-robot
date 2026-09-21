"""统计一次运行里逐帧（含传播帧）出现过的实例标签及观测帧数。"""
import collections
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
cnt = collections.Counter()
obs = collections.defaultdict(set)
files = sorted((p / "instances_3d").glob("frame_*/instances_3d.json"))
if not files:
    files = sorted((p / "instances_3d").glob("*.json"))
for f in files:
    for item in json.loads(f.read_text()):
        lab = str(item.get("label", "?")).strip().lower()
        cnt[lab] += 1
        obs[lab].add(f.parent.name)
print(f"{p}  帧目录 {len(files)}")
print(f"{'标签':<18} {'出现次数':>8} {'出现的帧数':>10}")
for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"{k:<18} {v:>8} {len(obs[k]):>10}")
