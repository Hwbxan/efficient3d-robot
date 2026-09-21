"""统计一个场景所有帧的帧级检测标签频次，看哪些 GT 类别根本没被检出过。"""

import collections
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.glob("frame_*/instances_3d.json"))
counter = collections.Counter()
per_frame = []
for f in files:
    items = json.loads(f.read_text())
    per_frame.append(len(items))
    for it in items:
        counter[str(it.get("label", "?")).strip().lower()] += 1

print(f"{root.parent.name}: {len(files)} 帧，帧均实例 {sum(per_frame)/max(len(per_frame),1):.2f}")
print(f"{'label':>20} {'出现帧次数':>10}")
for k, v in counter.most_common():
    print(f"{k:>20} {v:>10}")
