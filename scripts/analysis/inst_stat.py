"""融合实例的观测强度统计：每个实例在多少帧被看到、积了多少体素/点。

用途：判断"实例表面不完整"到底是因为看得少（检测召回低），
还是因为看到了但掩码本身残缺。
"""

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
items = data["instances"] if isinstance(data, dict) else data

print(f"{path.parent.name}  实例数 {len(items)}")

rows = []
for it in items:
    obs = int(it.get("observation_count", 0) or 0)
    frames = it.get("source_frames", []) or []
    rows.append((
        obs,
        len(frames),
        int(it.get("voxel_count", 0) or 0),
        int(it.get("point_count", 0) or 0),
        str(it.get("label", "?")),
        int(it.get("global_id", -1)),
    ))

rows.sort(key=lambda r: -r[0])
print(f"{'gid':>5} {'label':>16} {'obs':>5} {'srcFrames':>10} {'voxel':>8} {'points':>9}")
for obs, nf, vox, pts, label, gid in rows:
    print(f"{gid:>5} {label:>16} {obs:>5} {nf:>10} {vox:>8} {pts:>9}")

if rows:
    obs_arr = [r[0] for r in rows]
    vox_arr = [r[2] for r in rows]
    print(
        f"\n观测帧数: 中位数 {sorted(obs_arr)[len(obs_arr)//2]}  "
        f"最小 {min(obs_arr)} 最大 {max(obs_arr)}  "
        f"| 体素: 中位 {sorted(vox_arr)[len(vox_arr)//2]}  "
        f"最小 {min(vox_arr)} 最大 {max(vox_arr)}"
    )
