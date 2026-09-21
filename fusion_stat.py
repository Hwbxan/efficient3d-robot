"""融合实例统计：标签分布 + 体素规模分布，看小物体是不是在融合时被大物体吞了。"""
import json
import sys
from pathlib import Path

ROOT = Path("/data/efficient3d_robot")
RUNS = sys.argv[1:] or ["outputs/rt8_sam_sam2.1-hiera-base-plus", "outputs/rt8_v8"]
SCENES = ["office0", "office1", "office2", "office3", "office4",
          "room0", "room1", "room2"]

for run in RUNS:
    for sc in SCENES:
        p = ROOT / run / sc / "fusion_attempt_01" / "instance_map.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        ins = d["instances"]
        vox = sorted(int(i.get("voxel_count", 0)) for i in ins)
        small = [i for i in ins if int(i.get("voxel_count", 0)) < 300]
        print("=" * 96)
        print(run, sc, "instances =", len(ins),
              "voxel median =", vox[len(vox) // 2] if vox else 0,
              "small(<300) =", len(small))
        from collections import Counter
        c = Counter(i.get("label", "?") for i in ins)
        print("  labels:", dict(c.most_common()))
        if small:
            cs = Counter(i.get("label", "?") for i in small)
            print("  small labels:", dict(cs.most_common()))
