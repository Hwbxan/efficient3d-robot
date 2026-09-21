"""SAM2 骨干对照：hiera-tiny（v5 现役） vs hiera-base-plus，同一批场景同一套参数。"""

import subprocess
import sys

ROOT = "/data/efficient3d_robot"
SCENES = ["office_0", "office_1", "office_2", "office_4", "room_0", "room_1"]


def eval_one(run_root, scenes):
    cmd = [sys.executable, "eval_mesh_protocol.py", "--run-root", run_root,
           "--scenes", ",".join(scenes)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 8 and parts[0] in SCENES and parts[1] == "GT":
            # scene GT n 预测 a/b 物体顶点覆盖 cov% AP25 a AP50 b AP75 c R25 R50 R75
            try:
                res[parts[0]] = {
                    "cov": float(parts[6].rstrip("%")),
                    "ap25": float(parts[8]),
                    "ap50": float(parts[10]),
                    "ap75": float(parts[12]),
                }
            except ValueError:
                pass
    return res


tiny = eval_one("outputs/rt8_v5", SCENES)
bp = eval_one("outputs/rt8_sam_sam2.1-hiera-base-plus", SCENES)

print(f"{'scene':>9} {'AP25 tiny':>10} {'→ bp':>8} {'Δ':>7} | "
      f"{'AP50 tiny':>10} {'→ bp':>8} {'Δ':>7} | "
      f"{'AP75 tiny':>10} {'→ bp':>8} {'Δ':>7}")
print("-" * 92)
sums = {k: [0.0, 0.0] for k in ("ap25", "ap50", "ap75")}
n = 0
for s in SCENES:
    a, b = tiny.get(s), bp.get(s)
    if not a or not b:
        print(f"{s:>9} 数据缺失")
        continue
    n += 1
    for k in sums:
        sums[k][0] += a[k]
        sums[k][1] += b[k]
    print(f"{s:>9} {a['ap25']:>10.1f} {b['ap25']:>8.1f} {b['ap25']-a['ap25']:>+7.1f} | "
          f"{a['ap50']:>10.1f} {b['ap50']:>8.1f} {b['ap50']-a['ap50']:>+7.1f} | "
          f"{a['ap75']:>10.1f} {b['ap75']:>8.1f} {b['ap75']-a['ap75']:>+7.1f}")

print("-" * 92)
print(f"{'宏平均':>9} {sums['ap25'][0]/n:>10.1f} {sums['ap25'][1]/n:>8.1f} "
      f"{(sums['ap25'][1]-sums['ap25'][0])/n:>+7.1f} | "
      f"{sums['ap50'][0]/n:>10.1f} {sums['ap50'][1]/n:>8.1f} "
      f"{(sums['ap50'][1]-sums['ap50'][0])/n:>+7.1f} | "
      f"{sums['ap75'][0]/n:>10.1f} {sums['ap75'][1]/n:>8.1f} "
      f"{(sums['ap75'][1]-sums['ap75'][0])/n:>+7.1f}")
print(f"\n参与统计的场景数：{n}")
