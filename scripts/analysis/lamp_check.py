"""用数值方式确认 office_2 的 lamp 在图像里到底多大、在哪。

不能靠肉眼看图，就用投影：把某个 lamp 的 GT 顶点投到若干帧上，
统计它在画面里的像素包围盒、像素面积、深度。若它确实只占顶部一小条、
且多个 lamp 挤在同一条带里，就能解释为什么 DINO 只给出一整块 ceiling。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/efficient3d_robot")
sys.path.insert(0, str(ROOT))

from src.datasets.replica_sequence import (  # noqa: E402
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    create_camera_matrix,
    load_camera_poses,
)
import eval_mesh_protocol as E  # noqa: E402

scene = "office_2"
scene_dir = ROOT / "datasets/processed/Replica" / scene.replace("_", "")
mesh_p = ROOT / "datasets/processed/Replica" / scene / "habitat" / "mesh_semantic.ply"
if not mesh_p.is_file():
    cand = list((ROOT / "datasets").rglob(f"{scene}/habitat/mesh_semantic.ply"))
    if not cand:
        print(f"找不到 GT mesh：{scene}")
        sys.exit(1)
    mesh_p = cand[0]
print(f"GT mesh: {mesh_p}")
info = E.json.loads((mesh_p.parent / "info_semantic.json").read_text())
obj_meta = {int(o["id"]): o for o in info["objects"]}

xyz, tri, face_obj = E.read_semantic_ply(mesh_p)
vobj = E.vertex_object_ids(len(xyz), tri, face_obj)

lamps = [int(g) for g in sorted(np.unique(vobj))
         if int(g) >= 0
         and E.norm((obj_meta.get(int(g)) or {}).get("class_name", "")) == "lamp"]
print(f"{scene} 的 lamp 对象 id：{lamps}")

poses = load_camera_poses(scene_dir / "traj.txt")
K = create_camera_matrix()
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

frames = list(range(0, 2000, 200))
print(f"\n每帧图像 {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
for gid in lamps:
    m = vobj == gid
    pts = xyz[m]
    areas, ytop, ybot, widths, heights, deps = [], [], [], [], [], []
    for fi in frames:
        c2w = np.asarray(poses[fi], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        ok = z > 0.05
        if ok.sum() < 20:
            continue
        u = fx * cam[ok, 0] / z[ok] + cx
        v = fy * cam[ok, 1] / z[ok] + cy
        inside = (u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT)
        if inside.sum() < 20:
            continue
        u, v = u[inside], v[inside]
        areas.append((u.max() - u.min()) * (v.max() - v.min()))
        ytop.append(v.min()); ybot.append(v.max())
        widths.append(u.max() - u.min()); heights.append(v.max() - v.min())
        deps.append(z[ok][inside].mean())
    if not areas:
        print(f"  lamp {gid}: 这些采样帧里都没进画面")
        continue
    print(f"  lamp {gid}: 可见帧 {len(areas)}/{len(frames)} | "
          f"像素宽 {np.mean(widths):5.0f} 高 {np.mean(heights):4.0f} | "
          f"面积 {np.mean(areas)/1e3:6.1f}k px ({np.mean(areas)/(IMAGE_WIDTH*IMAGE_HEIGHT)*100:4.1f}% 画面) | "
          f"y 范围 {np.mean(ytop):5.0f}~{np.mean(ybot):5.0f} | "
          f"深度 {np.mean(deps):4.2f} m")
