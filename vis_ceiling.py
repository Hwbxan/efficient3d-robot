"""覆盖率的不可约上限：物体表面到底有多少是轨迹根本没看到的？

动机：网格协议评测里 office_2 的物体顶点覆盖率只有 57%，而未覆盖 35.6%。
这 35.6% 有两种完全不同的成因：

  (a) 轨迹从未看到（物体背面/底面、被遮挡）——不可约，OVI-MAP 也一样受限；
  (b) 看到了但没检出 / 掩码残缺 —— 可修，值得投入。

两者必须分开，否则会把不可约的损失当成可修的短板，白花力气。
做法：把 GT 顶点按轨迹各帧投影到深度图上，与渲染深度比对判定可见性。

用法：--scene office_2 --run-root outputs/rt8_v5 [--stride 20] [--tol 0.08]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/data/efficient3d_robot")
sys.path.insert(0, str(ROOT))

from src.datasets.replica_sequence import (  # noqa: E402
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    create_camera_matrix,
    load_camera_poses,
    read_depth_image,
)
import eval_mesh_protocol as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="office_2")
    ap.add_argument("--run-root", default="outputs/rt8_v5")
    ap.add_argument("--stride", type=int, default=20,
                    help="每 N 帧采样一次做可见性判定")
    ap.add_argument("--tol", type=float, default=0.08,
                    help="顶点深度与渲染深度的容差（米）")
    ap.add_argument("--dist", type=float, default=0.05,
                    help="覆盖率判定的 kNN 距离（米）")
    ap.add_argument("--per-object", action="store_true")
    args = ap.parse_args()

    scene_dir = ROOT / "datasets/processed/Replica" / args.scene.replace("_", "")
    mesh_p = scene_dir.parent / args.scene / "habitat" / "mesh_semantic.ply"
    if not mesh_p.is_file():
        mesh_p = ROOT / "datasets/processed/Replica" / args.scene / "habitat" / "mesh_semantic.ply"
    if not mesh_p.is_file():
        # 直接按 Replica 原始布局找
        cand = list((ROOT / "datasets").rglob(f"{args.scene}/habitat/mesh_semantic.ply"))
        if not cand:
            print(f"找不到 GT mesh：{args.scene}")
            return
        mesh_p = cand[0]

    info_p = mesh_p.parent / "info_semantic.json"
    info = E.json.loads(info_p.read_text())
    obj_meta = {int(o["id"]): o for o in info["objects"]}

    xyz, tri, face_obj = E.read_semantic_ply(mesh_p)
    vobj = E.vertex_object_ids(len(xyz), tri, face_obj)

    struct_norm = {E.norm(s) for s in E.STRUCTURAL}
    is_struct = np.zeros(len(xyz), dtype=bool)
    for gid in np.unique(vobj):
        gid = int(gid)
        if gid < 0:
            continue
        cname = (obj_meta.get(gid) or {}).get("class_name", "?")
        if E.norm(cname) in struct_norm:
            is_struct[vobj == gid] = True
    valid = (~is_struct) & (vobj >= 0)
    print(f"{args.scene}: GT 网格顶点 {len(xyz)}，待评测（非结构且属于物体）{int(valid.sum())}")

    # ---- 可见性 ----
    poses = load_camera_poses(scene_dir / "traj.txt")
    K = create_camera_matrix()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    idx = np.where(valid)[0]
    pts = xyz[idx]                       # (M,3)
    n_visible_frames = np.zeros(len(pts), dtype=np.int32)

    frame_ids = list(range(0, len(poses), args.stride))
    for fi in frame_ids:
        c2w = np.asarray(poses[fi], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        R, t = w2c[:3, :3], w2c[:3, 3]
        cam = pts @ R.T + t
        z = cam[:, 2]
        ok = z > 0.05
        u = np.full(len(pts), -1.0)
        v = np.full(len(pts), -1.0)
        u[ok] = fx * cam[ok, 0] / z[ok] + cx
        v[ok] = fy * cam[ok, 1] / z[ok] + cy
        inside = (u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT) & ok

        depth = read_depth_image(scene_dir / "results" / f"depth{fi:06d}.png")
        ui = np.clip(u[inside].astype(int), 0, IMAGE_WIDTH - 1)
        vi = np.clip(v[inside].astype(int), 0, IMAGE_HEIGHT - 1)
        d_render = depth[vi, ui].astype(np.float64)
        d_vertex = z[inside]
        vis = np.isfinite(d_render) & (d_render > 0) & (np.abs(d_render - d_vertex) <= args.tol)
        n_visible_frames[np.where(inside)[0][vis]] += 1

    visible = n_visible_frames > 0
    vis_frac = float(visible.mean())
    print(f"采样帧数 {len(frame_ids)}（每 {args.stride} 帧）")
    print(f"曾可见顶点比例：{vis_frac*100:.1f}%   "
          f"（平均每个可见顶点被看到 {n_visible_frames[visible].mean():.1f} 次）")

    # ---- 覆盖率 ----
    run_dir = ROOT / args.run_root / args.scene.replace("_", "")
    pred = E.load_prediction(run_dir)
    if pred is None:
        print("无预测点云")
        return
    pxyz, pids, id_score, id_label = pred
    tree = cKDTree(pxyz)
    dist, _ = tree.query(xyz[idx], distance_upper_bound=args.dist)
    covered = np.isfinite(dist)

    cov_all = float(covered.mean())
    cov_vis = float(covered[visible].mean())
    cov_inv = float(covered[~visible].mean()) if (~visible).any() else float("nan")

    print()
    print("=" * 64)
    print(f"覆盖率（全部待评测顶点）      {cov_all*100:5.1f}%")
    print(f"覆盖率（仅『曾可见』顶点）    {cov_vis*100:5.1f}%   ← 可修空间在这")
    print(f"覆盖率（『从未可见』顶点）    {cov_inv*100:5.1f}%   ← 应接近 0，否则是误匹配")
    print()
    print(f"不可约上限：即使检测完美、掩码完美，覆盖率最高也只能到 {vis_frac*100:.1f}%")
    print(f"其中已实现 {cov_vis*100:.1f}%，还剩 {(vis_frac - cov_all)*100:.1f} 个百分点的可修空间")
    print("=" * 64)

    if args.per_object:
        print(f"\n{'cls':>18} {'GT顶点':>8} {'可见%':>7} {'覆盖%':>7} {'可见且覆盖%':>11}")
        rows = []
        for gid in sorted({int(g) for g in np.unique(vobj) if int(g) >= 0}):
            cname = (obj_meta.get(gid) or {}).get("class_name", "?")
            if E.norm(cname) in struct_norm:
                continue
            m = (vobj[idx] == gid)
            if m.sum() < 50:
                continue
            vf = float(visible[m].mean())
            cf = float(covered[m].mean())
            rows.append((cf, cname, int(m.sum()), vf, cf))
        for cf, cname, n, vf, cf2 in sorted(rows):
            print(f"{cname:>18} {n:>8} {vf*100:>6.1f}% {cf*100:>6.1f}%")


if __name__ == "__main__":
    main()
