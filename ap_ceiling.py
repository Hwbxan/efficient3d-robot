"""这套轨迹下 AP 的硬上限：只看得见的部分才有可能被重建。

推理（很硬，不依赖任何方法假设）
--------------------------------
GT 物体 i 的顶点集 V_i，轨迹上真正被相机看到的子集 S_i ⊆ V_i。
任何预测实例 P 在网格协议下只能覆盖到 S_i 里的顶点（没看到的面无从重建，
靠猜/靠对称补全属于"幻觉"，不是感知系统的能力）。于是

    IoU_i = |V_i ∩ P| / |V_i ∪ P| ≤ |S_i| / |V_i| = f_i   （可见顶点占比）

f_i 就是物体 i 的 IoU 天花板。给定排序最优（天花板高的排前面），
AP@t ≈ 满足 f_i ≥ t 的物体占比 —— 这是任何方法在这条轨迹上都无法超越的数。

它回答一个关键问题：我们离 OVI-MAP 的 76.7/50.8/22.0 差的一大截里，
有多少是"轨迹根本没看到"造成的（不可修），有多少是我们自己没做好（可修）。

用法：--run-root outputs/rt8_sam_sam2.1-hiera-base-plus [--stride 10] [--tol 0.08]
"""

import argparse
import json
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

SCENES = ["office_0", "office_1", "office_2", "office_3", "office_4",
          "room_0", "room_1", "room_2"]


def find_mesh(scene):
    for base in (ROOT / "datasets/processed/Replica",
                 ROOT / "datasets/raw/replica_v1"):
        p = base / scene / "habitat" / "mesh_semantic.ply"
        if p.is_file():
            return p
        p2 = base / scene.replace("_", "") / "habitat" / "mesh_semantic.ply"
        if p2.is_file():
            return p2
    cand = list((ROOT / "datasets").rglob(f"{scene}/habitat/mesh_semantic.ply"))
    return cand[0] if cand else None


def object_iou(vobj_valid, pred_at_vertex, gid_mask, pid_at_vertex):
    """物体 gid 的最佳预测实例 IoU。"""
    inter_counts = {}
    idx = np.where(gid_mask)[0]
    if len(idx) == 0:
        return 0.0, -1
    for p in pid_at_vertex[idx]:
        if p < 0:
            continue
        inter_counts[int(p)] = inter_counts.get(int(p), 0) + 1
    if not inter_counts:
        return 0.0, -1
    best_pid, inter = max(inter_counts.items(), key=lambda kv: kv[1])
    n_pred = int((pid_at_vertex == best_pid).sum())
    union = len(idx) + n_pred - inter
    return inter / max(union, 1), best_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_sam_sam2.1-hiera-base-plus")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.08)
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--min-vert", type=int, default=50)
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--dump", default="ap_ceiling.json")
    args = ap.parse_args()

    struct_norm = {E.norm(s) for s in E.STRUCTURAL}
    K = create_camera_matrix()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    print("=" * 104)
    print(f"AP 硬上限分析（轨迹可见性）  run={args.run_root}  "
          f"stride={args.stride}  tol={args.tol}  kNN={args.dist}")
    print("=" * 104)

    all_rows = []
    per_class = {}
    for scene in [s.strip() for s in args.scenes.split(",")]:
        mesh_p = find_mesh(scene)
        if mesh_p is None:
            print(f"  [跳过] {scene}: 无 GT mesh")
            continue
        info = E.json.loads((mesh_p.parent / "info_semantic.json").read_text())
        obj_meta = {int(o["id"]): o for o in info["objects"]}
        xyz, tri, face_obj = E.read_semantic_ply(mesh_p)
        vobj = E.vertex_object_ids(len(xyz), tri, face_obj)

        is_struct = np.zeros(len(xyz), dtype=bool)
        for gid in np.unique(vobj):
            gid = int(gid)
            if gid < 0:
                continue
            cname = (obj_meta.get(gid) or {}).get("class_name", "?")
            if E.norm(cname) in struct_norm:
                is_struct[vobj == gid] = True
        valid = (~is_struct) & (vobj >= 0)
        idx = np.where(valid)[0]
        pts = xyz[idx]

        # ---- 可见性 ----
        # 注意：GT mesh 在 raw/replica_v1/<scene>/habitat/，而深度图与轨迹在
        # processed/Replica/<scene 去下划线>/，两处不是同一目录，必须分开取。
        scene_dir = ROOT / "datasets/processed/Replica" / scene.replace("_", "")
        traj = scene_dir / "traj.txt"
        poses = load_camera_poses(traj)
        n_vis = np.zeros(len(pts), dtype=np.int32)
        for fi in range(0, len(poses), args.stride):
            c2w = np.asarray(poses[fi], dtype=np.float64)
            w2c = np.linalg.inv(c2w)
            cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
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
            d = depth[vi, ui].astype(np.float64)
            vis = np.isfinite(d) & (d > 0) & (np.abs(d - cam[inside, 2]) <= args.tol)
            n_vis[np.where(inside)[0][vis]] += 1
        visible = n_vis > 0

        # ---- 我们的实际覆盖 ----
        run_dir = ROOT / args.run_root / scene.replace("_", "")
        pred = E.load_prediction(run_dir)
        pid_at = None
        if pred is not None:
            pxyz, pids, _, _ = pred
            tree = cKDTree(pxyz)
            dist, nn = tree.query(xyz[idx], distance_upper_bound=args.dist)
            hit = np.isfinite(dist)
            pid_at = np.where(hit, pids[np.minimum(nn, len(pids) - 1)], -1)

        # ---- 逐物体 ----
        gids = sorted({int(g) for g in np.unique(vobj) if int(g) >= 0})
        rows = []
        for gid in gids:
            cname = (obj_meta.get(gid) or {}).get("class_name", "?")
            if E.norm(cname) in struct_norm:
                continue
            m = (vobj[idx] == gid)
            if int(m.sum()) < args.min_vert:
                continue
            f = float(visible[m].mean())
            iou = 0.0
            if pid_at is not None:
                iou, _ = object_iou(None, None, m, pid_at)
            rows.append({"cls": cname, "n": int(m.sum()), "f": f, "iou": iou})
            d = per_class.setdefault(E.norm(cname), {"n": 0, "sum_f": 0.0,
                                                     "sum_iou": 0.0, "unreach": 0})
            d["n"] += 1
            d["sum_f"] += f
            d["sum_iou"] += iou
            if f < 0.25:
                d["unreach"] += 1

        N = len(rows)
        c25 = sum(1 for r in rows if r["f"] >= 0.25) / max(N, 1)
        c50 = sum(1 for r in rows if r["f"] >= 0.50) / max(N, 1)
        c75 = sum(1 for r in rows if r["f"] >= 0.75) / max(N, 1)
        a25 = sum(1 for r in rows if r["iou"] >= 0.25) / max(N, 1)
        a50 = sum(1 for r in rows if r["iou"] >= 0.50) / max(N, 1)
        a75 = sum(1 for r in rows if r["iou"] >= 0.75) / max(N, 1)
        mf = float(np.mean([r["f"] for r in rows])) if rows else 0.0
        mi = float(np.mean([r["iou"] for r in rows])) if rows else 0.0
        print(f"{scene:<9} GT物体 {N:>3}  平均可见率 {mf*100:5.1f}%  "
              f"平均IoU {mi*100:5.1f}%  | 可达率 25:{c25*100:5.1f} "
              f"50:{c50*100:5.1f} 75:{c75*100:5.1f}  | 已达 25:{a25*100:5.1f} "
              f"50:{a50*100:5.1f} 75:{a75*100:5.1f}")
        all_rows.append({"scene": scene, "n": N, "ceil25": c25, "ceil50": c50,
                         "ceil75": c75, "got25": a25, "got50": a50, "got75": a75,
                         "mean_f": mf, "mean_iou": mi, "rows": rows})

    print("-" * 104)
    m = lambda k: float(np.mean([r[k] for r in all_rows]))  # noqa: E731
    print(f"宏平均  可达率(硬上限) AP25 {m('ceil25')*100:5.1f}  "
          f"AP50 {m('ceil50')*100:5.1f}  AP75 {m('ceil75')*100:5.1f}")
    print(f"宏平均  实际达成        AP25 {m('got25')*100:5.1f}  "
          f"AP50 {m('got50')*100:5.1f}  AP75 {m('got75')*100:5.1f}")
    print(f"宏平均  平均可见率 {m('mean_f')*100:.1f}%  平均 IoU {m('mean_iou')*100:.1f}%  "
          f"（实现度 {m('mean_iou')/max(m('mean_f'),1e-9)*100:.1f}%）")

    print("\n按类别（出现 >=3 次的类，按平均可见率升序，越靠前越看不见）：")
    print(f"{'类别':<20} {'物体数':>6} {'平均可见率':>10} {'平均IoU':>8} {'完全不可达(<25%)':>18}")
    items = [(k, v) for k, v in per_class.items() if v["n"] >= 3]
    for k, v in sorted(items, key=lambda kv: kv[1]["sum_f"] / kv[1]["n"]):
        print(f"{k:<20} {v['n']:>6} {v['sum_f']/v['n']*100:>9.1f}% "
              f"{v['sum_iou']/v['n']*100:>7.1f}% {v['unreach']:>15}/{v['n']}")

    if args.dump:
        out = ROOT / args.dump
        out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2, default=float))
        print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
