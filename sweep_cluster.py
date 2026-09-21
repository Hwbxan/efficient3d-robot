"""验证：对每个实例点云做连通性拆分 / 离群剔除，能否提升网格协议 AP。

假设：跨帧关联把不同物体的观测并成了同一个全局实例（诊断里 desk 实例
覆盖了 45% 的 sofa 顶点）。若如此，实例点云在空间上应该是若干个分离的簇，
用 DBSCAN 切成最大连通分量就能把"粘连"解开。

这是纯后处理，可以在已有 fusion 结果上直接试，不需要重跑检测。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/data/efficient3d_robot")
from eval_mesh_protocol import (  # noqa: E402
    REPLICA, STRUCTURAL, norm, read_semantic_ply, read_xyz_ply,
    vertex_object_ids, average_precision,
)


def load_all(run_dir):
    fusion = run_dir / "fusion_attempt_01"
    imap = json.loads((fusion / "instance_map.json").read_text())
    pts, ids, meta = [], [], {}
    for inst in imap["instances"]:
        gid = int(inst["global_id"])
        p = fusion / inst.get("point_cloud_path", f"individual/G{gid:03d}.ply")
        if not p.is_file():
            continue
        xyz = read_xyz_ply(p)
        if len(xyz) == 0:
            continue
        pts.append(xyz)
        ids.append(np.full(len(xyz), gid, dtype=np.int64))
        meta[gid] = int(inst.get("observation_count", 0)
                        or len(inst.get("source_frames", [])) or 1)
    return np.vstack(pts), np.concatenate(ids), meta, len(imap["instances"])


def outlier_clean(pxyz, pids, method, **kw):
    """温和的离群点剔除：不做簇拆分，只去掉稀疏/偏离点。

    DBSCAN top=1 在 office_0 上收益大但不泛化（office_2 反而变差），
    因为它会切掉被遮挡物体的真实部分。这里改用只剔离群点的保守做法。
    """
    import open3d as o3d
    new_pts, new_ids = [], []
    kept_frac = []
    for gid in np.unique(pids):
        sub = pxyz[pids == gid]
        n0 = len(sub)
        if n0 < 30:
            new_pts.append(sub)
            new_ids.append(np.full(n0, gid, dtype=np.int64))
            kept_frac.append(1.0)
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sub)
        if method == "radius":
            _, idx = pcd.remove_radius_outlier(nb_points=kw["nb_points"],
                                               radius=kw["radius"])
        else:
            _, idx = pcd.remove_statistical_outlier(
                nb_neighbors=kw["nb_neighbors"], std_ratio=kw["std_ratio"])
        sub = sub[np.asarray(idx)]
        if len(sub) == 0:
            continue
        new_pts.append(sub)
        new_ids.append(np.full(len(sub), gid, dtype=np.int64))
        kept_frac.append(len(sub) / n0)
    if not new_pts:
        return None
    return (np.vstack(new_pts), np.concatenate(new_ids),
            float(np.mean(kept_frac)))


def cluster_split(pxyz, pids, eps, min_pts, keep_top=1, adaptive=False):
    """用 DBSCAN 把每个实例切成连通簇，只保留最大的 keep_top 个簇。

    返回新的 (点云, 实例id)：被拆开的簇获得新的 id（原id*1000+k）。
    """
    try:
        import open3d as o3d
    except ImportError:
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise SystemExit("需要 open3d 或 scikit-learn")
        new_pts, new_ids = [], []
        for gid in np.unique(pids):
            m = pids == gid
            sub = pxyz[m]
            mp = max(10, len(sub) // 200) if adaptive else min_pts
            if len(sub) < mp:
                continue
            lab = DBSCAN(eps=eps, min_samples=mp).fit_predict(sub)
            u, c = np.unique(lab[lab >= 0], return_counts=True)
            if len(u) == 0:
                continue
            order = u[np.argsort(-c)][:keep_top]
            for k, cl in enumerate(order):
                sel = lab == cl
                new_pts.append(sub[sel])
                new_ids.append(np.full(int(sel.sum()),
                                       int(gid) * 1000 + k, dtype=np.int64))
        if not new_pts:
            return None
        return np.vstack(new_pts), np.concatenate(new_ids)

    new_pts, new_ids = [], []
    for gid in np.unique(pids):
        m = pids == gid
        sub = pxyz[m]
        mp = max(10, len(sub) // 200) if adaptive else min_pts
        if len(sub) < mp:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sub)
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=mp,
                                               print_progress=False))
        u, c = np.unique(labels[labels >= 0], return_counts=True)
        if len(u) == 0:
            continue
        order = u[np.argsort(-c)][:keep_top]
        for k, cl in enumerate(order):
            sel = labels == cl
            new_pts.append(sub[sel])
            new_ids.append(np.full(int(sel.sum()),
                                   int(gid) * 1000 + k, dtype=np.int64))
    if not new_pts:
        return None
    return np.vstack(new_pts), np.concatenate(new_ids)


def evaluate(xyz, vobj, valid, gts, pxyz, pids, id_score, dist):
    tree = cKDTree(pxyz)
    d, nn = tree.query(xyz, distance_upper_bound=dist)
    pav = np.where(np.isfinite(d), pids[np.minimum(nn, len(pids) - 1)], -1)
    pred_best = {}
    for gid, cname, g in gts:
        pvp = pav[g]
        pvp = pvp[pvp >= 0]
        if len(pvp) == 0:
            continue
        u, cnt = np.unique(pvp, return_counts=True)
        b = int(u[int(np.argmax(cnt))])
        pm = (pav == b) & valid
        iou = int(cnt.max()) / max(int(np.logical_or(g, pm).sum()), 1)
        if b not in pred_best or iou > pred_best[b][1]:
            pred_best[b] = [gid, iou]
    plist = sorted(id_score.keys())
    ious = [pred_best.get(p, [-1, 0.0])[1] for p in plist]
    gm = [pred_best.get(p, [-1, 0.0])[0] for p in plist]
    ss = [id_score[p] for p in plist]
    aps = [average_precision(ious, gm, ss, t, len(gts)) * 100
           for t in (0.25, 0.5, 0.75)]
    cov = float(((pav >= 0) & valid).sum()) / max(int(valid.sum()), 1) * 100
    return aps, cov, len(plist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v3")
    ap.add_argument("--scenes", default="office_0")
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--eps", default="0.05")
    ap.add_argument("--minpts", default="20")
    ap.add_argument("--adaptive", action="store_true",
                    help="min_points 按实例点数自适应：max(10, n//200)")
    ap.add_argument("--outlier", action="store_true",
                    help="跑温和的离群点剔除网格（不做簇拆分）")
    a = ap.parse_args()
    run_root = Path("/data/efficient3d_robot") / a.run_root

    for scene in [s.strip() for s in a.scenes.split(",")]:
        xyz, tri, fo = read_semantic_ply(
            REPLICA / scene / "habitat" / "mesh_semantic.ply")
        vobj = vertex_object_ids(len(xyz), tri, fo)
        info = json.loads((REPLICA / scene / "habitat" /
                           "info_semantic.json").read_text())
        om = {int(o["id"]): o for o in info["objects"]}
        struct_norm = {norm(s) for s in STRUCTURAL}
        is_struct = np.zeros(len(xyz), dtype=bool)
        for gid in np.unique(vobj):
            gid = int(gid)
            if gid < 0:
                continue
            if norm((om.get(gid) or {}).get("class_name", "?")) in struct_norm:
                is_struct[vobj == gid] = True
        valid = (~is_struct) & (vobj >= 0)
        gts = []
        for gid in sorted({int(v) for v in np.unique(vobj) if int(v) >= 0}):
            c = (om.get(gid) or {}).get("class_name", "?")
            if norm(c) in struct_norm:
                continue
            g = (vobj == gid) & valid
            if int(g.sum()) < 50:
                continue
            gts.append((gid, c, g))

        pxyz, pids, meta, n_raw = load_all(run_root / scene.replace("_", ""))
        id_score0 = {g: float(np.log1p(max(o, 1))) for g, o in meta.items()}
        print(f"\n===== {scene}  GT物体 {len(gts)}  原始实例 {n_raw} =====")
        print(f"{'配置':<28}{'实例':>7}{'AP25':>7}{'AP50':>7}{'AP75':>7}{'覆盖%':>8}")
        aps, cov, n = evaluate(xyz, vobj, valid, gts, pxyz, pids,
                               id_score0, a.dist)
        print(f"{'基线（不拆分）':<28}{n:>7}{aps[0]:>7.1f}{aps[1]:>7.1f}"
              f"{aps[2]:>7.1f}{cov:>8.1f}")
        if a.outlier:
            grid = ([("radius", dict(nb_points=nb, radius=r))
                     for r in (0.05, 0.08, 0.12) for nb in (3, 5, 8)]
                    + [("stat", dict(nb_neighbors=nb, std_ratio=s))
                       for s in (1.0, 1.5, 2.0) for nb in (10, 20)])
            for method, kw in grid:
                try:
                    out = outlier_clean(pxyz, pids, method, **kw)
                except Exception as exc:
                    print(f"  {method} {kw} 失败: {exc}")
                    continue
                if out is None:
                    continue
                q, qid, frac = out
                aps, cov, n = evaluate(xyz, vobj, valid, gts, q, qid,
                                       id_score0, a.dist)
                tag = (f"{method} {kw}"
                       f" 保留{frac*100:.0f}%").ljust(34)
                print(f"{tag}{n:>7}{aps[0]:>7.1f}{aps[1]:>7.1f}"
                      f"{aps[2]:>7.1f}{cov:>8.1f}")

        for eps in [float(x) for x in a.eps.split(",")]:
            for mpts in ([int(x) for x in a.minpts.split(",")]
                         if not a.adaptive else [0]):
                for kt in (1, 2):
                    out = cluster_split(pxyz, pids, eps, mpts, kt,
                                        adaptive=a.adaptive)
                    if out is None:
                        continue
                    q, qid = out
                    # 新簇沿用母实例的观测数作为分数
                    sc = {}
                    for g in np.unique(qid):
                        sc[int(g)] = id_score0.get(int(g) // 1000, 1.0)
                    try:
                        aps, cov, n = evaluate(xyz, vobj, valid, gts, q, qid,
                                               sc, a.dist)
                    except Exception as exc:
                        print(f"  eps={eps} mpts={mpts} top={kt} 失败: {exc}")
                        continue
                    tag = f"DBSCAN eps={eps} min={'adap' if a.adaptive else mpts} top={kt}"
                    print(f"{tag:<28}{n:>7}{aps[0]:>7.1f}{aps[1]:>7.1f}"
                          f"{aps[2]:>7.1f}{cov:>8.1f}")


if __name__ == "__main__":
    main()
