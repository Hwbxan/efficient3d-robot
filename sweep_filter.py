"""扫描"弱证据实例过滤"超参对网格协议 AP 的影响。

一次加载 mesh + 全量预测点云，之后每个配置只改 pid 掩码，避免重复 IO。
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
        a, b = inst.get("bbox_min_world"), inst.get("bbox_max_world")
        meta[gid] = dict(
            label=inst.get("label", "?"),
            obs=int(inst.get("observation_count", 0)
                    or len(inst.get("source_frames", [])) or 1),
            voxel=int(inst.get("voxel_count", 0)),
            extent=float(max(np.asarray(b) - np.asarray(a))) if a and b else 0.0,
        )
    return np.vstack(pts), np.concatenate(ids), meta, imap


def evaluate(scene, run_dir, dist, min_obs, max_extent, min_voxel,
             cache, verbose=False):
    if scene not in cache:
        xyz, tri, fo = read_semantic_ply(
            REPLICA / scene / "habitat" / "mesh_semantic.ply")
        vobj = vertex_object_ids(len(xyz), tri, fo)
        info = json.loads((REPLICA / scene / "habitat" /
                           "info_semantic.json").read_text())
        om = {int(o["id"]): o for o in info["objects"]}
        pxyz, pids, meta, imap = load_all(run_dir / scene.replace("_", ""))
        struct_norm = {norm(s) for s in STRUCTURAL}
        is_struct = np.zeros(len(xyz), dtype=bool)
        for gid in np.unique(vobj):
            gid = int(gid)
            if gid < 0:
                continue
            if norm((om.get(gid) or {}).get("class_name", "?")) in struct_norm:
                is_struct[vobj == gid] = True
        valid = ~is_struct
        gts = []
        for gid in sorted({int(v) for v in np.unique(vobj) if int(v) >= 0}):
            c = (om.get(gid) or {}).get("class_name", "?")
            if norm(c) in struct_norm:
                continue
            g = (vobj == gid) & valid
            if int(g.sum()) < 50:
                continue
            gts.append((gid, c, g))
        cache[scene] = dict(xyz=xyz, vobj=vobj, valid=valid, gts=gts,
                            pxyz=pxyz, pids=pids, meta=meta,
                            n_raw=len(imap["instances"]))

    c = cache[scene]
    keep = {g for g, m in c["meta"].items()
            if m["obs"] >= min_obs and m["voxel"] >= min_voxel
            and (max_extent <= 0 or m["extent"] <= max_extent)}
    sel = np.isin(c["pids"], list(keep))
    if sel.sum() < 50:
        return None
    pxyz, pids = c["pxyz"][sel], c["pids"][sel]
    id_score = {g: float(np.log1p(max(c["meta"][g]["obs"], 1))) for g in keep}

    tree = cKDTree(pxyz)
    d, nn = tree.query(c["xyz"], distance_upper_bound=dist)
    pav = np.where(np.isfinite(d), pids[np.minimum(nn, len(pids) - 1)], -1)
    pred_best = {}
    for gid, cname, g in c["gts"]:
        pvp = pav[g]
        pvp = pvp[pvp >= 0]
        if len(pvp) == 0:
            continue
        u, cnt = np.unique(pvp, return_counts=True)
        b = int(u[int(np.argmax(cnt))])
        pm = (pav == b) & c["valid"]
        iou = int(cnt.max()) / max(int(np.logical_or(g, pm).sum()), 1)
        if b not in pred_best or iou > pred_best[b][1]:
            pred_best[b] = [gid, iou]
    plist = sorted(id_score.keys())
    ious = [pred_best.get(p, [-1, 0.0])[1] for p in plist]
    gm = [pred_best.get(p, [-1, 0.0])[0] for p in plist]
    ss = [id_score[p] for p in plist]
    n_gt = len(c["gts"])
    aps = [average_precision(ious, gm, ss, t, n_gt) * 100
           for t in (0.25, 0.5, 0.75)]
    cov = float(((pav >= 0) & c["valid"]).sum()) / max(int(c["valid"].sum()), 1)
    return dict(n_gt=n_gt, n_pred=len(plist), n_raw=c["n_raw"],
                ap25=aps[0], ap50=aps[1], ap75=aps[2], cov=cov * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v3")
    ap.add_argument("--scenes", default="office_0")
    ap.add_argument("--dist", type=float, default=0.05)
    a = ap.parse_args()
    run_root = Path("/data/efficient3d_robot") / a.run_root
    scenes = [s.strip() for s in a.scenes.split(",")]
    cache = {}

    print(f"{'min_obs':>8}{'max_ext':>9}{'min_vox':>9}{'n_pred':>8}"
          f"{'AP25':>7}{'AP50':>7}{'AP75':>7}{'覆盖%':>8}")
    rows = []
    for min_obs in (0, 10, 20, 30, 50, 80):
        for max_extent in (0.0, 2.5, 3.0, 4.0):
            for min_voxel in (0, 200, 1000):
                acc = []
                for s in scenes:
                    r = evaluate(s, run_root, a.dist, min_obs, max_extent,
                                 min_voxel, cache)
                    if r:
                        acc.append(r)
                if not acc:
                    continue
                m = lambda k: np.mean([r[k] for r in acc])  # noqa: E731
                npred = int(np.mean([r["n_pred"] for r in acc]))
                nraw = int(np.mean([r["n_raw"] for r in acc]))
                row = (min_obs, max_extent, min_voxel, npred, nraw,
                       m("ap25"), m("ap50"), m("ap75"), m("cov"))
                rows.append(row)
                print(f"{min_obs:>8}{max_extent:>9.1f}{min_voxel:>9}"
                      f"{f'{npred}/{nraw}':>8}{row[5]:>7.1f}{row[6]:>7.1f}"
                      f"{row[7]:>7.1f}{row[8]:>8.1f}")
    if rows:
        best = max(rows, key=lambda r: r[5] + r[6])
        print(f"\n按 AP25+AP50 最优：min_obs={best[0]} max_extent={best[1]} "
              f"min_voxel={best[2]}  → AP25 {best[5]:.1f} AP50 {best[6]:.1f} "
              f"AP75 {best[7]:.1f} 预测 {best[3]}/{best[4]}")


if __name__ == "__main__":
    main()
