"""诊断：Replica 网格协议下，我们的实例地图到底差在哪。

输出两块：
1. 逐 GT 物体的顶点数 / 覆盖率 / IoU / 命中的预测实例及其标签
2. 距离阈值敏感性：dist = 0.02/0.05/0.10/0.20 下的 AP 与覆盖率
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/data/efficient3d_robot")
from eval_mesh_protocol import (  # noqa: E402
    REPLICA, STRUCTURAL, SCENES, norm, read_semantic_ply,
    load_prediction, vertex_object_ids, average_precision,
)


def build(scene, run_root, min_vert=50):
    xyz, tri, fo = read_semantic_ply(REPLICA / scene / "habitat" / "mesh_semantic.ply")
    vobj = vertex_object_ids(len(xyz), tri, fo)
    info = json.loads((REPLICA / scene / "habitat" / "info_semantic.json").read_text())
    om = {int(o["id"]): o for o in info["objects"]}
    px, pid, sc, lab = load_prediction(run_root / scene.replace("_", ""))
    return xyz, vobj, om, px, pid, sc, lab, min_vert


def gt_list(vobj, om, min_vert):
    out = []
    for gid in sorted({int(v) for v in np.unique(vobj) if int(v) >= 0}):
        c = om.get(gid, {}).get("class_name", "?")
        if norm(c) in {norm(s) for s in STRUCTURAL}:
            continue
        g = vobj == gid
        if int(g.sum()) < min_vert:
            continue
        out.append((gid, c, g))
    return out


def detail(scene, run_root, dist):
    xyz, vobj, om, px, pid, sc, lab, mv = build(scene, run_root)
    tree = cKDTree(px)
    d, nn = tree.query(xyz, distance_upper_bound=dist)
    pav = np.where(np.isfinite(d), pid[np.minimum(nn, len(pid) - 1)], -1)
    print(f"\n===== {scene}  dist={dist}  =====")
    print(f"{'类别':<16}{'GT顶点':>8}{'覆盖%':>7}{'IoU':>7}{'命中实例':>9}  预测标签")
    rows = []
    for gid, c, g in gt_list(vobj, om, mv):
        pv = pav[g]
        pvp = pv[pv >= 0]
        cov = float((pav[g] >= 0).mean()) * 100
        if len(pvp) == 0:
            rows.append((c, int(g.sum()), cov, 0.0, -1, "-"))
            continue
        u, cnt = np.unique(pvp, return_counts=True)
        b = int(u[int(np.argmax(cnt))])
        pm = pav == b
        inter = int(cnt.max())
        union = int(np.logical_or(g, pm).sum())
        rows.append((c, int(g.sum()), cov, inter / max(union, 1), b,
                     lab.get(b, "?")))
    for c, n, cov, iou, b, pl in sorted(rows, key=lambda r: -r[2]):
        print(f"{c:<16}{n:>8}{cov:>7.1f}{iou:>7.2f}{b:>9}  {pl}")
    return rows


def sensitivity(scene, run_root):
    xyz, vobj, om, px, pid, sc, lab, mv = build(scene, run_root)
    gts = gt_list(vobj, om, mv)
    print(f"\n===== {scene} 距离阈值敏感性 =====")
    print(f"{'dist':>6}{'覆盖%':>8}{'AP25':>7}{'AP50':>7}{'AP75':>7}   (GT {len(gts)} 物体)")
    tree = cKDTree(px)
    for dist in (0.02, 0.05, 0.10, 0.20, 0.50):
        d, nn = tree.query(xyz, distance_upper_bound=dist)
        pav = np.where(np.isfinite(d), pid[np.minimum(nn, len(pid) - 1)], -1)
        pred_best = {}
        for gid, c, g in gts:
            pvp = pav[g]
            pvp = pvp[pvp >= 0]
            if len(pvp) == 0:
                continue
            u, cnt = np.unique(pvp, return_counts=True)
            b = int(u[int(np.argmax(cnt))])
            pm = pav == b
            iou = int(cnt.max()) / max(int(np.logical_or(g, pm).sum()), 1)
            if b not in pred_best or iou > pred_best[b][1]:
                pred_best[b] = [gid, iou]
        plist = sorted(sc.keys())
        ious = [pred_best.get(p, [-1, 0.0])[1] for p in plist]
        gm = [pred_best.get(p, [-1, 0.0])[0] for p in plist]
        ss = [sc.get(p, 0.0) for p in plist]
        aps = [average_precision(ious, gm, ss, t, len(gts)) * 100
               for t in (0.25, 0.5, 0.75)]
        cov = float(np.isfinite(d).mean()) * 100
        print(f"{dist:>6.2f}{cov:>8.1f}{aps[0]:>7.1f}{aps[1]:>7.1f}{aps[2]:>7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v3")
    ap.add_argument("--scenes", default="office_0")
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--sens", action="store_true")
    a = ap.parse_args()
    run_root = Path("/data/efficient3d_robot") / a.run_root
    for s in a.scenes.split(","):
        s = s.strip()
        if a.sens:
            sensitivity(s, run_root)
        else:
            detail(s, run_root, a.dist)


if __name__ == "__main__":
    main()
