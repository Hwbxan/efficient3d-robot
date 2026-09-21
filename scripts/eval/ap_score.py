"""换掉置信度打分：现在用的是 log1p(观测帧数)，它到底拖了多少后腿？

网格协议 AP 先按分数降序、再扫阈值，所以**排序就是一切**（预测集合固定时）。
当前分数 = log1p(观测帧数)，本质只衡量"这个东西在画面里待了多久"，
跟"这个实例建得准不准"几乎无关——远处的大柜子观测帧数很多但掩码稀烂，
近处的小花瓶只被看到几帧却抠得很准，前者反而排前面。

本脚本把逐帧的 detection_score（Grounding DINO 置信度）和
sam_predicted_iou（SAM2 自估掩码质量）聚合到实例级，比较若干打分下的 AP，
并给出 oracle 排序作为上界。纯离线，不需要重跑序列。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/data/efficient3d_robot")
sys.path.insert(0, str(ROOT))
import eval_mesh_protocol as E  # noqa: E402

SCENES = ["office_0", "office_1", "office_2", "office_3", "office_4",
          "room_0", "room_1", "room_2"]


def aggregate(run_dir):
    """(gid -> 聚合特征) 来自 tracking.json + instances_3d。"""
    track_p = run_dir / "association" / "tracking.json"
    if not track_p.is_file():
        return None
    tr = json.loads(track_p.read_text())
    agg = {}
    for fr in tr["frames"]:
        fi = int(fr["frame_index"])
        inst_p = run_dir / "instances_3d" / f"frame_{fi:06d}" / "instances_3d.json"
        if not inst_p.is_file():
            continue
        per_local = {int(x["local_instance_id"]): x
                     for x in json.loads(inst_p.read_text())}
        # instances_3d.json 里 sam_predicted_iou 恒为 None（写盘时漏传），
        # 真正的掩码质量在 segmentation/frame_*_instances.json 里，这里补上。
        seg_p = run_dir / "segmentation" / f"frame_{fi:06d}_instances.json"
        per_seg = {}
        if seg_p.is_file():
            per_seg = {int(x["local_instance_id"]): x
                       for x in json.loads(seg_p.read_text())}
        for a in fr["associations"]:
            if a.get("global_id") is None:      # 未关联的孤立实例
                continue
            gid = int(a["global_id"])
            rec = per_local.get(int(a["local_instance_id"]))
            if rec is None:
                continue
            seg = per_seg.get(int(a["local_instance_id"])) or {}
            e = agg.setdefault(gid, {"ds": [], "iou": [], "pts": [],
                                     "vdr": [], "labels": {}})
            ds = rec.get("detection_score")
            si = seg.get("sam_predicted_iou")
            if ds is not None:
                e["ds"].append(float(ds))
            if si is not None:
                e["iou"].append(float(si))
            e["pts"].append(float(rec.get("point_count", 0)))
            e["vdr"].append(float(rec.get("valid_depth_ratio", 0)))
            lab = str(rec.get("label", "?")).strip().lower()
            e["labels"][lab] = e["labels"].get(lab, 0) + 1
    out = {}
    for gid, e in agg.items():
        lab = max(e["labels"].items(), key=lambda kv: kv[1])[0]
        out[gid] = {
            "label": lab,
            "n": len(e["ds"]) or len(e["pts"]),
            "det_score": float(np.mean(e["ds"])) if e["ds"] else 0.0,
            "det_score_max": float(np.max(e["ds"])) if e["ds"] else 0.0,
            "sam_iou": float(np.mean(e["iou"])) if e["iou"] else 0.0,
            "points": float(np.mean(e["pts"])) if e["pts"] else 0.0,
            "vdr": float(np.mean(e["vdr"])) if e["vdr"] else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_sam_sam2.1-hiera-base-plus")
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--min-vert", type=int, default=50)
    ap.add_argument("--scenes", default=",".join(SCENES))
    args = ap.parse_args()

    struct_norm = {E.norm(s) for s in E.STRUCTURAL}
    print("=" * 100)
    print(f"打分函数对比  run={args.run_root}  kNN={args.dist}")
    print("=" * 100)

    acc = {}
    for scene in [s.strip() for s in args.scenes.split(",")]:
        run_dir = ROOT / args.run_root / scene.replace("_", "")
        if not run_dir.is_dir():
            continue
        mesh_p = None
        for base in (ROOT / "datasets/raw/replica_v1", ROOT / "datasets/processed/Replica"):
            for cand in (base / scene / "habitat" / "mesh_semantic.ply",
                         base / scene.replace("_", "") / "habitat" / "mesh_semantic.ply"):
                if cand.is_file():
                    mesh_p = cand
                    break
            if mesh_p:
                break
        if mesh_p is None:
            continue
        info = json.loads((mesh_p.parent / "info_semantic.json").read_text())
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

        pred = E.load_prediction(run_dir)
        if pred is None:
            continue
        pxyz, pids, id_score, id_label = pred
        agg = aggregate(run_dir) or {}

        tree = cKDTree(pxyz)
        dist, nn = tree.query(xyz[idx], distance_upper_bound=args.dist)
        hit = np.isfinite(dist)
        pid_at = np.where(hit, pids[np.minimum(nn, len(pids) - 1)], -1)
        gsub = vobj[idx]

        gids = sorted({int(g) for g in np.unique(gsub) if int(g) >= 0})
        gids = [g for g in gids if int((gsub == g).sum()) >= args.min_vert]
        n_gt = len(gids)
        gid_set = {g: k for k, g in enumerate(gids)}

        uniq = [int(p) for p in np.unique(pids)]
        best_iou, best_g = [], []
        for p in uniq:
            bi, bg = 0.0, -1
            for g in gids:
                gm = (gsub == g)
                n_g = int(gm.sum())
                inter = int(((pid_at == p) & gm).sum())
                if inter == 0:
                    continue
                n_p = int((pid_at == p).sum())
                v = inter / max(n_g + n_p - inter, 1)
                if v > bi:
                    bi, bg = v, g
            best_iou.append(bi)
            best_g.append(bg)
        best_iou = np.array(best_iou)
        gt_names = [(obj_meta.get(g) or {}).get("class_name", "?") if g >= 0 else "?"
                    for g in best_g]
        labels = [agg.get(p, {}).get("label", str(id_label.get(p, "?"))) for p in uniq]
        ok = np.array([E.label_match(l, g) for l, g in zip(labels, gt_names)])
        gtid = np.array([gid_set[g] if g >= 0 else -1 for g in best_g])

        n = len(uniq)
        obs = np.array([id_score.get(p, 1.0) for p in uniq])
        ds = np.array([agg.get(p, {}).get("det_score", 0.0) for p in uniq])
        dsmax = np.array([agg.get(p, {}).get("det_score_max", 0.0) for p in uniq])
        si = np.array([agg.get(p, {}).get("sam_iou", 0.0) for p in uniq])
        pts = np.array([agg.get(p, {}).get("points", 0.0) for p in uniq])
        vdr = np.array([agg.get(p, {}).get("vdr", 0.0) for p in uniq])
        nvox = np.array([float((pid_at == p).sum()) for p in uniq])

        scores = {
            "obs(现状)": obs,
            "det_score": ds,
            "det_score_max": dsmax,
            "voxels": np.log1p(nvox),
            "det×obs": ds * obs,
            "det×voxels": ds * np.log1p(nvox),
            "det×vdr": ds * vdr,
            "det×obs×vdr": ds * obs * vdr,
            "oracle(IoU)": best_iou + 1e-6,
        }
        if si.max() > 0:          # SAM2 自估 IoU 只在部分版本里写盘
            scores["sam_iou"] = si
            scores["det×sam"] = ds * si
            scores["det×sam×obs"] = ds * si * obs
        for name, s in scores.items():
            res = {t: E.average_precision(best_iou.copy(), np.where(ok, gtid, -1),
                                          s, t, n_gt) for t in (0.25, 0.5, 0.75)}
            acc.setdefault(name, []).append(res)
        hit_n = int(ok.sum())
        print(f"{scene:<9} GT {n_gt:>3}  预测 {n:>3}  标签对上 {hit_n:>3}  "
              f"sam_iou={'有' if si.max() > 0 else '无'}")
    print("-" * 100)
    print(f"{'打分':<16} {'AP25':>7} {'AP50':>7} {'AP75':>7}")
    rows = []
    for name, v in acc.items():
        f = lambda t: float(np.mean([x[t] for x in v])) * 100  # noqa: E731
        rows.append((f(0.25), name, f(0.5), f(0.75)))
    for a, name, b, c in sorted(rows, reverse=True):
        print(f"{name:<16} {a:>7.1f} {b:>7.1f} {c:>7.1f}")


if __name__ == "__main__":
    main()
