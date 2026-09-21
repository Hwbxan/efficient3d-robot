"""语义缺口诊断：几何已经匹配上的预测，有多少是标签说错了？

背景
----
v6 的 class-agnostic AP25（seen）已经到 79.7，但 labeled 只有 53.0。
同一个预测集合、同一个排序，只因为"标签对不上 GT 类名"掉了 26.7 分。
本脚本把这个损失拆开：

  (1) 混淆矩阵：我们说的标签 -> 它实际压中的 GT 类名（只统计 IoU>=0.25 的）
  (2) Oracle 标签上界：把每个预测的标签直接换成它压中的 GT 类名，AP 能到多少
  (3) 排序 + 标签双 Oracle：两者都修好能到多少

(2)-(3) 的差值就是"还能靠更好的打分捞回来多少"，(1) 告诉我们该修哪些类。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/data/efficient3d_robot")
sys.path.insert(0, str(ROOT))
import eval_mesh_protocol as E  # noqa: E402

SCENES = ["office_0", "office_1", "office_2", "office_3", "office_4",
          "room_0", "room_1", "room_2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_sam_sam2.1-hiera-base-plus")
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--min-vert", type=int, default=50)
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--mode", default="labeled", choices=["labeled", "agnostic"])
    args = ap.parse_args()

    struct_norm = {E.norm(s) for s in E.STRUCTURAL}
    print("=" * 100)
    print(f"语义缺口诊断  run={args.run_root}  kNN={args.dist}  min_vert={args.min_vert}")
    print("=" * 100)

    conf = Counter()
    conf_pairs = Counter()
    macro = {"obs": [], "oracle_label": [], "oracle_both": [], "reverse": []}

    for scene in [s.strip() for s in args.scenes.split(",")]:
        mesh_p = None
        for base in (ROOT / "datasets/processed/Replica", ROOT / "datasets/raw/replica_v1"):
            for cand in (base / scene / "habitat" / "mesh_semantic.ply",
                         base / scene.replace("_", "") / "habitat" / "mesh_semantic.ply"):
                if cand.is_file():
                    mesh_p = cand
                    break
            if mesh_p:
                break
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

        run_dir = ROOT / args.run_root / scene.replace("_", "")
        pred = E.load_prediction(run_dir)
        if pred is None:
            print(f"  [跳过] {scene}: 无预测")
            continue
        pxyz, pids, id_score, id_label = pred
        tree = cKDTree(pxyz)
        dist, nn = tree.query(xyz[idx], distance_upper_bound=args.dist)
        hit = np.isfinite(dist)
        pid_at = np.where(hit, pids[np.minimum(nn, len(pids) - 1)], -1)
        gsub = vobj[idx]

        gids = sorted({int(g) for g in np.unique(gsub) if int(g) >= 0})
        gids = [g for g in gids if int((gsub == g).sum()) >= args.min_vert]
        n_gt = len(gids)
        gid_set = {g: k for k, g in enumerate(gids)}

        uniq_pid = [int(p) for p in np.unique(pids)]
        iou_mat = np.zeros((len(uniq_pid), n_gt))
        for gi, g in enumerate(gids):
            gm = (gsub == g)
            n_g = int(gm.sum())
            for pi, p in enumerate(uniq_pid):
                inter = int(((pid_at == p) & gm).sum())
                if inter == 0:
                    continue
                n_p = int((pid_at == p).sum())
                iou_mat[pi, gi] = inter / max(n_g + n_p - inter, 1)

        best_g = np.argmax(iou_mat, axis=1)
        best_iou = iou_mat[np.arange(len(uniq_pid)), best_g]
        gt_of = [gids[k] for k in best_g]

        our_labels = [str(id_label.get(p, "?")) for p in uniq_pid]
        gt_names = [(obj_meta.get(g) or {}).get("class_name", "?") for g in gt_of]

        def ok(our, gt):
            if args.mode == "agnostic":
                return True
            n = E.norm(our)
            g = E.norm(gt)
            return n == g or g in E.SYNONYM.get(n, set())

        match = np.array([ok(o, g) for o, g in zip(our_labels, gt_names)])

        for o, g, i in zip(our_labels, gt_names, best_iou):
            if i >= 0.25 and not ok(o, g):
                conf[g] += 1
                conf_pairs[(E.norm(o), E.norm(g))] += 1

        n_pred = len(uniq_pid)
        base_score = np.array([id_score.get(p, 1.0) for p in uniq_pid])
        rng = np.random.default_rng(0)

        def aps(scores, labels_ok):
            out = {}
            for t in (0.25, 0.50, 0.75):
                out[t] = E.average_precision(best_iou.copy(),
                                             np.where(labels_ok & (best_iou >= 0),
                                                      np.array([gid_set[g] for g in gt_of]), -1),
                                             scores, t, n_gt)
            return out

        a_obs = aps(base_score, match)
        a_lab = aps(base_score, np.ones(n_pred, dtype=bool))
        a_both = aps(best_iou + 1e-6, np.ones(n_pred, dtype=bool))
        a_rev = aps(-base_score, match)
        macro["obs"].append(a_obs)
        macro["oracle_label"].append(a_lab)
        macro["oracle_both"].append(a_both)
        macro["reverse"].append(a_rev)

        print(f"{scene:<9} GT {n_gt:>3} 预测 {n_pred:>3}  几何匹配上(IoU>=.25) {int((best_iou>=0.25).sum()):>3}，"
              f"其中标签错 {int(((best_iou>=0.25) & ~match).sum()):>3}   "
              f"obs {a_obs[0.25]*100:5.1f}/{a_obs[0.5]*100:5.1f}/{a_obs[0.75]*100:5.1f}  "
              f"换对标签 {a_lab[0.25]*100:5.1f}/{a_lab[0.5]*100:5.1f}/{a_lab[0.75]*100:5.1f}")

    print("-" * 100)
    for name in ("obs", "oracle_label", "oracle_both", "reverse"):
        v = macro[name]
        f = lambda t: float(np.mean([x[t] for x in v])) * 100  # noqa: E731
        print(f"{name:<14} AP25 {f(0.25):5.1f}   AP50 {f(0.5):5.1f}   AP75 {f(0.75):5.1f}")

    print("\n几何压中但标签说错的 GT 类别（前 20）：")
    for c, n in conf.most_common(20):
        print(f"  {c:<22} {n}")
    print("\n具体混淆对（我们说的 -> 实际压中的），前 20：")
    for (o, g), n in conf_pairs.most_common(20):
        print(f"  {o:<20} -> {g:<20} {n}")


if __name__ == "__main__":
    main()
