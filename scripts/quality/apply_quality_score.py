"""把"实例质量打分器"写进融合输出。

为什么需要
----------
AP 是 PR 曲线下面积，预测集合固定时完全由排序决定。融合阶段一直用
log(1+观测帧数) 当置信度，它与真实 IoU 的 Spearman 只有 +0.502：
一个沙发在 359 帧里出现过，但每帧只看到一小撮体素，分数虚高；
而一个被 30 帧稳定确认的小台灯分数很低。真正有信号的是**体素级的
多帧确认度**（fo_mean，Spearman +0.562）。

做法
----
用 Ridge 把"无 GT 的系统自身特征"映射到真实 IoU，留一场景交叉验证：
预测场景 s 时只许用其余 7 个场景训练。训练集用的是 Replica 标注
（IoU 来自评测协议），特征全部来自系统输出，不含任何 GT 信息。

写回 instance_map.json 的 quality_score 字段，评测脚本优先读它。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/efficient3d_robot")
import eval_mesh_protocol as E
from dump_rank_feat import det_feats
from sklearn.linear_model import Ridge

FUSION = os.environ.get("FUSION_DIR", "fusion_attempt_01")


def geom_feats(inst, fdir, vs):
    a = np.asarray(inst.get("bbox_min_world", [0, 0, 0]), float)
    b = np.asarray(inst.get("bbox_max_world", [0, 0, 0]), float)
    dim = b - a
    vol = float(max(np.prod(np.maximum(dim, 1e-6)), 1e-9))
    nv = int(inst.get("voxel_count", 0))
    fo = None
    sp = inst.get("voxel_statistics_path")
    if sp:
        p = Path(sp)
        if not p.is_absolute():
            p = fdir / sp
        if p.is_file():
            try:
                fo = np.asarray(np.load(p, allow_pickle=True)["frame_observations"],
                                dtype=float)
            except Exception:
                fo = None
    n = int(fo.size) if fo is not None else 0
    return dict(
        obs=int(inst.get("observation_count", 0) or 0), vox=nv,
        input_pts=int(inst.get("input_point_count", 0) or 0),
        ext=float(dim.max()), ext_min=float(dim.min()), vol=vol,
        fill=float(nv * vs ** 3 / vol),
        mfvr=float(inst.get("multi_frame_voxel_ratio", 0.0) or 0.0),
        fo_mean=float(fo.mean()) if n else 0.0,
        fo_med=float(np.median(fo)) if n else 0.0,
        fo_f2=float((fo >= 2).mean()) if n else 0.0,
        fo_f5=float((fo >= 5).mean()) if n else 0.0,
    )


def build_X(rows, prior_map=None):
    L = np.log1p(np.maximum(np.asarray(
        [[r["obs"], r["fo_mean"], r["vox"], r["input_pts"], r["area_mean"]]
         for r in rows], dtype=float), 0.0))
    X = np.column_stack([
        L[:, 0], L[:, 1], L[:, 2], L[:, 3], L[:, 4],
        np.asarray([r["fill"] for r in rows], float),
        np.asarray([r["ext"] for r in rows], float),
        np.asarray([r["fo_f5"] for r in rows], float),
        np.asarray([r["mfvr"] for r in rows], float),
        np.asarray([r["det_max"] for r in rows], float),
    ])
    if prior_map is not None:
        X = np.column_stack([X, np.asarray(
            [prior_map.get(r["label"], prior_map["_g"]) for r in rows], float)])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def make_prior(train):
    acc = {}
    for r in train:
        acc.setdefault(r["label"], []).append(r["iou"])
    pri = {k: float(np.mean(v)) for k, v in acc.items()}
    pri["_g"] = float(np.mean([r["iou"] for r in train]))
    return pri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v10")
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--out", default="quality_scores.json")
    ap.add_argument("--train-mode", default="labeled",
                    choices=["labeled", "agnostic"],
                    help="用哪种 IoU 当回归目标。agnostic=纯几何质量，"
                         "不沾任何标签匹配口径，跨口径更稳")
    args = ap.parse_args()

    run_root = E.ROOT / args.run_root
    # ---- 1. 收集全部场景的特征 + IoU ----
    data = {}
    for sc in E.SCENES:
        r = E.eval_scene(sc, run_root, args.dist, mode=args.train_mode,
                         subset="all", verbose=False)
        if r is None:
            continue
        sd = run_root / sc.replace("_", "")
        fdir = sd / FUSION
        imap = json.loads((fdir / "instance_map.json").read_text())
        vs = float(imap.get("voxel_size_m", 0.05))
        df = det_feats(sd)
        rows = []
        for inst in imap["instances"]:
            gid = int(inst["global_id"])
            g = geom_feats(inst, fdir, vs)
            d = df.get(gid, {})
            g.update(label=str(inst.get("label", "?")), gid=gid,
                     det_mean=d.get("det_mean", 0.0),
                     det_max=d.get("det_max", 0.0),
                     sam_mean=d.get("sam_mean", 0.0),
                     area_mean=d.get("area_mean", 0.0),
                     n_det=d.get("n_det", 0),
                     iou=float(r["pred_best"].get(str(gid), [-1, 0.0])[1]))
            rows.append(g)
        data[sc] = rows
    scenes = sorted(data)
    print(f"收集 {sum(len(v) for v in data.values())} 个实例 / {len(scenes)} 场景")

    # ---- 2. 留一场景 CV：写回各场景 instance_map.json ----
    out = {}
    for sc in scenes:
        tr = [r for s2 in scenes if s2 != sc for r in data[s2]]
        te = data[sc]
        pri = make_prior(tr)
        Xtr = build_X(tr, pri)
        Xte = build_X(te, pri)
        ytr = np.array([r["iou"] for r in tr], float)
        m = Ridge(alpha=args.alpha).fit(Xtr, ytr)
        pred = m.predict(Xte)
        for r, p in zip(te, pred):
            out[f"{sc}|{r['gid']}"] = float(p)

        fdir = run_root / sc.replace("_", "") / FUSION
        imap_p = fdir / "instance_map.json"
        imap = json.loads(imap_p.read_text())
        for inst in imap["instances"]:
            k = f"{sc}|{int(inst['global_id'])}"
            if k in out:
                inst["quality_score"] = out[k]
        imap_p.write_text(json.dumps(imap, ensure_ascii=False, indent=2))
        print(f"{sc:<9} 训练 {len(tr):3d} 预测 {len(te):3d}  "
              f"pred [{pred.min():+.3f},{pred.max():+.3f}]  -> {imap_p.name}")

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nquality_score 已写回 {len(scenes)} 个场景的 instance_map.json")
    print("下一步：python3 patch_eval_score.py 让评测读 quality_score")


if __name__ == "__main__":
    main()
