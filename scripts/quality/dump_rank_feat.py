"""抽取"实例质量"打分所需的特征 + labeled IoU 标签。

约束（必须遵守，否则就是自欺）：
  - 所有特征只能来自系统自身的输出（融合实例、逐帧检测、几何），
    **不得**使用任何 GT mesh / GT 类别信息。
  - 标签（IoU）来自评测协议，只用于训练排序，不用于选特征。

输出 rank_feat.json：一行一个全局实例。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/efficient3d_robot")
import eval_mesh_protocol as E

FUSION = os.environ.get("FUSION_DIR", "fusion_attempt_01")


def det_feats(scene_dir):
    """global_id -> 逐帧检测特征聚合。"""
    trk_p = scene_dir / "association" / "tracking.json"
    if not trk_p.is_file():
        return {}
    trk = json.loads(trk_p.read_text())
    # frame -> {local_id: seg}
    seg_cache = {}
    acc = {}
    for fr in trk.get("frames", []):
        fi = int(fr.get("frame_index", -1))
        assocs = fr.get("associations", [])
        if not assocs:
            continue
        if fi not in seg_cache:
            p = scene_dir / "segmentation" / f"frame_{fi:06d}_instances.json"
            if not p.is_file():
                seg_cache[fi] = {}
                continue
            seg_cache[fi] = {int(s.get("local_instance_id", -1)): s
                             for s in json.loads(p.read_text())}
        for a in assocs:
            gid = a.get("global_id")
            if gid is None:
                continue
            lid = int(a.get("local_instance_id", -1))
            s = seg_cache[fi].get(lid)
            if s is None:
                continue
            acc.setdefault(int(gid), []).append((
                float(s.get("detection_score", 0.0) or 0.0),
                float(s.get("sam_predicted_iou", 0.0) or 0.0),
                float(s.get("area_pixels", 0) or 0.0)))
    out = {}
    for gid, rows in acc.items():
        a = np.array(rows, dtype=np.float64)
        det, sam, area = a[:, 0], a[:, 1], a[:, 2]
        out[gid] = dict(
            det_mean=float(det.mean()), det_max=float(det.max()),
            det_std=float(det.std()), sam_mean=float(sam.mean()),
            sam_min=float(sam.min()), area_mean=float(area.mean()),
            n_det=int(len(rows)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v10")
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--dump", default="rank_feat.json")
    args = ap.parse_args()

    run_root = E.ROOT / args.run_root
    rows = []
    for sc in E.SCENES:
        r = E.eval_scene(sc, run_root, args.dist, mode="labeled",
                         subset="all", verbose=False)
        if r is None:
            print(f"  [跳过] {sc}")
            continue
        sd = run_root / sc.replace("_", "")
        fdir = sd / FUSION
        imap = json.loads((fdir / "instance_map.json").read_text())
        vs = float(imap.get("voxel_size_m", 0.05))
        dfeat = det_feats(sd)
        n_gt = r["n_gt"]
        for inst in imap["instances"]:
            gid = int(inst["global_id"])
            pb = r["pred_best"].get(str(gid), [-1, 0.0])
            iou = float(pb[1])
            gt_id = int(pb[0])
            a = np.asarray(inst.get("bbox_min_world", [0, 0, 0]), dtype=float)
            b = np.asarray(inst.get("bbox_max_world", [0, 0, 0]), dtype=float)
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
                        z = np.load(p, allow_pickle=True)
                        fo = np.asarray(z["frame_observations"], dtype=float)
                    except Exception:
                        fo = None
            d = dfeat.get(gid, {})
            row = dict(
                scene=sc, gid=gid, label=str(inst.get("label", "?")),
                iou=iou, gt_id=gt_id, n_gt=n_gt,
                obs=int(inst.get("observation_count", 0) or 0),
                vox=nv,
                input_pts=int(inst.get("input_point_count", 0) or 0),
                ext=float(dim.max()), ext_min=float(dim.min()),
                vol=vol, fill=float(nv * vs ** 3 / vol),
                mfvr=float(inst.get("multi_frame_voxel_ratio", 0.0) or 0.0),
                fo_med=float(np.median(fo)) if fo is not None and fo.size else 0.0,
                fo_mean=float(fo.mean()) if fo is not None and fo.size else 0.0,
                fo_f2=float((fo >= 2).mean()) if fo is not None and fo.size else 0.0,
                fo_f5=float((fo >= 5).mean()) if fo is not None and fo.size else 0.0,
                det_mean=d.get("det_mean", 0.0), det_max=d.get("det_max", 0.0),
                det_std=d.get("det_std", 0.0), sam_mean=d.get("sam_mean", 0.0),
                sam_min=d.get("sam_min", 0.0),
                area_mean=d.get("area_mean", 0.0), n_det=d.get("n_det", 0),
            )
            rows.append(row)
        print(f"{sc:<9} 实例 {len(imap['instances']):3d}  GT {n_gt:3d}  有检测特征 "
              f"{sum(1 for i in imap['instances'] if int(i['global_id']) in dfeat)}")

    Path(args.dump).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n共 {len(rows)} 个实例 -> {args.dump}")
    y = np.array([r["iou"] for r in rows])
    print(f"IoU: mean {y.mean():.3f}  命中>=0.25 {int((y>=.25).sum())} "
          f"({100*(y>=.25).mean():.1f}%)  >=0.5 {int((y>=.5).sum())}")

    # 单特征与 IoU 的 Spearman 相关，快速看哪些有信号
    keys = ["obs", "vox", "input_pts", "ext", "ext_min", "vol", "fill",
            "mfvr", "fo_med", "fo_mean", "fo_f2", "fo_f5", "det_mean",
            "det_max", "det_std", "sam_mean", "sam_min", "area_mean", "n_det"]
    print("\n单特征 Spearman 相关（对 log1p(obs) 基线对比）：")
    from scipy.stats import spearmanr
    base = np.log1p(np.array([r["obs"] for r in rows], dtype=float))
    print(f"  {'log1p(obs) [基线]':<20} {spearmanr(base, y).correlation:+.3f}")
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        c = spearmanr(v, y).correlation
        print(f"  {k:<20} {c:+.3f}")


if __name__ == "__main__":
    main()
