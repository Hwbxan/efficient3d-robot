"""归因：GT 物体的 IoU 缺口里，多少是"几何够不着"，多少是"标签对不上"。

对每个 GT 物体算两个 IoU：
  iou_any  = 最佳预测实例（不看类别）      -> 几何 / 分割能力
  iou_lbl  = 最佳预测实例（类别必须匹配）  -> 加上开放词汇命名的损失
差值就是纯标签损失。

用法：python3 diag_gap.py --run-root outputs/rt8_v10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/efficient3d_robot")
import eval_mesh_protocol as E


def scene_rows(scene, run_root, dist=0.05, min_vert=50):
    mesh_p = E.REPLICA / scene / "habitat" / "mesh_semantic.ply"
    info_p = E.REPLICA / scene / "habitat" / "info_semantic.json"
    info = json.loads(info_p.read_text())
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

    run_dir = run_root / scene.replace("_", "")
    pred = E.load_prediction(run_dir)
    if pred is None:
        return []
    pxyz, pids, id_score, id_label = pred
    from scipy.spatial import cKDTree
    tree = cKDTree(pxyz)
    d, nn = tree.query(xyz, distance_upper_bound=dist)
    hit = np.isfinite(d)
    pv_all = np.where(hit, pids[np.minimum(nn, len(pids) - 1)], -1)

    rows = []
    for gid in sorted({int(v) for v in np.unique(vobj) if int(v) >= 0}):
        meta = obj_meta.get(gid)
        cname = (meta or {}).get("class_name", "?")
        if E.norm(cname) in struct_norm:
            continue
        gmask = (vobj == gid) & valid
        if int(gmask.sum()) < min_vert:
            continue
        pv = pv_all[gmask]
        pv = pv[pv >= 0]
        rec = dict(scene=scene, gid=gid, cls=cname, n=int(gmask.sum()),
                   cov=float(pv.size) / int(gmask.sum()),
                   iou_any=0.0, pid_any=-1, lbl_any="",
                   iou_lbl=0.0, pid_lbl=-1)
        if pv.size == 0:
            rows.append(rec)
            continue
        uniq, cnt = np.unique(pv, return_counts=True)
        best_any = int(uniq[int(np.argmax(cnt))])
        pmask = (pv_all == best_any) & valid
        union = int(np.logical_or(gmask, pmask).sum())
        rec["iou_any"] = int(cnt.max()) / max(union, 1)
        rec["pid_any"] = best_any
        rec["lbl_any"] = id_label.get(best_any, "?")

        keep = [u for u in uniq
                if E.label_match(id_label.get(int(u), "?"), cname)]
        if keep:
            kset = {int(k) for k in keep}
            sel = np.array([c for u, c in zip(uniq, cnt) if int(u) in kset])
            su = np.array([u for u in uniq if int(u) in kset])
            b = int(su[int(np.argmax(sel))])
            pmask = (pv_all == b) & valid
            rec["iou_lbl"] = int(sel.max()) / max(int(np.logical_or(gmask, pmask).sum()), 1)
            rec["pid_lbl"] = b
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v10")
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--min-vert", type=int, default=50)
    ap.add_argument("--dump", default="gap_rows.json")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    run_root = E.ROOT / args.run_root
    all_rows = []
    for sc in E.SCENES:
        r = scene_rows(sc, run_root, args.dist, args.min_vert)
        if not r:
            print(f"  [跳过] {sc}")
            continue
        a = np.array([x["iou_any"] for x in r])
        l = np.array([x["iou_lbl"] for x in r])
        print(f"{sc:<9} GT {len(r):3d}  R25: any {int((a>=.25).sum()):3d} "
              f"lbl {int((l>=.25).sum()):3d}  R50: any {int((a>=.5).sum()):3d} "
              f"lbl {int((l>=.5).sum()):3d}   纯标签损失 "
              f"{int(((a>=.25)&(l<.25)).sum()):3d}")
        all_rows += r

    a = np.array([x["iou_any"] for x in all_rows])
    l = np.array([x["iou_lbl"] for x in all_rows])
    n = len(all_rows)
    print(f"\n=== 合计 {n} 个 GT 物体 ===")
    for t in (0.25, 0.50, 0.75):
        print(f"  R{int(t*100)}: 几何上界 {int((a>=t).sum()):3d} ({100*(a>=t).mean():.1f}%)"
              f"   命名后 {int((l>=t).sum()):3d} ({100*(l>=t).mean():.1f}%)"
              f"   差 {int(((a>=t)&(l<t)).sum()):3d}")
    print(f"  完全没被任何预测覆盖（cov=0）的物体：{int(sum(1 for x in all_rows if x['cov']==0))}")
    print(f"  cov>0 但 iou_any<0.25：{int(((a<.25)&(np.array([x['cov'] for x in all_rows])>0)).sum())}")

    # 按类别
    print(f"\n=== 按 GT 类别（物体数 >= 2）===")
    by = {}
    for x in all_rows:
        by.setdefault(x["cls"], []).append(x)
    rows = sorted(by.items(), key=lambda kv: -len(kv[1]))
    print(f"{'类别':<18}{'物体':>4}{'平均顶点':>9}{'cov':>7}"
          f"{'IoU_any':>9}{'IoU_lbl':>9}{'R25any':>7}{'R25lbl':>7}")
    for c, xs in rows:
        if len(xs) < 2:
            continue
        aa = np.mean([x["iou_any"] for x in xs])
        ll = np.mean([x["iou_lbl"] for x in xs])
        print(f"{c:<18}{len(xs):>4}{np.mean([x['n'] for x in xs]):>9.0f}"
              f"{np.mean([x['cov'] for x in xs]):>7.2f}"
              f"{aa:>9.3f}{ll:>9.3f}"
              f"{sum(1 for x in xs if x['iou_any']>=.25):>7}"
              f"{sum(1 for x in xs if x['iou_lbl']>=.25):>7}")

    # 最大缺口：几何能到 0.25 但命名后归零
    print(f"\n=== 纯标签损失（几何已有实例，命名对不上）前 {args.top} ===")
    lost = [x for x in all_rows if x["iou_any"] >= 0.25 and x["iou_lbl"] < 0.25]
    for x in sorted(lost, key=lambda x: -x["n"])[:args.top]:
        print(f"  {x['scene']:<9} {x['cls']:<16} 顶点{x['n']:>6}  "
              f"iou_any {x['iou_any']:.2f}  预测标签 “{x['lbl_any']}”")

    # 几何缺口：有覆盖但 IoU 低（过分割 / 欠分割）
    print(f"\n=== 几何缺口（cov>0.3 但 iou_any<0.25）前 {args.top} ===")
    geo = [x for x in all_rows if x["cov"] > 0.3 and x["iou_any"] < 0.25]
    for x in sorted(geo, key=lambda x: -x["n"])[:args.top]:
        print(f"  {x['scene']:<9} {x['cls']:<16} 顶点{x['n']:>6}  "
              f"cov {x['cov']:.2f}  iou_any {x['iou_any']:.2f}  预测标签 “{x['lbl_any']}”")

    if args.dump:
        Path(args.dump).write_text(json.dumps(all_rows, ensure_ascii=False, indent=1))
        print(f"\n写入 {args.dump}")


if __name__ == "__main__":
    main()
