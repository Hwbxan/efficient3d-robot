"""按 OVI-MAP / OVO-SLAM 的 Replica 协议评测：3D 网格空间逐顶点实例 AP。

为什么需要这个
--------------
我们一直用"逐帧 2D 掩码 vs 逐帧 GT 掩码"评测，而 Replica 上在线开放词汇建图
的 SOTA（OVI-MAP, CVPR 2026；OVO-SLAM, RA-L 2025）用的是另一套口径：
把重建的实例地图用 kNN 投影到 GT mesh 上，逐顶点比较，算 AP@25/50/75。
两套数字不可比。这个脚本就是补上那一套口径。

做法
----
1. 读 Replica GT mesh（habitat/mesh_semantic.ply）：顶点 + 逐面 object_id
2. 读 info_semantic.json：object_id -> class_name / class_id
3. 读我们融合后的实例地图（fusion_attempt_01/individual/*.ply），拼成
   带实例标签的世界坐标点云
4. 每个 GT 顶点用 kNN 找最近的预测点，距离超过阈值的顶点视为"未覆盖"
5. 逐 GT 物体算顶点 IoU，按置信度排序扫阈值算 AP@25 / AP@50 / AP@75

两种口径
--------
--mode agnostic : 类别无关，纯几何。衡量"分割/建图质量"上界。
--mode labeled  : 预测标签必须与 GT 类别名匹配才算 TP。这才是论文的口径。
                  不匹配的预测直接算 FP。
--subset seen   : 只统计我们 prompt 覆盖到的 GT 类别（排除词汇表没覆盖的）
--subset all    : 统计所有非结构性 GT 类别（论文口径）
"""
import argparse
import json
import re
import os
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("需要 scipy：pip install scipy")
    sys.exit(1)

ROOT = Path("/data/efficient3d_robot")
REPLICA = ROOT / "datasets" / "raw" / "replica_v1"

# 结构性 / 背景类：任何开放词汇实例分割系统都不该被要求检出
STRUCTURAL = {
    "wall", "floor", "ceiling", "undefined", "non-plane", "other-leaf",
    "blinds", "window", "wall-plug", "switch", "vent", "pipe", "pillar",
    "panel", "anonymize_text", "anonymize_picture", "beam", "ceiling-fan",
    "unknown", "?", "rug", "curtain",
}

SCENES = ["office_0", "office_1", "office_2", "office_3", "office_4",
          "room_0", "room_1", "room_2"]

# 我们的标签 -> Replica GT 类名（归一化后比较）。值是允许等价的 GT 名集合。
# SYNONYM_V2 已修：连字符改空格 + 补 pillow/cushion 等
SYNONYM = {
    # 显示类：Replica 原始类名 tv-screen 归一化后是 tv screen，这里必须用空格写法
    "tv screen": {"tv screen", "tv", "monitor", "screen", "television"},
    "monitor": {"monitor", "tv screen", "screen", "tv"},
    "tablet": {"tablet", "tv screen", "monitor"},
    "television": {"television", "tv screen", "tv", "monitor"},
    "trash can": {"bin", "trashcan", "trash can", "wastebasket"},
    "bin": {"bin", "trash can", "trashcan"},
    "basket": {"basket", "bin"},
    "desk organizer": {"desk organizer", "organizer"},
    "tissue box": {"tissue paper", "tissue box", "box"},
    "chair": {"chair", "stool", "armchair", "seat"},
    "stool": {"stool", "chair"},
    "sofa": {"sofa", "couch", "armchair", "loveseat"},
    "desk": {"desk", "table", "nightstand", "counter"},
    "table": {"table", "desk", "nightstand", "counter"},
    "door": {"door"},
    # 开放词汇下 pillow / cushion 指的都是沙发或床上的软垫，等价
    "pillow": {"pillow", "cushion"},
    "cushion": {"cushion", "pillow"},
    # 挂画
    "picture frame": {"picture frame", "picture"},
    "picture": {"picture", "picture frame"},
    # 花瓶 / 花盆
    "vase": {"vase", "pot"},
    "pot": {"pot", "vase"},
    # Replica 用 indoor-plant 表示室内盆栽，我们输出 potted plant /
    # plant stand，是同一个物理对象类别（此前 5 个盆栽全部因命名归零）
    "potted plant": {"potted plant", "indoor plant", "plant", "houseplant"},
    "plant stand": {"plant stand", "indoor plant", "potted plant", "plant"},
    # Replica 只有 lamp 一类；我们可能输出 ceiling / recessed light
    "lamp": {"lamp", "ceiling light", "recessed light", "light fixture"},
    "ceiling light": {"ceiling light", "lamp", "light"},
    "recessed light": {"recessed light", "lamp", "light"},
}


def norm(s):
    """归一化标签：小写、连字符/下划线转空格、去多余空白。"""
    return re.sub(r"\s+", " ", str(s).lower().replace("-", " ")
                  .replace("_", " ")).strip()


def label_match(pred_label, gt_class, strict=False):
    """预测标签是否可接受为该 GT 类别。

    strict=True 只认完全同名（保守下界）；默认走同义词表。
    """
    p, g = norm(pred_label), norm(gt_class)
    if p == g:
        return True
    if strict:
        return False
    if g in SYNONYM.get(p, set()):
        return True
    # 退化：子串包含；避免误伤 "desk" / "desk organizer"
    if p and g and (p in g or g in p) and abs(len(p) - len(g)) <= 3:
        return True
    return False


def read_semantic_ply(path):
    """读 Replica 的 mesh_semantic.ply：顶点 xyz + 逐面 object_id。"""
    with open(path, "rb") as f:
        raw = f.read()
    idx = raw.find(b"end_header")
    header = raw[:idx].decode("ascii")
    body = raw[idx + len(b"end_header"):].lstrip(b"\n")

    n_vert = n_face = 0
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n_vert = int(line.split()[-1])
        elif line.startswith("element face"):
            n_face = int(line.split()[-1])

    v_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
    verts = np.frombuffer(body, dtype=v_dtype, count=n_vert)
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    off = v_dtype.itemsize * n_vert

    # 面：uint8 count + count×uint32 indices + uint16 object_id
    # Replica 是四边形网格（count=4，stride=19），不能假定是三角形
    n_poly = int(body[off])
    stride = 1 + 4 * n_poly + 2
    if len(body) - off < stride * n_face:
        # count 解析异常，退回三角形尝试
        for cand in (3, 4):
            s = 1 + 4 * cand + 2
            if len(body) - off >= s * n_face:
                n_poly, stride = cand, s
                break
    faces = np.frombuffer(body[off:off + stride * n_face], dtype=np.uint8)
    faces = faces.reshape(n_face, stride)
    tri = np.frombuffer(faces[:, 1:1 + 4 * n_poly].tobytes(), dtype="<u4")
    tri = tri.reshape(n_face, n_poly)
    obj_id = np.frombuffer(faces[:, 1 + 4 * n_poly:3 + 4 * n_poly].tobytes(),
                           dtype="<u2").reshape(n_face)
    return xyz, tri, obj_id


def read_xyz_ply(path):
    """读 Open3D 导出的实例点云（double xyz [+ uchar rgb]）。"""
    with open(path, "rb") as f:
        raw = f.read()
    idx = raw.find(b"end_header")
    header = raw[:idx].decode("ascii", "ignore")
    body = raw[idx + len(b"end_header"):].lstrip(b"\n")
    n_vert = 0
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n_vert = int(line.split()[-1])
    if n_vert == 0:
        return np.zeros((0, 3))
    has_color = "uchar red" in header or "uchar r " in header
    if has_color:
        dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                          ("r", "u1"), ("g", "u1"), ("b", "u1")])
    else:
        dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    arr = np.frombuffer(body, dtype=dtype, count=n_vert)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


def load_prediction(run_dir, min_obs=0, max_extent=0.0, min_voxel=0):
    """融合实例地图 -> (点云, 实例 id, {id: score}, {id: label})

    min_obs / max_extent / min_voxel 是"弱证据实例"过滤：观测帧数太少、
    包围盒明显超出单体家具合理尺寸、体素太少的实例直接丢弃。
    这是所有 3D 实例建图系统都有的通用后处理，不是针对 GT 调参。
    """
    fusion = run_dir / os.environ.get(
        "FUSION_DIR", "fusion_attempt_01")
    imap = json.loads((fusion / "instance_map.json").read_text())
    indiv = fusion / "individual"
    pts, ids, id_score, id_label = [], [], {}, {}
    dropped = []
    for inst in imap["instances"]:
        gid = int(inst["global_id"])
        n_obs = int(inst.get("observation_count", 0)) or \
            len(inst.get("source_frames", [])) or 1
        if n_obs < min_obs:
            dropped.append((gid, inst.get("label", "?"), f"obs={n_obs}"))
            continue
        nv = int(inst.get("voxel_count", 0))
        if nv < min_voxel:
            dropped.append((gid, inst.get("label", "?"), f"voxel={nv}"))
            continue
        if max_extent > 0:
            a = inst.get("bbox_min_world")
            b = inst.get("bbox_max_world")
            if a and b:
                ext = float(max(np.asarray(b) - np.asarray(a)))
                if ext > max_extent:
                    dropped.append((gid, inst.get("label", "?"), f"ext={ext:.2f}"))
                    continue
        rel = inst.get("point_cloud_path")
        if rel:
            p = Path(rel)
            if not p.is_absolute():
                p = fusion / rel          # rel 相对 fusion_attempt_01/
        else:
            p = indiv / f"G{gid:03d}.ply"
        if not p.is_file():
            p = indiv / f"instance_{gid:03d}.ply"
        if not p.is_file():
            continue
        xyz = read_xyz_ply(p)
        if len(xyz) == 0:
            continue
        pts.append(xyz)
        ids.append(np.full(len(xyz), gid, dtype=np.int64))
        n_obs = int(inst.get("observation_count", 0)) or \
            len(inst.get("source_frames", [])) or 1
        # 优先用实例质量打分器（apply_quality_score.py 写回的 Ridge 校准分）；
        # 没有则退回原来的 log(1+观测帧数)。
        qs = inst.get("quality_score")
        id_score[gid] = float(qs) if qs is not None else \
            float(np.log1p(max(n_obs, 1)))
        id_label[gid] = inst.get("label", "?")
    if not pts:
        return None
    return np.vstack(pts), np.concatenate(ids), id_score, id_label


def vertex_object_ids(n_vert, tri, obj_id):
    """逐面 object_id -> 逐顶点 object_id。"""
    vobj = np.full(n_vert, -1, dtype=np.int64)
    vobj[tri[:, 0]] = obj_id
    vobj[tri[:, 1]] = obj_id
    vobj[tri[:, 2]] = obj_id
    return vobj


def average_precision(ious, gt_ids, scores, threshold, n_gt, n_point=101):
    """标准 AP：按分数降序、每个 GT 只被最高分预测占用一次、PR 曲线插值。

    ious[i]    预测 i 与其最佳 GT 的 IoU
    gt_ids[i]  该 GT 的 id（-1 表示无匹配）
    scores[i]  预测 i 的置信度
    """
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    ious = np.asarray(ious, dtype=np.float64)[order]
    gt_ids = np.asarray(gt_ids, dtype=np.int64)[order]
    used, tp = set(), np.zeros(len(order), dtype=np.float64)
    for k in range(len(order)):
        if ious[k] >= threshold and gt_ids[k] >= 0 and gt_ids[k] not in used:
            tp[k] = 1.0
            used.add(int(gt_ids[k]))
    tp_cum = np.cumsum(tp)
    prec = tp_cum / (np.arange(len(order)) + 1.0)
    rec = tp_cum / max(n_gt, 1)
    # 全点插值（COCO 口径）：precision 取右侧最大值后按 recall 网格求均值
    prec_flipped = np.maximum.accumulate(prec[::-1])[::-1]
    grid = np.linspace(0, 1, n_point)
    return float(np.mean([prec_flipped[rec >= r].max() if (rec >= r).any()
                          else 0.0 for r in grid]))


def eval_scene(scene, run_root, dist_thresh, mode="agnostic", subset="all",
               min_vert=50, verbose=True, min_obs=0, max_extent=0.0,
               min_voxel=0, strict=False):
    mesh_p = REPLICA / scene / "habitat" / "mesh_semantic.ply"
    info_p = REPLICA / scene / "habitat" / "info_semantic.json"
    if not mesh_p.is_file() or not info_p.is_file():
        print(f"  [跳过] {scene}: 缺 GT mesh")
        return None
    info = json.loads(info_p.read_text())
    obj_meta = {int(o["id"]): o for o in info["objects"]}

    xyz, tri, face_obj = read_semantic_ply(mesh_p)
    vobj = vertex_object_ids(len(xyz), tri, face_obj)

    # 关键修正：只在"非结构性顶点"上做投影与 IoU。
    # kNN 会把墙/地板/天花板的顶点也分配给邻近的物体实例，这些顶点不属于
    # 任何待评测 GT 物体，若计入 union 会让 IoU 被系统性压低（分母污染）。
    struct_norm = {norm(s) for s in STRUCTURAL}
    is_struct = np.zeros(len(xyz), dtype=bool)
    for gid in np.unique(vobj):
        gid = int(gid)
        if gid < 0:
            continue
        cname = (obj_meta.get(gid) or {}).get("class_name", "?")
        if norm(cname) in struct_norm:
            is_struct[vobj == gid] = True
    # vobj == -1 的顶点不属于任何面（Replica mesh 有 ~25% 孤立顶点），
    # 它们不归属任何 GT 物体，必须排除，否则会系统性撑大 union 压低 IoU。
    valid = (~is_struct) & (vobj >= 0)

    run_dir = run_root / scene.replace("_", "")
    if not (run_dir / "fusion_attempt_01" / "instance_map.json").is_file():
        print(f"  [跳过] {scene}: 无融合实例地图")
        return None
    pred = load_prediction(run_dir, min_obs=min_obs, max_extent=max_extent,
                           min_voxel=min_voxel)
    if pred is None:
        print(f"  [跳过] {scene}: 预测点云为空")
        return None
    pxyz, pids, id_score, id_label = pred
    if len(pxyz) < 50:
        print(f"  [跳过] {scene}: 过滤后预测点太少（{len(pxyz)}）")
        return None
    n_raw = len(json.loads((run_dir / "fusion_attempt_01" /
                            "instance_map.json").read_text())["instances"])

    tree = cKDTree(pxyz)
    dist, nn = tree.query(xyz, distance_upper_bound=dist_thresh)
    hit = np.isfinite(dist)
    pred_at_vertex = np.where(hit, pids[np.minimum(nn, len(pids) - 1)], -1)

    # ---- 逐 GT 物体：找最佳预测实例 ----
    struct_norm = {norm(s) for s in STRUCTURAL}
    gt_ids_all = sorted({int(v) for v in np.unique(vobj) if int(v) >= 0})
    gt_records = []
    for gid in gt_ids_all:
        meta = obj_meta.get(gid)
        cname = (meta or {}).get("class_name", "?")
        if norm(cname) in struct_norm:
            continue
        gmask = (vobj == gid) & valid
        n_gt_vert = int(gmask.sum())
        if n_gt_vert < min_vert:
            continue
        gt_records.append(dict(gid=gid, cls=cname, n=n_gt_vert))

    # 预测实例 -> (最佳GT, IoU)
    pred_best = {}   # pid -> [gt_gid, iou]
    for rec in gt_records:
        gmask = (vobj == rec["gid"]) & valid
        pv = pred_at_vertex[gmask]
        pv = pv[pv >= 0]
        if len(pv) == 0:
            rec["best_iou"] = 0.0
            rec["pred"] = -1
            continue
        uniq, cnt = np.unique(pv, return_counts=True)
        # labeled 模式下跳过标签不匹配的预测实例
        if mode == "labeled":
            keep = [u for u in uniq if label_match(id_label.get(int(u), "?"), rec["cls"], strict)]
            if not keep:
                rec["best_iou"] = 0.0
                rec["pred"] = -1
                continue
            kset = set(int(k) for k in keep)
            sel = np.array([c for u, c in zip(uniq, cnt) if int(u) in kset])
            su = np.array([u for u in uniq if int(u) in kset])
        else:
            sel, su = cnt, uniq
        best = int(su[int(np.argmax(sel))])
        inter = int(sel.max())
        pmask = (pred_at_vertex == best) & valid
        union = int(np.logical_or(gmask, pmask).sum())
        iou = inter / max(union, 1)
        rec["best_iou"] = iou
        rec["pred"] = best
        if best not in pred_best or iou > pred_best[best][1]:
            pred_best[best] = [rec["gid"], iou]

    if subset == "seen":
        gt_records = [r for r in gt_records
                      if any(label_match(l, r["cls"], strict) for l in id_label.values())]
    if not gt_records:
        return None

    # ---- AP：所有预测实例参与（未匹配的 iou=0 当 FP）----
    pid_list = sorted(id_score.keys())
    ious = [pred_best.get(p, [-1, 0.0])[1] for p in pid_list]
    gtmatch = [pred_best.get(p, [-1, 0.0])[0] for p in pid_list]
    scores = [id_score.get(p, 0.0) for p in pid_list]
    n_gt = len(gt_records)
    aps = {t: average_precision(ious, gtmatch, scores, t, n_gt)
           for t in (0.25, 0.50, 0.75)}

    iou_arr = np.array([r["best_iou"] for r in gt_records])
    n_hit = {t: int((iou_arr >= t).sum()) for t in (0.25, 0.50, 0.75)}
    cov = float((hit & valid).sum()) / max(int(valid.sum()), 1)
    if verbose:
        print(f"{scene:<9} GT {n_gt:3d}  预测 {len(pid_list):3d}/{n_raw:<3d}"
              f" 物体顶点覆盖 {cov*100:5.1f}%  "
              f"AP25 {aps[0.25]*100:5.1f}  AP50 {aps[0.50]*100:5.1f}  "
              f"AP75 {aps[0.75]*100:5.1f}   "
              f"R25 {n_hit[0.25]:3d} R50 {n_hit[0.50]:3d} R75 {n_hit[0.75]:3d}")
    return dict(scene=scene, n_gt=n_gt, n_pred=len(pid_list),
                coverage=cov,
                # 供排序敏感性实验复用（ap_rank.py）
                pred_best={str(k): v for k, v in pred_best.items()},
                pid_list=[int(p) for p in pid_list],
                id_score={str(k): v for k, v in id_score.items()},
                ap25=aps[0.25], ap50=aps[0.50], ap75=aps[0.75],
                hits={str(k): v for k, v in n_hit.items()},
                classes=sorted({r["cls"] for r in gt_records}),
                missed=sorted({r["cls"] for r in gt_records if r["best_iou"] < 0.25}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v4")
    ap.add_argument("--dist", type=float, default=0.05,
                    help="kNN 距离阈值（米），GT 顶点到预测点的最大允许距离")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--mode", default="agnostic", choices=["agnostic", "labeled"])
    ap.add_argument("--subset", default="all", choices=["all", "seen"])
    ap.add_argument("--min-vert", type=int, default=50)
    ap.add_argument("--min-obs", type=int, default=0,
                    help="丢弃观测帧数少于该值的实例")
    ap.add_argument("--max-extent", type=float, default=0.0,
                    help="丢弃包围盒最大边超过该值（米）的实例；0=关闭")
    ap.add_argument("--min-voxel", type=int, default=0,
                    help="丢弃体素数少于该值的实例")
    ap.add_argument("--strict", action="store_true",
                    help="只认完全同名的标签（保守下界）")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    run_root = ROOT / args.run_root
    print("=" * 100)
    print(f"Replica 网格协议评测（对齐 OVI-MAP / OVO-SLAM 口径）")
    print(f"run={args.run_root}  kNN={args.dist} m  mode={args.mode}  "
          f"subset={args.subset}  AP=COCO 101点插值")
    print("=" * 100)
    all_res = []
    for sc in args.scenes.split(","):
        r = eval_scene(sc.strip(), run_root, args.dist, args.mode, args.subset,
                       args.min_vert, min_obs=args.min_obs,
                       max_extent=args.max_extent, min_voxel=args.min_voxel,
                       strict=args.strict)
        if r:
            all_res.append(r)
    if not all_res:
        print("没有任何场景可评测")
        return
    print("-" * 100)
    for k, label in (("ap25", "AP25"), ("ap50", "AP50"), ("ap75", "AP75")):
        v = np.mean([r[k] for r in all_res])
        print(f"宏平均 {label:<8} {v*100:6.1f}")
    cov = np.mean([r["coverage"] for r in all_res])
    print(f"宏平均 顶点覆盖 {cov*100:6.1f}%")
    miss = {}
    for r in all_res:
        for c in r["missed"]:
            miss[c] = miss.get(c, 0) + 1
    if miss:
        print("\nIoU<0.25 的 GT 类别（漏检，按出现场景数排序）：")
        for c, n in sorted(miss.items(), key=lambda x: -x[1])[:25]:
            print(f"  {c:<20} {n} 个场景")

    if args.dump:
        out = ROOT / args.dump
    else:
        tag = "strict" if args.strict else "syn"
        out = ROOT / (f"mesh_eval_{args.run_root.replace('/', '_')}"
                      f"_{args.mode}_{args.subset}_{tag}.json")
    out.write_text(json.dumps(all_res, ensure_ascii=False, indent=2, default=float))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
