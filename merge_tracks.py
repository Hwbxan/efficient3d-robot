"""离线二次合并：把"跨帧关联断裂"拆碎的同标签实例重新并起来。

动机
----
diag_assoc2 显示：可见完整度只有 59%，而 82% 的碎片是**同标签**的
（tv-screen 被拆成 61%/16%/15% 三个，door 拆成 66%/32% 两个）。
这是关联判据太严导致的断裂，不是分割器问题。

问题：这个断裂必须用"3D query + 对比学习"才能修吗？
这里先试非学习做法：同标签 + 点云空间接触 → 合并。若能修好，
就没必要为此上学习模型。

风险：一排相邻的同标签椅子会被错误合并。所以要看 AP 是涨是跌。
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/data/efficient3d_robot")
from eval_mesh_protocol import (  # noqa: E402
    REPLICA, STRUCTURAL, norm, read_semantic_ply,
    vertex_object_ids, load_prediction, average_precision, label_match,
)


def build_union_find(labels, pxyz, pids, contact, emb=None, sim=0.0,
                     mode="geom"):
    """合并判据。mode:
      geom = 同标签 + 接触          （纯手工）
      emb  = 嵌入相似 + 接触        （学习表示）
      both = 同标签 且 嵌入相似 + 接触
    返回 {旧id: 新代表id}。"""
    ids = sorted(labels.keys())
    subs = {g: pxyz[pids == g] for g in ids}
    parent = {g: g for g in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 预先算好两两接触距离
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ta, tb = cKDTree(subs[a]), cKDTree(subs[b])
            d1, _ = ta.query(subs[b], k=1)
            d2, _ = tb.query(subs[a], k=1)
            if min(d1.min(), d2.min()) > contact:
                continue
            same_lab = label_match(labels[a], labels[b])
            if mode == "geom":
                if same_lab:
                    union(a, b)
                continue
            if emb is None:
                continue
            cs = float(np.dot(emb[a], emb[b]))
            if mode == "emb":
                if cs >= sim:
                    union(a, b)
            else:  # both
                if same_lab and cs >= sim:
                    union(a, b)
    return {g: find(g) for g in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v4")
    ap.add_argument("--scenes", default="office_0,office_1")
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--contact", default="0.02")
    ap.add_argument("--sim", default="0.5,0.6,0.7,0.8,0.9")
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

        pred = load_prediction(run_root / scene.replace("_", ""))
        pxyz, pids, id_score, id_label = pred

        def run_eval(pids_used, score_map, tag):
            tree = cKDTree(pxyz)
            d, nn = tree.query(xyz, distance_upper_bound=a.dist)
            pav = np.where(np.isfinite(d),
                           pids_used[np.minimum(nn, len(pids_used) - 1)], -1)
            pb = {}
            for gid, c, g in gts:
                pv = pav[g]
                pv = pv[pv >= 0]
                if len(pv) == 0:
                    continue
                u, cnt = np.unique(pv, return_counts=True)
                b = int(u[int(np.argmax(cnt))])
                pm = (pav == b) & valid
                iou = int(cnt.max()) / max(int(np.logical_or(g, pm).sum()), 1)
                if b not in pb or iou > pb[b][1]:
                    pb[b] = [gid, iou]
            pl = sorted(score_map.keys())
            ious = [pb.get(p, [-1, 0.0])[1] for p in pl]
            gm = [pb.get(p, [-1, 0.0])[0] for p in pl]
            ss = [score_map[p] for p in pl]
            aps = [average_precision(ious, gm, ss, t, len(gts)) * 100
                   for t in (0.25, 0.5, 0.75)]
            cov = float(((pav >= 0) & valid).sum()) / max(int(valid.sum()), 1)
            # 可见完整度
            vis = []
            for gid, c, g in gts:
                pv = pav[g]
                pv = pv[pv >= 0]
                if len(pv) == 0:
                    continue
                cnt = collections.Counter(pv.tolist())
                vis.append(cnt.most_common(1)[0][1] / len(pv))
            print(f"{tag:<34}{len(pl):>6}{aps[0]:>7.1f}{aps[1]:>7.1f}"
                  f"{aps[2]:>7.1f}{cov*100:>8.1f}"
                  f"{(np.mean(vis)*100 if vis else 0):>8.1f}")

        print(f"\n===== {scene}  GT {len(gts)}  原始实例 {len(id_score)} =====")
        print(f"{'配置':<34}{'实例':>6}{'AP25':>7}{'AP50':>7}{'AP75':>7}"
              f"{'覆盖%':>8}{'可见完整':>8}")
        run_eval(pids, id_score, "基线")

        for contact in [float(x) for x in a.contact.split(",")]:
            mapping = build_union_find(id_label, pxyz, pids, contact)
            new_ids = np.array([mapping[g] for g in pids], dtype=np.int64)
            new_score = {}
            for old, new in mapping.items():
                # 合并后分数取成员最大观测数
                new_score[new] = max(new_score.get(new, 0.0), id_score[old])
            run_eval(new_ids, new_score, f"手工 同标签+接触<={contact}m")

        # ---- 学习表示判据 ----
        emb_path = run_root / scene.replace("_", "") / "track_embeddings.npz"
        if not emb_path.is_file():
            print("  （无 track_embeddings.npz，跳过嵌入判据；"
                  "先跑 extract_track_emb.py）")
            continue
        z = np.load(str(emb_path))
        meta = json.loads(str(z["meta"]))
        emb = {}
        for g in id_score.keys():
            key = f"s_{g}"
            if key in z:
                v = z[key].astype(np.float64)
                n = np.linalg.norm(v)
                emb[g] = v / max(n, 1e-12)
        print(f"  已载入 {len(emb)} 个实例嵌入")

        # 相似度分布：同标签对 vs 异标签对（判断嵌入有没有判别力）
        sim_same, sim_diff = [], []
        ids = sorted(emb.keys())
        for i, x in enumerate(ids):
            for y in ids[i + 1:]:
                cs = float(np.dot(emb[x], emb[y]))
                (sim_same if label_match(id_label[x], id_label[y])
                 else sim_diff).append(cs)
        if sim_same and sim_diff:
            print(f"  嵌入余弦：同标签对 n={len(sim_same)} 均值 "
                  f"{np.mean(sim_same):.3f}  异标签对 n={len(sim_diff)} 均值 "
                  f"{np.mean(sim_diff):.3f}  差 "
                  f"{np.mean(sim_same) - np.mean(sim_diff):+.3f}")

        for sim in [float(x) for x in a.sim.split(",")]:
            for mode in ("emb", "both"):
                for contact in [float(x) for x in a.contact.split(",")]:
                    mapping = build_union_find(id_label, pxyz, pids, contact,
                                               emb=emb, sim=sim, mode=mode)
                    new_ids = np.array([mapping[g] for g in pids],
                                       dtype=np.int64)
                    new_score = {}
                    for old, new in mapping.items():
                        new_score[new] = max(new_score.get(new, 0.0),
                                             id_score[old])
                    tag = (f"{'嵌入' if mode == 'emb' else '嵌入+标签'}"
                           f" sim>={sim} 接触<={contact}m")
                    run_eval(new_ids, new_score, tag)


if __name__ == "__main__":
    main()
