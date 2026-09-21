"""量化跨帧关联质量：我们的手工关联到底错在哪、错多少。

三个指标
--------
1. 实例纯度 purity：一个预测实例覆盖的 GT 顶点里，最大单个 GT 物体占多少。
   低 = 一个实例把多个 GT 物体揉在一起（过合并 / 关联错误）。
2. GT 完整度 completeness：一个 GT 物体被多少个实例瓜分，最大那个占多少。
   低 = 同一个物体被拆成好几个实例（欠合并 / 关联断裂）。
3. 标签准确率：实例标签 vs 它主要覆盖的 GT 类别是否一致。

这三项能把"关联烂"和"分割粒度粗"区分开：
  - 纯度低 + 完整度高  → 关联把多个物体揉成一团（真·关联问题）
  - 纯度高 + 完整度低  → 物体被切碎（关联断裂）
  - 纯度低 + 完整度低  → 2D mask 粒度就不对（分割器问题）
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
    REPLICA, STRUCTURAL, norm, read_semantic_ply, read_xyz_ply,
    vertex_object_ids, load_prediction, label_match,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v4")
    ap.add_argument("--scenes", default="office_0")
    ap.add_argument("--dist", type=float, default=0.05)
    a = ap.parse_args()
    run_root = Path("/data/efficient3d_robot") / a.run_root

    print(f"{'场景':<10}{'实例':>5}{'GT':>5}{'纯度':>7}{'完整度':>8}"
          f"{'标签准':>7}{'揉合物体':>9}{'瓜分实例':>9}")
    agg = collections.defaultdict(list)
    detail = []
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

        pred = load_prediction(run_root / scene.replace("_", ""))
        if pred is None:
            print(f"{scene:<10} 无预测")
            continue
        pxyz, pids, id_score, id_label = pred
        tree = cKDTree(pxyz)
        d, nn = tree.query(xyz, distance_upper_bound=a.dist)
        pav = np.where(np.isfinite(d), pids[np.minimum(nn, len(pids) - 1)], -1)

        # 只保留有效 GT 顶点上的投影
        gv = vobj[valid]
        pv = pav[valid]

        # --- 实例 -> GT 分布 ---
        purities, n_merged = [], []
        for gid in sorted(set(id_score.keys())):
            sel = pv == gid
            if sel.sum() < 20:
                continue
            cnt = collections.Counter(gv[sel].tolist())
            tot = sum(cnt.values())
            top, topn = cnt.most_common(1)[0]
            purities.append(topn / tot)
            # 揉合：占比 >10% 的 GT 物体个数
            n_merged.append(sum(1 for c in cnt.values() if c / tot > 0.10))

        # --- GT -> 实例分布 ---
        comps, n_split = [], []
        label_ok, label_tot = 0, 0
        for gid in sorted({int(v) for v in np.unique(gv) if int(v) >= 0}):
            g = gv == gid
            if g.sum() < 50:
                continue
            sel = pv[g]
            sel = sel[sel >= 0]
            cname = (om.get(gid) or {}).get("class_name", "?")
            if len(sel) == 0:
                comps.append(0.0)
                n_split.append(0)
                continue
            cnt = collections.Counter(sel.tolist())
            top, topn = cnt.most_common(1)[0]
            comps.append(topn / g.sum())
            n_split.append(sum(1 for c in cnt.values() if c / g.sum() > 0.10))
            label_tot += 1
            if label_match(id_label.get(int(top), "?"), cname):
                label_ok += 1

        pur = float(np.mean(purities)) if purities else 0.0
        com = float(np.mean(comps)) if comps else 0.0
        lab = label_ok / max(label_tot, 1)
        nm = float(np.mean(n_merged)) if n_merged else 0.0
        ns = float(np.mean(n_split)) if n_split else 0.0
        print(f"{scene:<10}{len(id_score):>5}{label_tot:>5}{pur*100:>6.1f}%"
              f"{com*100:>7.1f}%{lab*100:>6.1f}%{nm:>9.1f}{ns:>9.1f}")
        agg["pur"].append(pur)
        agg["com"].append(com)
        agg["lab"].append(lab)
        agg["nm"].append(nm)
        agg["ns"].append(ns)
        detail.append((scene, pur, com, lab))

    if len(detail) > 1:
        print("-" * 62)
        print(f"{'宏平均':<10}{'':>10}{np.mean(agg['pur'])*100:>6.1f}%"
              f"{np.mean(agg['com'])*100:>7.1f}%{np.mean(agg['lab'])*100:>6.1f}%"
              f"{np.mean(agg['nm']):>9.1f}{np.mean(agg['ns']):>9.1f}")
        print("\n判读：揉合物体 >1.5 说明过合并严重；瓜分实例 >1.5 说明欠合并严重；")
        print("      两项都高 = 2D mask 粒度本身就不对（分割器问题，不是关联问题）")


if __name__ == "__main__":
    main()
