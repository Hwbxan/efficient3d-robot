"""精确分离"没看到" vs "看到了但拆碎"，并判断碎片是同标签还是异标签。

三项指标
--------
1. 可见完整度 = 最大实例占**已覆盖**顶点的比例（排除"物体没被看到"的影响）
   对比 原始完整度 = 最大实例占 GT 全部顶点的比例
2. 碎片标签一致性：瓜分同一 GT 的多个实例是否标签相同
   - 同标签 = 跨帧关联断裂（该合并没合并 → 关联问题）
   - 异标签 = 检测器给了不同标签（→ 前端问题）
3. 覆盖不全率 = GT 顶点完全没被任何实例覆盖的比例
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
    vertex_object_ids, load_prediction, label_match,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v4")
    ap.add_argument("--scenes", default="office_0,office_1")
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--show", action="store_true", help="打印每个 GT 的瓜分详情")
    a = ap.parse_args()
    run_root = Path("/data/efficient3d_robot") / a.run_root

    tot = collections.defaultdict(list)
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
            continue
        pxyz, pids, id_score, id_label = pred
        tree = cKDTree(pxyz)
        d, nn = tree.query(xyz, distance_upper_bound=a.dist)
        pav = np.where(np.isfinite(d), pids[np.minimum(nn, len(pids) - 1)], -1)
        gv, pv = vobj[valid], pav[valid]

        print(f"\n===== {scene} =====")
        if a.show:
            print(f"{'GT类别':<15}{'顶点':>6}{'覆盖%':>7}{'可见完整':>8}"
                  f"{'碎片':>5}  前3大实例(标签:占比)")
        for gid in sorted({int(v) for v in np.unique(gv) if int(v) >= 0}):
            g = gv == gid
            n = int(g.sum())
            if n < 50:
                continue
            cname = (om.get(gid) or {}).get("class_name", "?")
            sel = pv[g]
            covered = sel >= 0
            cov_frac = covered.mean()
            selv = sel[covered]
            if len(selv) == 0:
                tot["vis_comp"].append(0.0)
                tot["raw_comp"].append(0.0)
                tot["uncovered"].append(1.0)
                tot["nfrag"].append(0)
                if a.show:
                    print(f"{cname:<15}{n:>6}{0.0:>7.1f}{0.0:>8.1f}{0:>5}")
                continue
            cnt = collections.Counter(selv.tolist())
            mc = cnt.most_common(3)
            top, topn = mc[0]
            vis_comp = topn / len(selv)
            raw_comp = topn / n
            nfrag = sum(1 for c in cnt.values() if c / len(selv) > 0.10)
            # 碎片标签：占比 >10% 的实例标签
            frags = [(pid, c) for pid, c in cnt.items() if c / len(selv) > 0.10]
            frag_labels = {id_label.get(int(p), "?") for p, _ in frags}
            same_label = len(frag_labels) <= 1
            tot["vis_comp"].append(vis_comp)
            tot["raw_comp"].append(raw_comp)
            tot["uncovered"].append(1 - cov_frac)
            tot["nfrag"].append(nfrag)
            tot["same_label"].append(1.0 if same_label else 0.0)
            tot["top_correct"].append(
                1.0 if label_match(id_label.get(int(top), "?"), cname) else 0.0)
            if a.show:
                desc = " ".join(
                    f"{id_label.get(int(p), '?')}:{c*100//len(selv)}%"
                    for p, c in mc)
                print(f"{cname:<15}{n:>6}{cov_frac*100:>7.1f}"
                      f"{vis_comp*100:>8.1f}{nfrag:>5}  {desc}")

    print("\n" + "=" * 70)
    print("宏平均（GT 物体为单位的平均）")
    for k, name in (("vis_comp", "可见完整度（已看到的顶点里最大实例占多少）"),
                    ("raw_comp", "原始完整度（占 GT 全部顶点）"),
                    ("uncovered", "完全没被覆盖的顶点比例"),
                    ("nfrag", "碎片数（占比>10%的实例个数）"),
                    ("same_label", "碎片标签一致的比例"),
                    ("top_correct", "最大实例标签正确率")):
        if tot[k]:
            v = np.mean(tot[k])
            print(f"  {name:<38} {v*100 if k != 'nfrag' else v:>6.1f}"
                  f"{'%' if k != 'nfrag' else ' 个'}")


if __name__ == "__main__":
    main()
