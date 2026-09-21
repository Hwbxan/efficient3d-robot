"""排序（置信度）敏感性实验：同一组预测，换个排序能涨多少 AP？

动机：AP 是 PR 曲线下面积，预测集不变时只由排序决定。当前置信度是
log(1+观测次数)——一个很弱的代理。如果 oracle 排序（按真实 IoU 降序）能把
AP50 从 28.9 拉到 60+，那么瓶颈就不是"检不出物体"，而是"不会打分"，
这时训练/设计一个好的置信度比换检测器便宜得多，也更该优先做。

用法：--run-root outputs/rt8_v5 --scenes office_0,office_1
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/data/efficient3d_robot")
import eval_mesh_protocol as E


def ap_with(scores, ious, gtmatch, n_gt, threshold):
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    ious = np.asarray(ious, dtype=np.float64)[order]
    gt_ids = np.asarray(gtmatch, dtype=np.int64)[order]
    used, tp = set(), np.zeros(len(order), dtype=np.float64)
    for k in range(len(order)):
        if ious[k] >= threshold and gt_ids[k] >= 0 and gt_ids[k] not in used:
            tp[k] = 1.0
            used.add(int(gt_ids[k]))
    tp_cum = np.cumsum(tp)
    prec = tp_cum / (np.arange(len(order)) + 1.0)
    rec = tp_cum / max(n_gt, 1)
    prec_flipped = np.maximum.accumulate(prec[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    return float(np.mean([prec_flipped[rec >= r].max() if (rec >= r).any()
                          else 0.0 for r in grid]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v5")
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--scenes", default=",".join(E.SCENES))
    ap.add_argument("--subset", default="all", choices=["all", "seen"])
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    rows = []
    for scene in scenes:
        r = E.eval_scene(scene, E.ROOT / args.run_root, args.dist,
                         mode="agnostic", subset=args.subset, verbose=False)
        if r is None:
            print(f"  [跳过] {scene}")
            continue
        pid = r["pid_list"]
        pb = r["pred_best"]
        ious = np.array([pb.get(str(p), [-1, 0.0])[1] for p in pid])
        gtm = np.array([pb.get(str(p), [-1, 0.0])[0] for p in pid])
        obs = np.array([r["id_score"].get(str(p), 0.0) for p in pid])
        n_gt = r["n_gt"]

        out = {"scene": scene, "n_gt": n_gt, "n_pred": len(pid)}
        for t in (0.25, 0.50, 0.75):
            key = f"{int(t*100)}"
            out[f"obs{key}"] = ap_with(obs, ious, gtm, n_gt, t)
            out[f"orc{key}"] = ap_with(ious.copy(), ious, gtm, n_gt, t)
            out[f"rev{key}"] = ap_with(-obs, ious, gtm, n_gt, t)
            rng = np.random.default_rng(0)
            out[f"rnd{key}"] = float(np.mean([
                ap_with(rng.random(len(pid)), ious, gtm, n_gt, t)
                for _ in range(20)]))
        rows.append(out)
        print(f"{scene:<9} GT {n_gt:3d} 预测 {len(pid):3d} | "
              f"AP50: log(obs)={out['obs50']*100:5.1f}  "
              f"oracle={out['orc50']*100:5.1f}  "
              f"reverse={out['rev50']*100:5.1f}  "
              f"random={out['rnd50']*100:5.1f}   || "
              f"AP25: obs={out['obs25']*100:5.1f} oracle={out['orc25']*100:5.1f}   || "
              f"AP75: obs={out['orc75']*100:5.1f} oracle={out['orc75']*100:5.1f}")

    if rows:
        print("\n=== 汇总（8 场景均值）===")
        for key in ("obs", "orc", "rev", "rnd"):
            line = f"{key:8s}"
            for t in ("25", "50", "75"):
                line += f"  AP{t} {np.mean([r[f'{key}{t}'] for r in rows])*100:5.1f}"
            print(line)


if __name__ == "__main__":
    main()
