"""打分器第二轮：稳定性检验 + 融合搜索。

231 个样本 / 8 场景，单折 CV 的 AP 抖动可能很大。这里做两件事：
  1. 留一场景 CV 的逐折结果 + 跨场景标准误，看增益是不是被某一两个场景撑起来的
  2. 搜索"无学习启发式 x 学习模型"的 rank 融合，取稳健解
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

FEAT = Path("/workspace/rank_feat_v10.json")
THRESH = (0.25, 0.5, 0.75)


def ap(scores, ious, gt_ids, n_gt, t, n_point=101):
    order = np.argsort(-np.asarray(scores, dtype=float))
    ious = np.asarray(ious, dtype=float)[order]
    gt_ids = np.asarray(gt_ids, dtype=np.int64)[order]
    used, tp = set(), np.zeros(len(order))
    for k in range(len(order)):
        if ious[k] >= t and gt_ids[k] >= 0 and int(gt_ids[k]) not in used:
            tp[k] = 1.0
            used.add(int(gt_ids[k]))
    tp_c = np.cumsum(tp)
    prec = tp_c / (np.arange(len(order)) + 1.0)
    rec = tp_c / max(n_gt, 1)
    pf = np.maximum.accumulate(prec[::-1])[::-1]
    grid = np.linspace(0, 1, n_point)
    return float(np.mean([pf[rec >= r].max() if (rec >= r).any() else 0.0
                          for r in grid]))


def per_scene_ap(rows, score_map):
    out = {}
    for sc in sorted({r["scene"] for r in rows}):
        sub = [r for r in rows if r["scene"] == sc]
        s = np.array([score_map[id(r)] for r in sub], float)
        out[sc] = {t: ap(s, [r["iou"] for r in sub], [r["gt_id"] for r in sub],
                         sub[0]["n_gt"], t) * 100 for t in THRESH}
    return out


def macro(per):
    return {t: float(np.mean([per[sc][t] for sc in per])) for t in THRESH}


def log1p(a):
    return np.log1p(np.maximum(np.asarray(a, dtype=float), 0.0))


def main():
    rows = json.loads(FEAT.read_text())
    scenes = sorted({r["scene"] for r in rows})
    y = np.array([r["iou"] for r in rows])

    # ---- 基础打分 ----
    base = {id(r): float(np.log1p(max(r["obs"], 0))) for r in rows}
    fo = {id(r): float(r["fo_mean"]) for r in rows}

    per_base = per_scene_ap(rows, base)
    per_fo = per_scene_ap(rows, fo)
    m0, m1 = macro(per_base), macro(per_fo)

    print(f"{'场景':<10}{'AP25 基线->fo':>18}{'AP50 基线->fo':>18}"
          f"{'AP75 基线->fo':>18}")
    for sc in scenes:
        print(f"{sc:<10}"
              f"{per_base[sc][0.25]:>8.1f} ->{per_fo[sc][0.25]:>7.1f}"
              f"{per_base[sc][0.5]:>9.1f} ->{per_fo[sc][0.5]:>7.1f}"
              f"{per_base[sc][0.75]:>9.1f} ->{per_fo[sc][0.75]:>7.1f}")
    print(f"{'宏平均':<10}{m0[0.25]:>8.1f} ->{m1[0.25]:>7.1f}"
          f"{m0[0.5]:>9.1f} ->{m1[0.5]:>7.1f}"
          f"{m0[0.75]:>9.1f} ->{m1[0.75]:>7.1f}")
    d = {t: np.array([per_fo[sc][t] - per_base[sc][t] for sc in scenes])
         for t in THRESH}
    print("\nfo_mean 相对基线的逐场景增益（均值 ± 标准误）：")
    for t in THRESH:
        print(f"  AP{int(t*100)}: {d[t].mean():+.2f} ± {d[t].std(ddof=1)/np.sqrt(8):.2f}"
              f"   （{int((d[t]>0).sum())}/8 场景变好，"
              f"最差 {d[t].min():+.1f} 最好 {d[t].max():+.1f}）")

    # ---- 学习模型：留一场景 CV ----
    FKEYS = ["log_obs", "log_fo", "log_vox", "fill", "det_max", "ext",
             "log_area", "fo_f5", "log_input", "mfvr"]

    def feats(sub):
        return np.nan_to_num(np.column_stack([
            log1p([r["obs"] for r in sub]), log1p([r["fo_mean"] for r in sub]),
            log1p([r["vox"] for r in sub]),
            np.asarray([r["fill"] for r in sub], float),
            np.asarray([r["det_max"] for r in sub], float),
            np.asarray([r["ext"] for r in sub], float),
            log1p([r["area_mean"] for r in sub]),
            np.asarray([r["fo_f5"] for r in sub], float),
            log1p([r["input_pts"] for r in sub]),
            np.asarray([r["mfvr"] for r in sub], float),
        ]), nan=0.0, posinf=0.0, neginf=0.0)

    def prior(tr, te):
        acc = {}
        for r in tr:
            acc.setdefault(r["label"], []).append(r["iou"])
        glob = float(np.mean([r["iou"] for r in tr]))
        pri = {k: float(np.mean(v)) for k, v in acc.items()}
        return np.array([pri.get(r["label"], glob) for r in te], float)

    def cv(scenes_, mk, use_prior=True, small=False):
        sm = {}
        for sc in scenes_:
            tr = [r for r in rows if r["scene"] != sc]
            te = [r for r in rows if r["scene"] == sc]
            Xtr, Xte = feats(tr), feats(te)
            if use_prior:
                Xtr = np.column_stack([Xtr, prior(tr, tr)])
                Xte = np.column_stack([Xte, prior(tr, te)])
            if small:  # 只用 3 个最有信号的原始特征 + 先验
                Xtr, Xte = Xtr[:, [0, 1, 2, -1]], Xte[:, [0, 1, 2, -1]]
            m = mk()
            m.fit(Xtr, np.array([r["iou"] for r in tr], float))
            for r, v in zip(te, m.predict(Xte)):
                sm[id(r)] = float(v)
        return sm

    MKS = {
        "Ridge a=1": lambda: Ridge(alpha=1.0),
        "Ridge a=3": lambda: Ridge(alpha=3.0),
        "Ridge a=10": lambda: Ridge(alpha=10.0),
        "Ridge a=30": lambda: Ridge(alpha=30.0),
        "HGB d2": lambda: HistGradientBoostingRegressor(
            max_iter=120, max_depth=2, learning_rate=0.06,
            min_samples_leaf=8, l2_regularization=1.0, random_state=0),
        "HGB d3": lambda: HistGradientBoostingRegressor(
            max_iter=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=5, l2_regularization=1.0, random_state=0),
    }
    print(f"\n{'模型':<16}{'AP25':>7}{'AP50':>7}{'AP75':>7}{'Spear':>8}"
          f"{'AP25 标准误':>12}")
    cvs = {}
    for nm, mk in MKS.items():
        for small in (False, True):
            sm = cv(scenes, mk, True, small)
            per = per_scene_ap(rows, sm)
            m = macro(per)
            all_s = np.array([sm[id(r)] for r in rows])
            c = spearmanr(all_s, y).correlation
            se = np.std([per[sc][0.25] for sc in scenes], ddof=1) / np.sqrt(8)
            tag = nm + (" [4feat]" if small else "")
            cvs[tag] = (m, sm)
            print(f"{tag:<16}{m[0.25]:>7.1f}{m[0.5]:>7.1f}{m[0.75]:>7.1f}"
                  f"{c:>8.3f}{se:>12.2f}")

    # ---- rank 融合：启发式 fo_mean + 学习模型 ----
    print(f"\n{'rank 融合':<26}{'AP25':>7}{'AP50':>7}{'AP75':>7}")
    for tag, (m, sm) in cvs.items():
        for w in (0.3, 0.5, 0.7):
            def rk(a):
                o = np.argsort(np.argsort(a))
                return o / max(len(a) - 1, 1)
            mix = {}
            for sc in scenes:
                sub = [r for r in rows if r["scene"] == sc]
                a = rk(np.array([fo[id(r)] for r in sub]))
                b = rk(np.array([sm[id(r)] for r in sub]))
                for r, v in zip(sub, w * a + (1 - w) * b):
                    mix[id(r)] = float(v)
            mm = macro(per_scene_ap(rows, mix))
            print(f"{tag + f' w={w}':<26}{mm[0.25]:>7.1f}{mm[0.5]:>7.1f}"
                  f"{mm[0.75]:>7.1f}")


if __name__ == "__main__":
    main()
