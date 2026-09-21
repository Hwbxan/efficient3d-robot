"""实例质量打分器：留一场景交叉验证，只许用系统自身输出的特征。

为什么值得做：AP 是 PR 曲线下面积，预测集合不变时完全由排序决定。
当前置信度是 log(1+观测帧数)，Spearman 只有 +0.502；而"体素平均被多少帧
确认"（fo_mean）有 +0.562。把排序修好是几乎零成本的精度提升。

评估方式：留一场景（7 场景训练 / 1 场景测试），8 折，绝不泄漏。
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

FEAT = Path("/workspace/rank_feat_v10.json")


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


def eval_scores(rows, score_of):
    """score_of(rows) -> 每个实例的分数；返回宏平均 AP25/50/75。"""
    per = {}
    for sc in sorted({r["scene"] for r in rows}):
        sub = [r for r in rows if r["scene"] == sc]
        s = score_of(sub)
        ious = [r["iou"] for r in sub]
        gts = [r["gt_id"] for r in sub]
        n_gt = sub[0]["n_gt"]
        per[sc] = {t: ap(s, ious, gts, n_gt, t) for t in (0.25, 0.5, 0.75)}
    return ({t: float(np.mean([per[sc][t] for sc in per])) * 100
             for t in (0.25, 0.5, 0.75)}, per)


def log1p(a):
    return np.log1p(np.maximum(np.asarray(a, dtype=float), 0.0))


def main():
    rows = json.loads(FEAT.read_text())
    scenes = sorted({r["scene"] for r in rows})
    y = np.array([r["iou"] for r in rows])
    print(f"{len(rows)} 个实例 / {len(scenes)} 场景  "
          f"IoU mean {y.mean():.3f}  命中>=.25 {int((y>=.25).sum())} "
          f">=.5 {int((y>=.5).sum())}\n")

    def g(k):
        return np.array([r[k] for r in rows], dtype=float)

    cands = {
        "基线 log1p(obs)": lambda s: log1p([r["obs"] for r in s]),
        "fo_mean": lambda s: np.asarray([r["fo_mean"] for r in s], float),
        "log1p(fo_mean)": lambda s: log1p([r["fo_mean"] for r in s]),
        "log1p(obs)*log1p(fo_mean)": lambda s: (
            log1p([r["obs"] for r in s])
            * log1p([r["fo_mean"] for r in s])),
        "log1p(obs)+log1p(fo_mean)": lambda s: (
            log1p([r["obs"] for r in s])
            + log1p([r["fo_mean"] for r in s])),
        "fo_mean/obs": lambda s: (
            np.asarray([r["fo_mean"] for r in s], float)
            / (np.asarray([r["obs"] for r in s], float) + 1.0)),
        "log1p(n_det)": lambda s: log1p([r["n_det"] for r in s]),
        "det_max": lambda s: np.asarray([r["det_max"] for r in s], float),
    }

    print(f"{'打分方式':<34}{'AP25':>7}{'AP50':>7}{'AP75':>7}   {'Spearman':>9}")
    results = {}
    for name, fn in cands.items():
        m, _ = eval_scores(rows, fn)
        all_s = np.concatenate([fn([r for r in rows if r["scene"] == sc])
                                for sc in scenes])
        c = spearmanr(all_s, y).correlation
        results[name] = m
        print(f"{name:<34}{m[0.25]:>7.1f}{m[0.5]:>7.1f}{m[0.75]:>7.1f}"
              f"   {c:>+9.3f}")

    # oracle 上界
    m, _ = eval_scores(rows, lambda s: np.asarray([r["iou"] for r in s], float))
    print(f"{'ORACLE（用真实 IoU 排序）':<34}{m[0.25]:>7.1f}{m[0.5]:>7.1f}"
          f"{m[0.75]:>7.1f}   {'1.000':>9}")

    # ---------- 留一场景 CV 学习打分 ----------
    FKEYS = ["log_obs", "log_fo", "log_vox", "fill", "det_max", "ext",
             "log_area", "fo_f5", "log_input", "mfvr"]

    def feats(sub):
        X = np.column_stack([
            log1p([r["obs"] for r in sub]),
            log1p([r["fo_mean"] for r in sub]),
            log1p([r["vox"] for r in sub]),
            np.asarray([r["fill"] for r in sub], float),
            np.asarray([r["det_max"] for r in sub], float),
            np.asarray([r["ext"] for r in sub], float),
            log1p([r["area_mean"] for r in sub]),
            np.asarray([r["fo_f5"] for r in sub], float),
            log1p([r["input_pts"] for r in sub]),
            np.asarray([r["mfvr"] for r in sub], float),
        ])
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def label_prior(train, test):
        """类别先验：用训练集里该 label 的平均 IoU（越小越要下沉）。"""
        acc = {}
        for r in train:
            acc.setdefault(r["label"], []).append(r["iou"])
        glob = np.mean([r["iou"] for r in train]) if train else 0.0
        pri = {k: float(np.mean(v)) for k, v in acc.items()}
        return np.array([pri.get(r["label"], glob) for r in test], float)

    def cv_score(model_fn, use_prior=True, name=""):
        out = {}
        for sc in scenes:
            tr = [r for r in rows if r["scene"] != sc]
            te = [r for r in rows if r["scene"] == sc]
            Xtr, Xte = feats(tr), feats(te)
            ytr = np.array([r["iou"] for r in tr], float)
            if use_prior:
                Xtr = np.column_stack([Xtr, label_prior(tr, tr)])
                Xte = np.column_stack([Xte, label_prior(tr, te)])
            m = model_fn()
            m.fit(Xtr, ytr)
            out[sc] = m.predict(Xte)
        s_map = {}
        for sc in scenes:
            for r, v in zip([r for r in rows if r["scene"] == sc], out[sc]):
                s_map[id(r)] = v
        return lambda sub: np.array([s_map[id(r)] for r in sub], float)

    models = {
        "CV Ridge(alpha=1)": (lambda: Ridge(alpha=1.0), True),
        "CV Ridge(alpha=10)": (lambda: Ridge(alpha=10.0), True),
        "CV Ridge(无类别先验)": (lambda: Ridge(alpha=1.0), False),
        "CV HGB(浅)": (lambda: HistGradientBoostingRegressor(
            max_iter=120, max_depth=2, learning_rate=0.06,
            min_samples_leaf=8, l2_regularization=1.0,
            random_state=0), True),
        "CV HGB(中)": (lambda: HistGradientBoostingRegressor(
            max_iter=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=5, l2_regularization=1.0,
            random_state=0), True),
    }
    print()
    for name, (fn, pr) in models.items():
        scorer = cv_score(fn, pr)
        m, _ = eval_scores(rows, scorer)
        all_s = np.concatenate([scorer([r for r in rows if r["scene"] == s])
                                for s in scenes])
        c = spearmanr(all_s, y).correlation
        print(f"{name:<34}{m[0.25]:>7.1f}{m[0.5]:>7.1f}{m[0.75]:>7.1f}"
              f"   {c:>+9.3f}")


if __name__ == "__main__":
    main()
