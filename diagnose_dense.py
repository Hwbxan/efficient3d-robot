"""密集 GT 下的逐类别诊断：AP@.50 到底丢在哪。

GT 存储格式：instance_masks/instance{frame:06d}.png 是一张 uint16 标签图，
像素值 = Replica object_id；该帧可见 object_id 列表见 manifest.frames[f].visible_objects。
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import sys
GT_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/efficient3d_robot/outputs/gt_dense/office_0")
RUN = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/data/efficient3d_robot/outputs/dense_office0/office_0")
print(f"GT: {GT_ROOT}\nRUN: {RUN}")
IOU = 0.5
MIN_AREA = 100
# 与 evaluate_against_gt 的 --gt-classes 白名单一致：只有这些类别计入指标
WHITELIST = {
    "bin", "basket", "tissue-paper", "chair", "sofa", "stool", "armchair",
    "couch", "door", "table", "desk", "desk-organizer", "tv-screen",
    "tablet", "monitor",
}

gt_manifest = json.loads((GT_ROOT / "gt_manifest.json").read_text())
obj2cls = {int(k): v for k, v in gt_manifest["object_id_to_class"].items()}
gt_frames = {int(k): v for k, v in gt_manifest["frames"].items()}


def pred_masks(frame_index):
    seg_json = RUN / "segmentation" / f"frame_{frame_index:06d}_instances.json"
    if not seg_json.is_file():
        return []
    seg = json.loads(seg_json.read_text())
    if isinstance(seg, dict):
        seg = seg.get("instances", [])
    out = []
    for inst in seg:
        mp = Path(inst["mask_path"])
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        out.append((inst["label"], m > 0))
    return out


gt_class_tp = defaultdict(int)
gt_class_total = defaultdict(int)
pred_label_tp = defaultdict(int)
pred_label_total = defaultdict(int)
pred_confusion = defaultdict(lambda: defaultdict(int))  # pred_label -> gt_class

for fi in sorted(gt_frames):
    gt_path = GT_ROOT / "instance_masks" / f"instance{fi:06d}.png"
    if not gt_path.is_file():
        continue
    labelmap = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
    if labelmap is None:
        continue

    preds = pred_masks(fi)
    gt_items = []
    for oid in gt_frames[fi]["visible_objects"]:
        m = labelmap == oid
        if m.sum() < MIN_AREA:
            continue
        gt_items.append((oid, obj2cls.get(oid, "?"), m))

    for label, _ in preds:
        pred_label_total[label] += 1

    pairs = []
    for pi, (label, pm) in enumerate(preds):
        for gi, (oid, gcls, gm) in enumerate(gt_items):
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            if union:
                iou = inter / union
                if iou >= IOU:
                    pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pred_label_tp[preds[pi][0]] += 1
        gcls = gt_items[gi][1]
        gt_class_tp[gcls] += 1
        pred_confusion[preds[pi][0]][gcls] += 1

    for _, label, _ in gt_items:
        gt_class_total[label] += 1

in_wl = lambda c: c in WHITELIST
print("=" * 86)
print("GT 类别召回（IoU≥0.5）· office_0 密集 241 帧 · 每帧都检测")
print("★ = 在 15 类白名单内（计入 AP）；其余类别不在评测口径内，召回低属正常")
print("=" * 86)
print(f"{'':<2}{'GT 类别':<22}{'实例数':>8}{'命中':>8}{'召回':>8}")
for c in sorted(gt_class_total, key=lambda k: -gt_class_total[k]):
    t, h = gt_class_total[c], gt_class_tp[c]
    print(f"{'★' if in_wl(c) else ' ':<2}{c:<22}{t:>8}{h:>8}{(h / t if t else 0):>8.3f}")
wl_t = sum(v for k, v in gt_class_total.items() if in_wl(k))
wl_h = sum(v for k, v in gt_class_tp.items() if in_wl(k))
print(f"{'':<2}{'白名单合计':<22}{wl_t:>8}{wl_h:>8}{(wl_h / wl_t if wl_t else 0):>8.3f}")

print()
print("=" * 86)
print("预测标签精度")
print("=" * 86)
print(f"{'预测标签':<22}{'预测数':>8}{'命中':>8}{'精度':>8}   命中的都是哪些 GT 类别")
for c in sorted(pred_label_total, key=lambda k: -pred_label_total[k]):
    t, h = pred_label_total[c], pred_label_tp[c]
    top = sorted(pred_confusion[c].items(), key=lambda kv: -kv[1])[:4]
    top_s = ", ".join(f"{k}×{v}" for k, v in top)
    print(f"{c:<22}{t:>8}{h:>8}{(h / t if t else 0):>8.3f}   {top_s}")
