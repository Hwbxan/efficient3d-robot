"""把 v2 的召回拆成「检测帧」和「几何传播帧」两组，并给出 IoU 分布。

目的：区分两类精度损失
  A. 词汇/检测能力问题 —— 检测帧上就召回低（提示词没覆盖、DINO 检不出）
  B. 传播退化问题     —— 检测帧召回高、传播帧召回低（几何搬移的掩码不够准）
两者要的修复完全不同：A 加提示词，B 提掩码质量/提高检测频率。
"""
import json
import sys
import collections
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/data/efficient3d_robot")
RUNROOT = ROOT / "outputs" / (sys.argv[1] if len(sys.argv) > 1 else "rt8_v2")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # detect-interval

GTPAIR = [("office0", "office_0"), ("office1", "office_1"), ("office2", "office_2"),
          ("office3", "office_3"), ("office4", "office_4"), ("room0", "room_0"),
          ("room1", "room_1"), ("room2", "room_2")]
WHITELIST = {"bin", "basket", "tissue-paper", "chair", "sofa", "stool", "armchair",
             "couch", "door", "table", "desk", "desk-organizer", "tv-screen",
             "tablet", "monitor"}
AREA_FRAC = 0.4


def pred_masks(run, fi):
    seg_json = run / "segmentation" / f"frame_{fi:06d}_instances.json"
    if not seg_json.is_file():
        return []
    seg = json.loads(seg_json.read_text())
    if isinstance(seg, dict):
        seg = seg.get("instances", [])
    out = []
    for inst in seg:
        m = cv2.imread(str(inst["mask_path"]), cv2.IMREAD_UNCHANGED)
        if m is not None:
            out.append((inst.get("label", "?"), m > 0))
    return out


det_total = collections.Counter(); det_hit = collections.Counter()
det_hit25 = collections.Counter()
pro_total = collections.Counter(); pro_hit = collections.Counter()
pro_hit25 = collections.Counter()
iou_hist = collections.defaultdict(collections.Counter)  # cls -> bucket

for sname, gtname in GTPAIR:
    run = RUNROOT / sname
    gt = ROOT / "outputs" / "gt8" / gtname
    if not (run / "segmentation").is_dir():
        continue
    man = json.loads((gt / "gt_manifest.json").read_text())
    o2c = {int(k): v for k, v in man["object_id_to_class"].items()}
    frames = {int(k): v for k, v in man["frames"].items()}
    for fi in sorted(frames):
        mp = gt / "instance_masks" / f"instance{fi:06d}.png"
        if not mp.is_file():
            continue
        lab = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if lab is None:
            continue
        H, W = lab.shape
        ids = [int(o) for o in frames[fi]["visible_objects"] if int(o) in o2c]
        if not ids:
            continue
        preds = pred_masks(run, fi)
        is_det = (fi % N == 0)
        for oid in ids:
            cls = o2c[oid]
            if cls not in WHITELIST:
                continue
            gm = lab == oid
            a = int(gm.sum())
            if a == 0 or a / (H * W) > AREA_FRAC:
                continue
            best = 0.0
            for _l, pm in preds:
                inter = int(np.logical_and(gm, pm).sum())
                union = int(np.logical_or(gm, pm).sum())
                if union and inter / union > best:
                    best = inter / union
            b = min(int(best * 10), 9)
            iou_hist[cls][b] += 1
            if is_det:
                det_total[cls] += 1
                if best >= .5: det_hit[cls] += 1
                if best >= .25: det_hit25[cls] += 1
            else:
                pro_total[cls] += 1
                if best >= .5: pro_hit[cls] += 1
                if best >= .25: pro_hit25[cls] += 1

print("=" * 96)
print(f"召回拆分：检测帧（fi%{N}==0） vs 几何传播帧    —— 密集 GT stride=8，故传播帧占绝大多数")
print("=" * 96)
print(f"{'GT 类别':<16}{'检测帧':>7}{'R@.5':>7}{'R@.25':>7}   {'传播帧':>7}{'R@.5':>7}{'R@.25':>7}   {'落差.5':>8}")
for c in sorted(set(det_total) | set(pro_total),
                key=lambda k: -(det_total[k] + pro_total[k])):
    dt, dh, dh25 = det_total[c], det_hit[c], det_hit25[c]
    pt, ph, ph25 = pro_total[c], pro_hit[c], pro_hit25[c]
    r1 = dh / dt if dt else 0
    r2 = ph / pt if pt else 0
    print(f"{c:<16}{dt:>7}{r1:>7.3f}{(dh25/dt if dt else 0):>7.3f}   "
          f"{pt:>7}{r2:>7.3f}{(ph25/pt if pt else 0):>7.3f}   {r1-r2:>8.3f}")
DT, DH, DH25 = sum(det_total.values()), sum(det_hit.values()), sum(det_hit25.values())
PT, PH, PH25 = sum(pro_total.values()), sum(pro_hit.values()), sum(pro_hit25.values())
print(f"{'合计':<16}{DT:>7}{DH/DT:>7.3f}{DH25/DT:>7.3f}   "
      f"{PT:>7}{PH/PT:>7.3f}{PH25/PT:>7.3f}   {DH/DT-PH/PT:>8.3f}")

print("\n" + "=" * 96)
print("最佳 IoU 分布（各 GT 类别，10 档：.0 .1 .2 … .9+）")
print("=" * 96)
print(f"{'GT 类别':<16}" + "".join(f"{i/10:>7.1f}" for i in range(10)) + f"{'n':>8}")
for c in sorted(iou_hist, key=lambda k: -sum(iou_hist[k].values())):
    h = iou_hist[c]; n = sum(h.values())
    print(f"{c:<16}" + "".join(f"{h.get(i,0)/n:>7.2f}" for i in range(10)) + f"{n:>8}")
