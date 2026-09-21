"""v2 八场景逐类别召回诊断（密集 GT，stride=8）—— 找出精度缺口到底在哪。

与 evaluate_against_gt 的区别：这里按 GT 类别拆开统计"有多少 GT 实例被 IoU≥0.5 命中"，
并且额外统计"命中了但被 max-area 过滤 / 完全没预测"的情况。
"""
import json
import sys
import collections
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/data/efficient3d_robot")
RUNROOT = ROOT / "outputs" / sys.argv[1] if len(sys.argv) > 1 else ROOT / "outputs/rt8_v2"
GTPAIR = [("office0", "office_0"), ("office1", "office_1"), ("office2", "office_2"),
          ("office3", "office_3"), ("office4", "office_4"), ("room0", "room_0"),
          ("room1", "room_1"), ("room2", "room_2")]

WHITELIST = {"bin", "basket", "tissue-paper", "chair", "sofa", "stool", "armchair",
             "couch", "door", "table", "desk", "desk-organizer", "tv-screen",
             "tablet", "monitor"}
AREA_FRAC = 0.4  # 与 evaluate_against_gt --max-area-fraction 一致


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
        if m is None:
            continue
        out.append((inst.get("label", "?"), m > 0, inst.get("global_id")))
    return out


grand_total = collections.Counter()
grand_hit = collections.Counter()
grand_hit25 = collections.Counter()
per_scene = {}

for sname, gtname in GTPAIR:
    run = RUNROOT / sname
    gt = ROOT / "outputs" / "gt8" / gtname
    if not (run / "segmentation").is_dir() or not gt.is_dir():
        print(f"[跳过] {sname}")
        continue
    man = json.loads((gt / "gt_manifest.json").read_text())
    o2c = {int(k): v for k, v in man["object_id_to_class"].items()}
    frames = {int(k): v for k, v in man["frames"].items()}

    total = collections.Counter()
    hit = collections.Counter()
    hit25 = collections.Counter()
    npred_by_cls = collections.Counter()

    for fi in sorted(frames):
        f = frames[fi]
        mp = gt / "instance_masks" / f"instance{fi:06d}.png"
        if not mp.is_file():
            continue
        lab = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if lab is None:
            continue
        H, W = lab.shape
        ids = [int(o) for o in f["visible_objects"]]
        ids = [i for i in ids if i in o2c]
        if not ids:
            continue
        preds = pred_masks(run, fi)
        for oid in ids:
            cls = o2c[oid]
            if cls not in WHITELIST:
                continue
            gm = lab == oid
            area = int(gm.sum())
            if area == 0 or area / (H * W) > AREA_FRAC:
                continue
            total[cls] += 1
            best = 0.0
            for _l, pm, _g in preds:
                inter = int(np.logical_and(gm, pm).sum())
                union = int(np.logical_or(gm, pm).sum())
                if union and inter / union > best:
                    best = inter / union
            if best >= 0.5:
                hit[cls] += 1
            if best >= 0.25:
                hit25[cls] += 1
        for l, pm, _g in preds:
            npred_by_cls[l] += 1

    per_scene[sname] = (total, hit)
    grand_total.update(total)
    grand_hit.update(hit)
    grand_hit25.update(hit25)
    print(f"\n--- {sname} --- 预测标签分布 {dict(npred_by_cls.most_common(8))}")

print("\n" + "=" * 78)
print("八场景合计 · GT 类别召回（IoU≥0.5 / ≥0.25）")
print("=" * 78)
print(f"{'GT 类别':<18}{'实例数':>8}{'≥.5':>8}{'召回.5':>9}{'≥.25':>8}{'召回.25':>9}")
for c in sorted(grand_total, key=lambda k: -grand_total[k]):
    t, h, h25 = grand_total[c], grand_hit[c], grand_hit25[c]
    print(f"{c:<18}{t:>8}{h:>8}{h/t if t else 0:>9.3f}{h25:>8}{h25/t if t else 0:>9.3f}")
T, H, H25 = sum(grand_total.values()), sum(grand_hit.values()), sum(grand_hit25.values())
print(f"{'合计':<18}{T:>8}{H:>8}{H/T:>9.3f}{H25:>8}{H25/T:>9.3f}")

out = {s: {"total": dict(t), "hit": dict(h)} for s, (t, h) in per_scene.items()}
out["_all"] = {"total": dict(grand_total), "hit": dict(grand_hit),
               "hit25": dict(grand_hit25)}
Path("/data/efficient3d_robot/diag8.json").write_text(json.dumps(out, indent=2))
print("\n写入 diag8.json")
