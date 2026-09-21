"""小物体在哪一级被吃掉：逐帧实例 -> 跟踪轨道 -> 融合实例，逐级数标签。"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/data/efficient3d_robot")
SMALL = {"box", "book", "bottle", "plate", "tissue box", "tissue-paper",
         "tablet", "camera", "clock", "bowl", "vase", "cup", "candle",
         "desk organizer", "desk-organizer", "remote", "phone", "lamp"}

def count(run_dir):
    per_frame = Counter()
    inst_root = run_dir / "instances_3d"
    for p in sorted(inst_root.glob("frame_*/instances_3d.json")):
        for rec in json.loads(p.read_text()):
            per_frame[str(rec.get("label", "?")).strip().lower()] += 1
    tracks = Counter()
    tp = run_dir / "association" / "tracking.json"
    if tp.is_file():
        tr = json.loads(tp.read_text())
        lab = {}
        for fr in tr["frames"]:
            for a in fr["associations"]:
                if a.get("global_id") is not None:
                    lab.setdefault(int(a["global_id"]),
                                   str(a.get("raw_label", "?")).strip().lower())
        tracks = Counter(lab.values())
        n_tracks = len(lab)
    else:
        n_tracks = 0
    fused = Counter()
    fp = run_dir / "fusion_attempt_01" / "instance_map.json"
    if fp.is_file():
        ins = json.loads(fp.read_text())["instances"]
        fused = Counter(str(i.get("label", "?")).strip().lower() for i in ins)
        n_fused = len(ins)
    else:
        n_fused = 0
    return per_frame, tracks, n_tracks, fused, n_fused


for run in sys.argv[1:]:
    for sc in ["office2", "room0"]:
        d = ROOT / run / sc
        if not d.is_dir():
            continue
        pf, tr, nt, fu, nf = count(d)
        print("=" * 92)
        print(run, sc, "轨道", nt, "融合", nf)
        keys = sorted(set(list(pf) + list(tr) + list(fu)))
        print(f"{'标签':<16}{'逐帧次数':>9}{'轨道数':>7}{'融合数':>7}")
        for k in keys:
            if k not in SMALL:
                continue
            print(f"{k:<16}{pf.get(k,0):>9}{tr.get(k,0):>7}{fu.get(k,0):>7}")
