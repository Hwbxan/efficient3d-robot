import json
from pathlib import Path

base = Path("/data/efficient3d_robot/outputs")
cfgs = [
    ("baseline (N=1, 每帧检测)", base / "dense_office0/office_0"),
    ("N=5  (1检4传)", base / "rt_n5"),
    ("N=10 (1检9传)", base / "rt_n10"),
]

rows = []
for name, root in cfgs:
    ev = json.loads((root / "gt_eval_dense.json").read_text())
    lat = json.loads((root / "latency.json").read_text())
    frames = lat.get("frames", [])
    tot = sum(f.get("total_ms", 0.0) for f in frames)
    n = len(frames)
    mean = tot / n if n else 0.0
    det = [f for f in frames if f.get("mode", "detect") == "detect"]
    pro = [f for f in frames if f.get("mode") == "propagate"]
    dm = sum(f["total_ms"] for f in det) / len(det) if det else 0.0
    pm = sum(f["total_ms"] for f in pro) / len(pro) if pro else 0.0
    rows.append(dict(
        name=name, n=n, mean=mean, fps=1000.0 / mean if mean else 0,
        det_n=len(det), pro_n=len(pro), det_ms=dm, pro_ms=pm, ev=ev,
    ))

print("=" * 112)
print("密集 GT（office_0，241 连续帧，stride=1）下的诚实精度 vs 帧率")
print("=" * 112)
hdr = (f"{'配置':<24}{'FPS':>7}{'均值ms':>9}{'检帧':>6}{'传帧':>6}{'检ms':>8}{'传ms':>8}"
       f"{'AP@.25':>8}{'AP@.50':>8}{'P@.50':>7}{'R@.50':>7}{'GT':>7}{'Pred':>7}{'单轨率':>8}{'标签一致':>9}")
print(hdr)
print("-" * 112)
for r in rows:
    ev = r["ev"]
    d = ev["detection"]
    a25, a50 = d["iou_0.25"], d["iou_0.5"]
    asc = ev.get("association", {})
    lc = ev.get("label_consistency", {}).get("ratio", 0)
    print(f"{r['name']:<24}{r['fps']:>7.2f}{r['mean']:>9.1f}{r['det_n']:>6}{r['pro_n']:>6}"
          f"{r['det_ms']:>8.1f}{r['pro_ms']:>8.1f}"
          f"{a25['ap']:>8.3f}{a50['ap']:>8.3f}{a50['precision']:>7.3f}{a50['recall']:>7.3f}"
          f"{a50['gt_instances']:>7}{a50['predictions']:>7}"
          f"{asc.get('single_track_ratio', 0):>8.3f}{lc:>9.3f}")

print()
print("--- 各配置的 association 明细 ---")
for r in rows:
    print(r["name"], json.dumps(r["ev"].get("association", {}), ensure_ascii=False)[:300])

print()
print("--- gt_eval_dense.json 顶层键 ---")
print(list(rows[0]["ev"].keys()))
print()
print(json.dumps(rows[0]["ev"], ensure_ascii=False)[:2500])
