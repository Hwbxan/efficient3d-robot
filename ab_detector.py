"""检测器 / 提示词 A/B 实验台（office_0，241 密集帧，N=10，密集 GT）。

为什么可以跑得很快：N=10 时只有 25 帧需要真正过检测器，其余 216 帧走几何传播
（12.5 ms/帧）。而密集 GT 覆盖全部 241 帧，所以传播帧也会被评到 —— 一次实验约 1 分钟。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data/efficient3d_robot")
PY = "/miniconda3/bin/python3"
GT = "outputs/gt_dense/office_0"
CLASSES = ("bin,basket,tissue-paper,chair,sofa,stool,armchair,couch,door,"
           "table,desk,desk-organizer,tv-screen,tablet,monitor")
FRAMES = " ".join(str(i) for i in range(241))
OUTROOT = ROOT / "outputs/ab"

BASE = "checkpoints/grounding-dino-base"
TINY = "checkpoints/grounding-dino-tiny"
P1 = ["tv screen", "chair", "desk", "trash can", "door", "sofa"]
P3 = ["tv screen", "chair", "desk", "table", "trash can", "door", "sofa"]

CONFIGS = [
    dict(name="tiny_P0", dino=TINY,
         prompts=["computer monitor", "chair", "desk", "trash can", "door", "sofa"],
         box=0.30),
    dict(name="base_P0", dino=BASE,
         prompts=["computer monitor", "chair", "desk", "trash can", "door", "sofa"],
         box=0.30),
    dict(name="base_P1", dino=BASE, prompts=P1, box=0.30),
    dict(name="tiny_P1", dino=TINY, prompts=P1, box=0.30),
    dict(name="base_P2", dino=BASE,
         prompts=["computer monitor", "chair", "table", "trash can", "door", "sofa"],
         box=0.40),
    # —— 第二轮：在 base + "tv screen" 基础上精调 ——
    dict(name="base_P3", dino=BASE, prompts=P3, box=0.30),
    dict(name="base_P4", dino=BASE, prompts=P1, box=0.25),
    dict(name="base_P5", dino=BASE, prompts=P1, box=0.35),
    dict(name="base_P6", dino=BASE, prompts=P3, box=0.25),
]


def run(cfg):
    out = OUTROOT / cfg["name"]
    log = OUTROOT / f"{cfg['name']}.log"
    cmd = [PY, "-u", "-m", "tools.run_sequence_efficient",
           "--scene-directory", "datasets/processed/Replica/office0",
           "--run-directory", str(out),
           "--frames", *FRAMES.split(),
           "--dino-model", cfg["dino"],
           "--classes", *cfg["prompts"],
           "--box-threshold", str(cfg["box"]),
           "--detect-interval", "10",
           "--propagate-stride", "2",
           "--pixel-stride", "2",
           "--skip-ply", "--no-preview", "--no-fusion"]
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"[FAIL] {cfg['name']} rc={rc}，见 {log}")
        return None

    ev_out = out / "gt_eval.json"
    cmd2 = [PY, "-u", "-m", "tools.evaluate_against_gt",
            "--tracking-json", str(out / "association/tracking.json"),
            "--segmentation-root", str(out / "segmentation"),
            "--gt-root", GT, "--output", str(ev_out),
            "--gt-classes", CLASSES, "--max-area-fraction", "0.4"]
    with open(log, "a") as fh:
        rc2 = subprocess.call(cmd2, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if rc2 != 0:
        print(f"[FAIL eval] {cfg['name']} rc={rc2}")
        return None

    ev = json.loads(ev_out.read_text())
    lat = json.loads((out / "latency.json").read_text())
    frames = lat.get("frames", [])
    mean = sum(f.get("total_ms", 0.0) for f in frames) / max(len(frames), 1)
    det = [f for f in frames if f.get("mode", "detect") == "detect"]
    pro = [f for f in frames if f.get("mode") == "propagate"]
    return dict(
        name=cfg["name"],
        fps=1000.0 / mean, mean=mean,
        det_ms=sum(f["total_ms"] for f in det) / max(len(det), 1),
        pro_ms=sum(f["total_ms"] for f in pro) / max(len(pro), 1),
        ap25=ev["detection"]["iou_0.25"]["ap"], ap50=ev["detection"]["iou_0.5"]["ap"],
        p50=ev["detection"]["iou_0.5"]["precision"], r50=ev["detection"]["iou_0.5"]["recall"],
        preds=ev["detection"]["iou_0.5"]["predictions"],
        gt=ev["detection"]["iou_0.5"]["gt_instances"],
    )


def main():
    only = sys.argv[1:] or None
    OUTROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in CONFIGS:
        if only and cfg["name"] not in only:
            continue
        print(f">>> {cfg['name']} ...", flush=True)
        r = run(cfg)
        if r:
            rows.append(r)
            print(f"    FPS={r['fps']:.2f} AP@.25={r['ap25']:.3f} AP@.50={r['ap50']:.3f}", flush=True)
    print()
    print("=" * 104)
    print(f"{'配置':<12}{'FPS':>7}{'均值ms':>9}{'检ms':>8}{'传ms':>8}"
          f"{'AP@.25':>8}{'AP@.50':>8}{'P@.50':>7}{'R@.50':>7}{'Pred':>7}{'GT':>7}")
    print("-" * 104)
    for r in rows:
        print(f"{r['name']:<12}{r['fps']:>7.2f}{r['mean']:>9.1f}{r['det_ms']:>8.1f}{r['pro_ms']:>8.1f}"
              f"{r['ap25']:>8.3f}{r['ap50']:>8.3f}{r['p50']:>7.3f}{r['r50']:>7.3f}"
              f"{r['preds']:>7}{r['gt']:>7}")
    (OUTROOT / "ab_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
