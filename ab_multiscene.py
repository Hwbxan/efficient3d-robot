"""跨场景验证：避免只在 office_0 上调参导致过拟合。

跑 2 个候选配置 × 8 场景 × 前 1000 帧（N=10），用 stride=8 的密集 GT 评测。
每场景约 40 s（100 个检测帧 × 250 ms + 900 传播帧 × 11 ms）。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data/efficient3d_robot")
PY = "/miniconda3/bin/python3"
CLASSES = ("bin,basket,tissue-paper,chair,sofa,stool,armchair,couch,door,"
           "table,desk,desk-organizer,tv-screen,tablet,monitor")
SCENES = ["office0", "office1", "office2", "office3", "office4",
          "room0", "room1", "room2"]
NFRAMES = 1000
OUTROOT = ROOT / "outputs/ab8"

BASE = "checkpoints/grounding-dino-base"
CONFIGS = {
    "P1_b030": dict(dino=BASE, prompts=["tv screen", "chair", "desk", "trash can", "door", "sofa"], box=0.30),
    "P3_b025": dict(dino=BASE, prompts=["tv screen", "chair", "desk", "table", "trash can", "door", "sofa"], box=0.25),
}


def scene_gt_name(s):
    return s.replace("office", "office_").replace("room", "room_")


def run(cfg_name, cfg, scene):
    out = OUTROOT / cfg_name / scene
    log = OUTROOT / cfg_name / f"{scene}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", "-m", "tools.run_sequence_efficient",
           "--scene-directory", f"datasets/processed/Replica/{scene}",
           "--run-directory", str(out),
           "--frames", *[str(i) for i in range(NFRAMES)],
           "--dino-model", cfg["dino"],
           "--classes", *cfg["prompts"],
           "--box-threshold", str(cfg["box"]),
           "--detect-interval", "10", "--propagate-stride", "2",
           "--pixel-stride", "2", "--skip-ply", "--no-preview", "--no-fusion"]
    with open(log, "w") as fh:
        if subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT) != 0:
            print(f"[FAIL run] {cfg_name}/{scene}"); return None

    ev_out = out / "gt_eval.json"
    cmd2 = [PY, "-u", "-m", "tools.evaluate_against_gt",
            "--tracking-json", str(out / "association/tracking.json"),
            "--segmentation-root", str(out / "segmentation"),
            "--gt-root", f"outputs/gt8/{scene_gt_name(scene)}",
            "--output", str(ev_out), "--gt-classes", CLASSES,
            "--max-area-fraction", "0.4"]
    with open(log, "a") as fh:
        if subprocess.call(cmd2, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT) != 0:
            print(f"[FAIL eval] {cfg_name}/{scene}"); return None

    ev = json.loads(ev_out.read_text())
    lat = json.loads((out / "latency.json").read_text())
    fr = lat.get("frames", [])
    mean = sum(f.get("total_ms", 0.0) for f in fr) / max(len(fr), 1)
    return dict(scene=scene, fps=1000.0 / mean, mean=mean,
                ap25=ev["detection"]["iou_0.25"]["ap"],
                ap50=ev["detection"]["iou_0.5"]["ap"],
                gt=ev["detection"]["iou_0.5"]["gt_instances"])


def main():
    names = sys.argv[1:] or list(CONFIGS)
    OUTROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names:
        rows = []
        for sc in SCENES:
            print(f">>> {name} / {sc} ...", flush=True)
            r = run(name, CONFIGS[name], sc)
            if r:
                rows.append(r)
                print(f"    {sc}: FPS={r['fps']:.2f} AP25={r['ap25']:.3f} AP50={r['ap50']:.3f}", flush=True)
        results[name] = rows

    print()
    print("=" * 74)
    print(f"{'场景':<12}" + "".join(f"{n:>18}" for n in names))
    print(f"{'':<12}" + "".join(f"{'AP@.25 / AP@.50':>18}" for n in names))
    print("-" * 74)
    for i, sc in enumerate(SCENES):
        line = f"{sc:<12}"
        for n in names:
            r = next((x for x in results[n] if x["scene"] == sc), None)
            line += f"{r['ap25']:>8.3f}/{r['ap50']:<9.3f}" if r else f"{'—':>18}"
        print(line)
    print("-" * 74)
    for stat in ("ap25", "ap50", "fps"):
        line = f"{'宏平均 ' + stat:<12}"
        for n in names:
            vals = [r[stat] for r in results[n]]
            line += f"{sum(vals)/max(len(vals),1):>18.3f}"
        print(line)
    (OUTROOT / "ab8_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
