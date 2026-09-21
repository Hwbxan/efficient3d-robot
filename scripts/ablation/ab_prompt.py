"""提示词覆盖度实验：把 6 个提示词扩到覆盖全部 15 类 GT 白名单。

诊断依据（diag8/diag9）：八场景 10503 个白名单 GT 实例里，25.5% 的最佳 IoU = 0
（完全没被检出），且 IoU 分布是双峰的——要么 ≥0.8 要么 0，几乎没有"检到了但
掩码很烂"的中间态。所以精度瓶颈是**检测覆盖**，不是掩码质量。

完全漏检的绝对数量：
    table 689 · chair 371 · tablet 330 · tissue-paper 270 · desk-organizer 248 ·
    stool 180 · sofa 174 · bin 146 · basket 98 · door 81 · tv-screen 53

其中 tablet / tissue-paper / desk-organizer / stool / basket 根本不在 6 个提示词里，
检测帧上 R@.25 也只有 0.01~0.20 —— 纯粹的词汇缺口。

配置：
  A_P6     基线：6 提示词，不合并
  B_P13    13 提示词（覆盖全部 15 类白名单），不合并
  C_P13M   13 提示词 + 单帧内合并（--merge-coverage 0.5）
  D_P13X   13 提示词 + 单帧内合并 + 跨标签合并（同义提示词去重）
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data/efficient3d_robot")
PY = "/miniconda3/bin/python3"
CLASSES = ("bin,basket,tissue-paper,chair,sofa,stool,armchair,couch,door,"
           "table,desk,desk-organizer,tv-screen,tablet,monitor")
SCENES = os.environ.get("ABSCENES", "office0,room0,room1,room2").split(",")
NFRAMES = 1000
OUTROOT = ROOT / "outputs/abprompt"

BASE = "checkpoints/grounding-dino-base"
P6 = ["tv screen", "chair", "desk", "trash can", "door", "sofa"]
P13 = ["tv screen", "monitor", "tablet", "chair", "stool", "sofa",
       "desk", "table", "desk organizer", "trash can", "basket",
       "tissue box", "door"]

CONFIGS = {
    "A_P6":    dict(prompts=P6, box=0.30, merge=0.0, cross=False),
    "B_P13":   dict(prompts=P13, box=0.30, merge=0.0, cross=False),
    "C_P13M":  dict(prompts=P13, box=0.30, merge=0.5, cross=False),
    "D_P13X":  dict(prompts=P13, box=0.30, merge=0.5, cross=True),
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
           "--dino-model", BASE,
           "--classes", *cfg["prompts"],
           "--box-threshold", str(cfg["box"]),
           "--detect-interval", "10", "--propagate-stride", "2",
           "--pixel-stride", "2",
           "--merge-coverage", str(cfg["merge"])]
    if cfg["cross"]:
        cmd.append("--cross-label-merge")
    cmd += ["--skip-ply", "--no-preview", "--no-fusion"]
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
    d50 = ev["detection"]["iou_0.5"]
    return dict(scene=scene, fps=1000.0 / mean, mean=mean,
                ap25=ev["detection"]["iou_0.25"]["ap"],
                ap50=d50["ap"],
                p50=d50.get("precision"), r50=d50.get("recall"),
                gt=d50.get("gt_instances"), pred=d50.get("predictions"),
                tracks=len(json.loads(
                    (out / "association/tracking.json").read_text()).get("tracks", [])))


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
                print(f"    {sc}: FPS={r['fps']:.2f} AP25={r['ap25']:.3f} "
                      f"AP50={r['ap50']:.3f} P={r['p50']} R={r['r50']} "
                      f"tracks={r['tracks']}", flush=True)
        results[name] = rows

    print()
    print("=" * 84)
    print(f"{'场景':<10}" + "".join(f"{n:>18}" for n in names))
    print(f"{'':<10}" + "".join(f"{'AP@.25 / AP@.50':>18}" for n in names))
    print("-" * 84)
    for sc in SCENES:
        line = f"{sc:<10}"
        for n in names:
            r = next((x for x in results[n] if x["scene"] == sc), None)
            line += f"{r['ap25']:>8.3f}/{r['ap50']:<9.3f}" if r else f"{'—':>18}"
        print(line)
    print("-" * 84)
    for stat in ("ap25", "ap50", "fps"):
        line = f"{'宏平均 ' + stat:<10}"
        for n in names:
            vals = [r[stat] for r in results[n]]
            line += f"{sum(vals)/max(len(vals),1):>18.3f}"
        print(line)
    print("-" * 84)
    for stat in ("p50", "r50", "pred"):
        line = f"{'宏平均 ' + stat:<10}"
        for n in names:
            vals = [r[stat] for r in results[n] if r[stat] is not None]
            line += f"{sum(vals)/max(len(vals),1):>18.3f}" if vals else f"{'—':>18}"
        print(line)
    (OUTROOT / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
