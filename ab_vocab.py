"""开放词汇泛化实验：把提示词从 6 个扩到 16 个，看"新场景"能否被覆盖。

动机：8 个场景共 56 个语义类别，而当前只有 6 个提示词，覆盖率极低。
room_1 是卧室（bed / nightstand / blanket / comforter / book …），6 个提示词
一个都覆盖不到，AP@.50 只有 0.144 —— 这是**词汇覆盖问题**，不是感知能力问题。

做法：提示词扩到 16 个高频物体类，同时把 GT 白名单从 15 类扩到
"全部非结构类"（排除 wall/floor/ceiling/blinds/window/switch/vent/pipe/pillar/
panel/undefined/non-plane/other-leaf/anonymize_*），否则新增提示词会被算成 FP。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data/efficient3d_robot")
PY = "/miniconda3/bin/python3"
NFRAMES = 1000
OUTROOT = ROOT / "outputs/abvocab"

# 结构性 / 背景类：任何合理的开放词汇系统都不该被要求检出
STRUCTURAL = {
    "wall", "floor", "ceiling", "undefined", "non-plane", "other-leaf",
    "blinds", "window", "wall-plug", "switch", "vent", "pipe", "pillar",
    "panel", "anonymize_text", "anonymize_picture", "?",
}
OLD15 = ("bin,basket,tissue-paper,chair,sofa,stool,armchair,couch,door,"
         "table,desk,desk-organizer,tv-screen,tablet,monitor")

BASE = "checkpoints/grounding-dino-base"
CONFIGS = {
    # 基线：6 提示词 + 旧 15 类白名单
    "p6_old15": dict(prompts=["tv screen", "chair", "desk", "trash can", "door", "sofa"],
                     box=0.30, classes=OLD15),
    # 16 提示词 + 扩展白名单
    "p16_ext": dict(prompts=["tv screen", "chair", "desk", "table", "trash can", "door",
                             "sofa", "cushion", "pillow", "lamp", "book", "bottle",
                             "vase", "potted plant", "bench", "stool"],
                    box=0.30, classes=None),
}

SCENES = ["room1", "room0", "office0", "room2"]


def scene_gt_name(s):
    return s.replace("office", "office_").replace("room", "room_")


def extended_classes(scene):
    m = json.loads((ROOT / f"outputs/gt8/{scene_gt_name(scene)}/gt_manifest.json").read_text())
    o2c = {int(k): v for k, v in m["object_id_to_class"].items()}
    seen = {o2c.get(o, "?") for f in m["frames"].values() for o in f["visible_objects"]}
    keep = sorted(c for c in seen if c not in STRUCTURAL)
    return ",".join(keep)


def run(name, cfg, scene):
    out = OUTROOT / name / scene
    log = OUTROOT / name / f"{scene}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", "-m", "tools.run_sequence_efficient",
           "--scene-directory", f"datasets/processed/Replica/{scene}",
           "--run-directory", str(out),
           "--frames", *[str(i) for i in range(NFRAMES)],
           "--dino-model", BASE,
           "--classes", *cfg["prompts"],
           "--box-threshold", str(cfg["box"]),
           "--detect-interval", "10", "--propagate-stride", "2",
           "--pixel-stride", "2", "--skip-ply", "--no-preview", "--no-fusion"]
    with open(log, "w") as fh:
        if subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT) != 0:
            print(f"[FAIL run] {name}/{scene}"); return None

    classes = cfg["classes"] or extended_classes(scene)
    ev_out = out / "gt_eval.json"
    cmd2 = [PY, "-u", "-m", "tools.evaluate_against_gt",
            "--tracking-json", str(out / "association/tracking.json"),
            "--segmentation-root", str(out / "segmentation"),
            "--gt-root", f"outputs/gt8/{scene_gt_name(scene)}",
            "--output", str(ev_out), "--gt-classes", classes,
            "--max-area-fraction", "0.4"]
    with open(log, "a") as fh:
        if subprocess.call(cmd2, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT) != 0:
            print(f"[FAIL eval] {name}/{scene}"); return None

    ev = json.loads(ev_out.read_text())
    lat = json.loads((out / "latency.json").read_text())
    fr = lat.get("frames", [])
    mean = sum(f.get("total_ms", 0.0) for f in fr) / max(len(fr), 1)
    tracks = json.loads((out / "association/tracking.json").read_text())
    return dict(scene=scene, fps=1000.0 / mean,
                ap25=ev["detection"]["iou_0.25"]["ap"],
                ap50=ev["detection"]["iou_0.5"]["ap"],
                p50=ev["detection"]["iou_0.5"]["precision"],
                r50=ev["detection"]["iou_0.5"]["recall"],
                gt=ev["detection"]["iou_0.5"]["gt_instances"],
                tracks=len(tracks.get("tracks", [])),
                ncls=len(classes.split(",")))


def main():
    names = sys.argv[1:] or list(CONFIGS)
    OUTROOT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name in names:
        rows = []
        for sc in SCENES:
            print(f">>> {name} / {sc} ...", flush=True)
            r = run(name, CONFIGS[name], sc)
            if r:
                rows.append(r)
                print(f"    {sc}: FPS={r['fps']:.2f} AP25={r['ap25']:.3f} "
                      f"AP50={r['ap50']:.3f} GT={r['gt']} tracks={r['tracks']}", flush=True)
        res[name] = rows

    print()
    print("=" * 88)
    print(f"{'场景':<10}" + "".join(f"{n:>26}" for n in names))
    print(f"{'':<10}" + "".join(f"{'AP50 / P / R / GT数':>26}" for n in names))
    print("-" * 88)
    for sc in SCENES:
        line = f"{sc:<10}"
        for n in names:
            r = next((x for x in res[n] if x["scene"] == sc), None)
            line += (f"{r['ap50']:>7.3f}/{r['p50']:>5.2f}/{r['r50']:>5.2f}/{r['gt']:<6}"
                     if r else f"{'—':>26}")
        print(line)
    print("-" * 88)
    for k in ("ap25", "ap50", "fps", "tracks"):
        line = f"{'平均 ' + k:<10}"
        for n in names:
            v = [r[k] for r in res[n]]
            line += f"{sum(v)/max(len(v),1):>26.3f}"
        print(line)
    (OUTROOT / "abvocab_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
