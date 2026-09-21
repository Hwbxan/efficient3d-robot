"""汇总最终版（rt8_v10）8 场景指标，产出 /data/efficient3d_robot/demo_a_stats_v10.json。

AP 用 Replica 网格协议（对齐 OVI-MAP / OVO-SLAM），四种口径全算：
labeled/agnostic × all/seen，另加 --strict 的保守下界。
性能直接用每个场景 latency.json 的 summary。
"""
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/data/efficient3d_robot")
PY = "/miniconda3/bin/python3"
RUNROOT = ROOT / "outputs/rt8_v10"
OUT = ROOT / "demo_a_stats_v10.json"
FUSION = "fusion_attempt_01"

SCENES = [("office0", "office_0", "Office 0"), ("office1", "office_1", "Office 1"),
          ("office2", "office_2", "Office 2"), ("office3", "office_3", "Office 3"),
          ("office4", "office_4", "Office 4"), ("room0", "room_0", "Room 0"),
          ("room1", "room_1", "Room 1"), ("room2", "room_2", "Room 2")]

AP_RE = re.compile(r"AP25\s+([\d.]+)\s+AP50\s+([\d.]+)\s+AP75\s+([\d.]+)")


def eval_scene(gt_name, mode, subset, strict=False):
    cmd = [PY, "eval_mesh_protocol.py",
           "--run-root", str(RUNROOT.relative_to(ROOT)),
           "--dist", "0.05", "--mode", mode, "--subset", subset,
           "--scenes", gt_name]
    if strict:
        cmd.append("--strict")
    text = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                          text=True).stdout
    m = AP_RE.search(text)
    if not m:
        return None
    return [float(m.group(1)) / 100.0, float(m.group(2)) / 100.0,
            float(m.group(3)) / 100.0]


def main():
    import os
    os.environ["FUSION_DIR"] = FUSION
    scenes = []
    for s, gt, label in SCENES:
        out = RUNROOT / s
        lat = json.loads((out / "latency.json").read_text())
        sm = lat["summary"]
        imap = json.loads((out / FUSION / "instance_map.json").read_text())
        insts = imap.get("instances", [])
        rec = {
            "name": s,
            "label": label,
            "frames": sm.get("measured_frames", 2000),
            "fps": sm.get("fps_mean", 0.0),
            "mean_ms": sm.get("mean_total_ms", 0.0),
            "p95_ms": sm.get("p95_total_ms", 0.0),
            "detect_ms": sm.get("detect_frame_mean_ms", 0.0),
            "propagate_ms": sm.get("propagate_frame_mean_ms", 0.0),
            "detect_frames": sm.get("detect_frames", 0),
            "instances": len(insts),
        }
        for mode in ("labeled", "agnostic"):
            for subset in ("all", "seen"):
                v = eval_scene(gt, mode, subset)
                if v:
                    rec[f"ap_{mode}_{subset}"] = v
        v = eval_scene(gt, "labeled", "all", strict=True)
        if v:
            rec["ap_labeled_all_strict"] = v
        print(s, "fps=%.2f" % rec["fps"], "inst=%d" % rec["instances"],
              rec.get("ap_labeled_all"), rec.get("ap_agnostic_all"))
        scenes.append(rec)

    def mean(key, i=0):
        vals = [r[key][i] for r in scenes if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "fps_mean": sum(r["fps"] for r in scenes) / len(scenes),
        "fps_min": min(r["fps"] for r in scenes),
        "instances": sum(r["instances"] for r in scenes),
        "ap_labeled_all": [mean("ap_labeled_all", i) for i in range(3)],
        "ap_agnostic_all": [mean("ap_agnostic_all", i) for i in range(3)],
        "ap_labeled_seen": [mean("ap_labeled_seen", i) for i in range(3)],
        "ap_agnostic_seen": [mean("ap_agnostic_seen", i) for i in range(3)],
        "ap_labeled_all_strict": [mean("ap_labeled_all_strict", i)
                                  for i in range(3)],
    }
    OUT.write_text(json.dumps({
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run": "outputs/rt8_v10",
        "protocol": "Replica mesh protocol, kNN 5cm, COCO 101-point AP",
        "scenes": scenes,
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入", OUT)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
