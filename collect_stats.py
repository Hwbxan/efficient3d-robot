"""汇总 Demo A 最终跑的 8 场景指标，产出 /workspace/demo_a_stats.json。

评测口径：stride=8 密集 GT（每场景 250 帧，覆盖整条 2000 帧轨迹）。
N=10 的检测帧是 0,10,20…，GT 帧是 0,8,16…，只在 ≡0 (mod 40) 处重合，
97.5% 的评测帧是几何传播帧。
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/data/efficient3d_robot")
PY = "/miniconda3/bin/python3"
CLASSES = ("bin,basket,tissue-paper,chair,sofa,stool,armchair,couch,door,"
           "table,desk,desk-organizer,tv-screen,tablet,monitor")
SCENES = [("office0", "office_0", "Office 0"), ("office1", "office_1", "Office 1"),
          ("office2", "office_2", "Office 2"), ("office3", "office_3", "Office 3"),
          ("office4", "office_4", "Office 4"), ("room0", "room_0", "Room 0"),
          ("room1", "room_1", "Room 1"), ("room2", "room_2", "Room 2")]
RUNROOT = ROOT / "outputs/rt8_final"
OUT = ROOT / "demo_a_stats.json"


def main():
    scenes = []
    for s, gtname, label in SCENES:
        out = RUNROOT / s
        lat = json.loads((out / "latency.json").read_text())
        fr = lat.get("frames", [])
        mean = sum(f.get("total_ms", 0.0) for f in fr) / max(len(fr), 1)
        det = [f for f in fr if f.get("mode", "detect") == "detect"]
        pro = [f for f in fr if f.get("mode") == "propagate"]
        dms = sum(f["total_ms"] for f in det) / max(len(det), 1)
        pms = sum(f["total_ms"] for f in pro) / max(len(pro), 1)

        ev_out = out / "gt_eval.json"
        log = out / "gt_eval.log"
        rc = 0
        if not ev_out.is_file():
            cmd = [PY, "-u", "-m", "tools.evaluate_against_gt",
                   "--tracking-json", str(out / "association/tracking.json"),
                   "--segmentation-root", str(out / "segmentation"),
                   "--gt-root", f"outputs/gt8/{gtname}",
                   "--output", str(ev_out), "--gt-classes", CLASSES,
                   "--max-area-fraction", "0.4"]
            with open(log, "w") as fh:
                rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        ap25 = ap50 = None
        if rc == 0:
            ev = json.loads(ev_out.read_text())
            ap25 = ev["detection"]["iou_0.25"]["ap"]
            ap50 = ev["detection"]["iou_0.5"]["ap"]
        else:
            print(f"[WARN] 评测失败 {s}，见 {log}")

        tracks = json.loads((out / "association/tracking.json").read_text())
        scenes.append(dict(name=s, label=label, frames=len(fr), fps=1000.0 / mean,
                           mean_ms=mean, detect_ms=dms, propagate_ms=pms,
                           instances=len(tracks.get("tracks", [])),
                           ap25=ap25, ap50=ap50))
        print(f"{s}: frames={len(fr)} fps={1000.0/mean:.2f} det={dms:.0f} "
              f"prop={pms:.1f} tracks={len(tracks.get('tracks', []))} "
              f"AP25={ap25} AP50={ap50}", flush=True)

    n = len(scenes)
    mean_fps = sum(s["fps"] for s in scenes) / n
    mean_ap25 = sum(s["ap25"] for s in scenes if s["ap25"] is not None) / n
    mean_ap50 = sum(s["ap50"] for s in scenes if s["ap50"] is not None) / n
    total_frames = sum(s["frames"] for s in scenes)
    total_inst = sum(s["instances"] for s in scenes)
    stats = dict(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        scenes=scenes,
        overall=dict(
            fps_mean=mean_fps, ap25_mean=mean_ap25, ap50_mean=mean_ap50,
            detect_ms_mean=sum(s["detect_ms"] for s in scenes) / n,
            propagate_ms_mean=sum(s["propagate_ms"] for s in scenes) / n,
            kpis=[
                [f"{total_frames:,}", "连续处理帧数（8 场景 × 2000）", {}],
                [f"{mean_fps:.1f}", "平均吞吐 FPS", {"good": True}],
                [f"{mean_ap50:.3f}", "宏平均 AP@.50", {"good": True}],
                [f"{mean_ap25:.3f}", "宏平均 AP@.25", {}],
                [f"{total_inst}", "重建 3D 实例总数", {}],
                [f"{sum(s['propagate_ms'] for s in scenes)/n:.1f} ms", "传播帧单帧耗时", {}],
            ],
            notes=("""
<div class="ok">
<b>本次相对上一版的三个实质变化：</b>
<ol>
<li><b>换检测器</b> Grounding DINO tiny → base：单场景 AP@.50 0.231 → 0.272。
因为 N=10 只有 1/10 的帧需要过检测器，base 的额外开销被摊薄，<b>FPS 只掉 1.1</b>。</li>
<li><b>改提示词</b> <code>computer monitor</code> → <code>tv screen</code>：AP@.50 0.272 → 0.312。
逐类别诊断显示旧提示词把 241 次预测里 102 次打在墙上的装饰画上，真正的显示器只命中 1 次。</li>
<li><b>修评测口径</b>：早期 GT 是 stride=10 的 200 帧，与 N=10 的检测帧完全重合，
传播帧一帧都没被评到，导致不同配置 AP 完全相同（假象）。现改用 stride=8 密集 GT。</li>
</ol>
</div>
"""),
        ),
    )
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n宏平均 FPS={mean_fps:.2f} AP@.25={mean_ap25:.3f} AP@.50={mean_ap50:.3f}")
    print(f"写入 {OUT}")


if __name__ == "__main__":
    main()
