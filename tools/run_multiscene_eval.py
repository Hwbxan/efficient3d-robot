"""多场景端到端评测编排器：渲染 → GT → 在线推理 → 评估 → 汇总。

把单个场景的完整评测串成一条命令，18 个场景批量跑，结果汇总到
`outputs/multiscene_eval/summary.json`。每一步都有"已完成"标记，
中断后重跑会跳过已完成的步骤。

三个阶段用不同的解释器：
- 渲染用 habitat 环境（/miniconda3/envs/e3d-habitat/bin/python）；
- GT 生成 / 推理 / 评估用主环境（torch + CUDA）。

用法（项目根目录）：
python -m tools.run_multiscene_eval --scenes office_1 room_0
python -m tools.run_multiscene_eval --all --frame-stride 10

产出：
    outputs/multiscene_eval/<scene>/...        每场景运行目录
    outputs/multiscene_eval/summary.json       汇总指标
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HABITAT_PYTHON = "/miniconda3/envs/e3d-habitat/bin/python"
MAIN_PYTHON = sys.executable

ALL_SCENES = [
    "apartment_0", "apartment_1", "apartment_2",
    "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
    "frl_apartment_3", "frl_apartment_4", "frl_apartment_5",
    "hotel_0",
    "office_0", "office_1", "office_2", "office_3", "office_4",
    "room_0", "room_1", "room_2",
]

# GT 白名单要覆盖「提示词真正能检出的那几类」，而不只是 6 个常见词。
# Replica 的办公室场景把桌子标成 desk、显示器标成 monitor，早期白名单只认
# table / tv-screen，于是这些物体既进不了 GT，检测器检出后还被算成 FP。
GT_CLASSES = [
    "bin", "basket", "tissue-paper",
    "chair", "sofa", "stool", "armchair", "couch",
    "door",
    "table", "desk", "desk-organizer",
    "tv-screen", "tablet", "monitor",
]
DEFAULT_PROMPTS = ["computer monitor", "chair", "desk", "trash can", "door", "sofa"]


def run(command, log_path, description):
    """执行命令，输出同时写日志；失败时抛出带日志路径的异常。"""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  → {description}")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start
    if process.returncode != 0:
        raise RuntimeError(
            f"{description} 失败（退出码 {process.returncode}），日志：{log_path}"
        )
    print(f"    完成，用时 {elapsed:.1f}s")
    return elapsed


def replica_sequence_name(scene: str) -> str:
    """Replica 官方场景名 → Nice-SLAM RGB-D 序列目录名。

    `datasets/raw/replica_v1/` 用官方命名 `office_1`（带下划线），
    而 Nice-SLAM 发布的 RGB-D 扫描包里是 `office1`（无下划线）。
    """

    return scene.replace("_", "")


def available_frame_count(scene_directory: Path) -> int:
    """读取场景实际渲染出的帧数。

    habitat 在走到无可通行位置时可能提前结束，因此实际帧数常常少于
    --render-frames；评测范围必须以真实帧数为准。
    """

    traj = scene_directory / "traj.txt"
    if traj.exists():
        with traj.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    return len(list((scene_directory / "results").glob("frame*.jpg")))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="跑全部 18 个 Replica 场景")
    parser.add_argument("--real-only", action="store_true",
                        help="只评测有现成 RGB-D 序列的场景（Nice-SLAM 发布的 8 个标准场景），"
                             "其余场景既不渲染也不评测")
    parser.add_argument("--output-root", default="outputs/multiscene_eval")
    parser.add_argument("--real-root", default="datasets/processed/Replica",
                        help="现成 RGB-D 序列根目录（Nice-SLAM 发布，目录名无下划线）")
    parser.add_argument("--render-root", default="datasets/processed/Replica_rendered")
    parser.add_argument("--gt-root", default="outputs/gt")
    parser.add_argument("--raw-root", default="datasets/raw/replica_v1")
    parser.add_argument("--render-frames", type=int, default=300)
    parser.add_argument("--render-step", type=float, default=0.25)
    parser.add_argument("--render-seed", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--end-frame", type=int, default=None,
                        help="评测最后一帧（默认渲染帧数-1）")
    parser.add_argument("--dino-model", default="checkpoints/grounding-dino-base")
    parser.add_argument("--sam-model", default="checkpoints/sam2.1-hiera-tiny")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--gt-classes", nargs="+", default=GT_CLASSES,
                        help="GT 类别白名单（默认覆盖扩展后的家具类；传 6 类旧值可复现早期口径）")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-box-area-fraction", type=float, default=0.40)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--merge-coverage", type=float, default=0.0,
                        help="单帧内同标签观测体素覆盖率达到该值即合并；0 表示不合并（默认）")
    parser.add_argument("--merge-max-center-distance", type=float, default=0.80)
    parser.add_argument("--no-fp16", action="store_true",
                        help="2D 前端用 fp32（默认 fp16，更快且精度基本无损）")
    parser.add_argument("--max-area-fraction", type=float, default=0.40,
                        help="评测时丢弃框面积占比超过该值的检测")
    parser.add_argument("--skip-render", action="store_true",
                        help="跳过渲染（场景目录已存在时）")
    parser.add_argument("--skip-run", action="store_true",
                        help="跳过推理（只重算评估）")
    args = parser.parse_args()

    if args.all:
        scenes = ALL_SCENES
    elif args.scenes:
        scenes = args.scenes
    else:
        parser.error("需要 --scenes 或 --all")

    if args.real_only:
        real_root = Path(args.real_root)
        scenes = [
            s for s in scenes
            if (real_root / replica_sequence_name(s) / "traj.txt").exists()
        ]
        if not scenes:
            parser.error(f"--real-only：{real_root} 下没有找到任何现成序列")
        print(f"--real-only：评测 {len(scenes)} 个有现成序列的场景 {scenes}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) \
        if summary_path.exists() else {"scenes": {}, "config": {}}

    summary["config"] = {
        "real_root": args.real_root,
        "render_frames": args.render_frames,
        "render_step": args.render_step,
        "render_seed": args.render_seed,
        "frame_stride": args.frame_stride,
        "dino_model": args.dino_model,
        "sam_model": args.sam_model,
        "classes": args.classes,
        "box_threshold": args.box_threshold,
        "gt_classes": args.gt_classes,
        "max_area_fraction": args.max_area_fraction,
    }

    for scene in scenes:
        print(f"\n===== 场景 {scene} =====")
        # 现成的 Nice-SLAM RGB-D 扫描序列优先：它是社区标准数据，
        # 轨迹本身就覆盖家具，比我们自己用 habitat 沿 navmesh 渲染的
        # 序列可信得多，也省掉整段渲染开销。只有没有现成序列的场景
        # 才退回 habitat 渲染。
        sequence_name = replica_sequence_name(scene)
        real_directory = Path(args.real_root) / sequence_name
        rendered_directory = Path(args.render_root) / scene

        if (real_directory / "traj.txt").exists():
            scene_directory = real_directory
            scene_source = "nice-slam"
            print(f"  → 使用现成 RGB-D 序列 {scene_directory}")
        else:
            scene_directory = rendered_directory
            scene_source = "habitat-render"
            print(f"  → 无现成序列，回退到 habitat 渲染 {scene_directory}")

        gt_directory = Path(args.gt_root) / scene
        run_directory = output_root / scene
        timings = {}

        # ---- 1. 渲染（仅无现成序列时）----
        if not args.skip_render and not (scene_directory / "traj.txt").exists():
            timings["render_s"] = run(
                [HABITAT_PYTHON, "-m", "tools.render_replica_sequences",
                 "--scene", scene,
                 "--raw-root", args.raw_root,
                 "--output-root", args.render_root,
                 "--frames", str(args.render_frames),
                 "--step", str(args.render_step),
                 "--seed", str(args.render_seed)],
                run_directory / "logs" / "render.log",
                f"渲染 {scene} 的 RGB-D 序列",
            )
        elif (scene_directory / "traj.txt").exists():
            print("  → 渲染已存在，跳过")

        if not (scene_directory / "traj.txt").exists():
            print(f"  ! 跳过 {scene}：场景目录不存在 {scene_directory}")
            continue

        total_frames = available_frame_count(scene_directory)
        end_frame = args.end_frame if args.end_frame is not None else total_frames - 1
        end_frame = min(end_frame, total_frames - 1)
        eval_frames = list(range(0, end_frame + 1, args.frame_stride))
        print(f"  → 可用帧 {total_frames}，评测 {len(eval_frames)} 帧（步长 {args.frame_stride}）")

        # ---- 2. GT ----
        if not (gt_directory / "gt_manifest.json").exists():
            timings["gt_s"] = run(
                [MAIN_PYTHON, "-m", "tools.generate_gt_instance_masks",
                 "--scene-directory", str(scene_directory),
                 "--replica-scene", scene,
                 "--replica-root", args.raw_root,
                 "--output-directory", str(gt_directory),
                 "--start-frame", "0",
                 "--end-frame", str(end_frame),
                 "--frame-stride", str(args.frame_stride),
                 "--preview-count", "2"],
                run_directory / "logs" / "gt.log",
                f"生成 {scene} 的 GT 实例掩码",
            )
        else:
            print("  → GT 已存在，跳过")

        # ---- 3. 在线推理 ----
        if not args.skip_run and not (run_directory / "association" / "tracking.json").exists():
            timings["run_s"] = run(
                [MAIN_PYTHON, "tools/run_sequence_efficient.py",
                 "--scene-directory", str(scene_directory),
                 "--dino-model", args.dino_model,
                 "--sam-model", args.sam_model,
                 "--run-directory", str(run_directory),
                 "--frames", *[str(f) for f in eval_frames],
                 "--classes", *args.classes,
                 "--box-threshold", str(args.box_threshold),
                 "--text-threshold", str(args.text_threshold),
                 "--max-box-area-fraction", str(args.max_box_area_fraction),
                 "--voxel-size", str(args.voxel_size),
                 "--merge-coverage", str(args.merge_coverage),
                 "--merge-max-center-distance",
                 str(args.merge_max_center_distance)]
                + (["--no-fp16"] if args.no_fp16 else []),
                run_directory / "logs" / "run.log",
                f"在线推理 {scene}（{len(eval_frames)} 帧）",
            )
        else:
            print("  → 推理结果已存在，跳过")

        # ---- 4. 评估 ----
        evaluation_path = run_directory / "gt_eval.json"
        timings["eval_s"] = run(
            [MAIN_PYTHON, "-m", "tools.evaluate_against_gt",
             "--tracking-json", str(run_directory / "association" / "tracking.json"),
             "--segmentation-root", str(run_directory / "segmentation"),
             "--gt-root", str(gt_directory),
             "--output", str(evaluation_path),
             "--gt-classes", ",".join(args.gt_classes),
             "--max-area-fraction", str(args.max_area_fraction)],
            run_directory / "logs" / "eval.log",
            f"评估 {scene}",
        )

        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        manifest = json.loads((gt_directory / "gt_manifest.json").read_text(encoding="utf-8"))
        alignments = [
            frame["median_abs_diff_m"] for frame in manifest["frames"].values()
            if frame.get("median_abs_diff_m") is not None
        ]

        summary["scenes"][scene] = {
            "sequence": sequence_name,
            "source": scene_source,
            "frames": evaluation["evaluated_frames"],
            "gt_instances": evaluation["detection"]["iou_0.25"]["gt_instances"],
            "detection": evaluation["detection"],
            "association": evaluation["association"],
            "label_consistency": evaluation["label_consistency"],
            "unassigned_observations": evaluation["unassigned_observations"],
            "gt_depth_alignment_max_m": max(alignments) if alignments else None,
            "gt_visible_objects": manifest["visible_object_count"],
            "timings_s": timings,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        detection = evaluation["detection"]
        print(
            f"  AP@0.25={detection['iou_0.25']['ap']:.4f}  "
            f"AP@0.50={detection['iou_0.5']['ap']:.4f}  "
            f"单轨率={evaluation['association']['single_track_ratio']:.3f}  "
            f"纯轨率={evaluation['association']['pure_track_ratio']:.3f}"
        )

    print(f"\n汇总：{summary_path}")
    print_summary(summary)


def print_summary(summary):
    scenes = summary["scenes"]
    if not scenes:
        return
    print(f"\n{'场景':<18}{'帧':>5}{'GT':>6}{'AP@.25':>9}{'AP@.5':>9}"
          f"{'单轨率':>8}{'纯轨率':>8}{'碎片':>7}{'标签':>7}")
    rows = []
    for name, item in scenes.items():
        detection = item["detection"]
        association = item["association"]
        rows.append((
            name,
            item["frames"],
            detection["iou_0.25"]["gt_instances"],
            detection["iou_0.25"]["ap"],
            detection["iou_0.5"]["ap"],
            association["single_track_ratio"],
            association["pure_track_ratio"],
            association["fragmentation_mean"],
            item["label_consistency"]["ratio"],
        ))
    for row in rows:
        print(f"{row[0]:<18}{row[1]:>5}{row[2]:>6}{row[3]:>9.4f}{row[4]:>9.4f}"
              f"{row[5]:>8.3f}{row[6]:>8.3f}{row[7]:>7.3f}{row[8]:>7.3f}")

    def mean(index):
        values = [row[index] for row in rows if row[index] is not None]
        return sum(values) / len(values) if values else 0.0

    print(f"{'平均':<18}{'':>5}{'':>6}{mean(3):>9.4f}{mean(4):>9.4f}"
          f"{mean(5):>8.3f}{mean(6):>8.3f}{mean(7):>7.3f}{mean(8):>7.3f}")


if __name__ == "__main__":
    main()
