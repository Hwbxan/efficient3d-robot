"""回放 31 个采样帧，复用六帧基线，并在独立目录重新关联和融合。

仅使用标准库编排已有命令；不修改模型或关联算法。
默认逐帧启动推理脚本，因此本脚本的总耗时不能作为实时性能指标。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_FRAMES = {0, 10, 20, 30, 40, 50}


def absolute_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_records(path, asset_field):
    """检查局部 ID 和所有引用文件；空列表允许表示该帧没有实例。"""
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"实例 JSON 应为列表：{path}")
    ids = [item["local_instance_id"] for item in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"局部 ID 重复：{path}")
    for item in records:
        asset = absolute_path(item[asset_field])
        if not asset.is_file() or asset.stat().st_size == 0:
            raise FileNotFoundError(f"实例引用文件缺失或为空：{asset}")
    return records


def copy_records(source, target, asset_field, asset_directory):
    """复制缓存及引用文件，并将新 JSON 的路径指向实验目录内的副本。"""
    records = read_records(source, asset_field)
    asset_directory.mkdir(parents=True, exist_ok=True)
    for item in records:
        original = absolute_path(item[asset_field])
        destination = asset_directory / f"{item['local_instance_id']:02d}_{original.name}"
        shutil.copy2(original, destination)
        item[asset_field] = str(destination.resolve())
    save_json(target, records)


def run_module(module, arguments, log_path):
    command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n运行 {module}；日志：{log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        if return_code != 0:
            raise RuntimeError(f"{module} 运行失败，退出码 {return_code}；请查看 {log_path}")


def validate_frame(segmentation_json, instance_json):
    masks = read_records(segmentation_json, "mask_path")
    points = read_records(instance_json, "point_cloud_path")
    labels = {item["local_instance_id"]: item["label"] for item in masks}
    for item in points:
        if labels.get(item["local_instance_id"]) != item["label"]:
            raise ValueError(f"2D / 3D 局部 ID 或标签不一致：{instance_json}")


def prepare_frame(args, frame, run_directory):
    name = f"frame_{frame:06d}"
    segmentation = run_directory / "segmentation"
    instances = run_directory / "instances_3d"
    segmentation_json = segmentation / f"{name}_instances.json"
    instance_json = instances / name / "instances_3d.json"
    progress = run_directory / "progress"
    segmentation_done = progress / f"{name}_segmentation.json"
    lifting_done = progress / f"{name}_lifting.json"
    cached_masks = ROOT / "outputs/perception/grounded_sam2" / f"{name}_instances.json"
    cached_points = ROOT / "outputs/instances_3d" / name / "instances_3d.json"
    reuse = args.reuse_baseline and frame in BASELINE_FRAMES

    if segmentation_done.is_file():
        read_records(segmentation_json, "mask_path")
        print("  已完成分割，跳过", flush=True)
    else:
        if reuse and cached_masks.is_file() and cached_points.is_file():
            validate_frame(cached_masks, cached_points)
            copy_records(cached_masks, segmentation_json, "mask_path", segmentation / f"{name}_masks")
            source = str(cached_masks)
            print("  已复制六帧基线的掩码缓存", flush=True)
        else:
            run_module("tools.inspect_grounded_sam2", [
                "--scene-directory", args.scene_directory, "--frame-index", frame,
                "--dino-model", args.dino_model, "--sam-model", args.sam_model,
                "--box-threshold", args.box_threshold, "--text-threshold", args.text_threshold,
                "--max-box-area-fraction", args.max_box_area_fraction,
                "--output-directory", segmentation, "--classes", *args.classes,
            ], run_directory / "logs" / f"{name}_segmentation.log")
            source = "new_inference"
        read_records(segmentation_json, "mask_path")
        save_json(segmentation_done, {"source": source})

    if lifting_done.is_file():
        validate_frame(segmentation_json, instance_json)
        print("  已完成 3D 提升，跳过", flush=True)
    else:
        source = load_json(segmentation_done)["source"]
        if source == str(cached_masks):
            copy_records(cached_points, instance_json, "point_cloud_path", instances / name / "individual")
            lifting_source = str(cached_points)
            print("  已复制六帧基线的世界坐标点云", flush=True)
        else:
            lifting_args = [
                "--scene-directory", args.scene_directory, "--frame-index", frame,
                "--instance-json", segmentation_json, "--pixel-stride", args.pixel_stride,
                "--erosion-iterations", args.erosion_iterations, "--output-directory", instances,
            ]
            if args.encoder_weights is not None:
                lifting_args += ["--encoder-weights", args.encoder_weights]
            run_module("tools.inspect_3d_instances", lifting_args,
                       run_directory / "logs" / f"{name}_lifting.log")
            lifting_source = "new_lifting"
        validate_frame(segmentation_json, instance_json)
        save_json(lifting_done, {"source": lifting_source})


def finish_sequence(args, run_directory):
    instances = run_directory / "instances_3d"
    tracking_path = run_directory / "association/tracking.json"
    tracking_done = run_directory / "progress/tracking.json"
    if not tracking_done.is_file():
        tracking_path.parent.mkdir(parents=True, exist_ok=True)
        run_module("tools.inspect_instance_tracking", [
            "--instances-root", instances, "--frames", *args.frames, "--output", tracking_path,
        ], run_directory / "logs/tracking.log")
        tracking = load_json(tracking_path)
        if [item["frame_index"] for item in tracking["frames"]] != args.frames:
            raise ValueError("关联输出的帧列表与本次实验不一致")
        save_json(tracking_done, {"path": str(tracking_path)})

    preview_directory = run_directory / "tracking_preview"
    preview_done = run_directory / "progress/preview.json"
    if not preview_done.is_file():
        run_module("tools.preview_instance_tracking", [
            "--scene-directory", args.scene_directory, "--tracking-json", tracking_path,
            "--mask-directory", run_directory / "segmentation", "--output-directory", preview_directory,
        ], run_directory / "logs/preview.log")
        if not (preview_directory / "tracking_contact_sheet.png").is_file():
            raise FileNotFoundError("未找到关联预览拼图")
        save_json(preview_done, {"path": str(preview_directory)})

    fusion_done = run_directory / "progress/fusion.json"
    if not fusion_done.is_file():
        # 融合脚本拒绝覆盖目录；若上次中断，保留失败输出并使用新尝试目录。
        attempt = 1
        while (run_directory / f"fusion_attempt_{attempt:02d}").exists():
            attempt += 1
        fusion_directory = run_directory / f"fusion_attempt_{attempt:02d}"
        run_module("tools.replay_instance_fusion", [
            "--tracking-json", tracking_path, "--instances-root", instances,
            "--voxel-size", args.voxel_size, "--output-directory", fusion_directory,
        ], run_directory / "logs/fusion.log")
        report = load_json(fusion_directory / "instance_map.json")
        if report["processed_frames"] != args.frames:
            raise ValueError("融合输出的帧列表与本次实验不一致")
        save_json(fusion_done, {"path": str(fusion_directory)})

    fusion_directory = Path(load_json(fusion_done)["path"])
    tracking = load_json(tracking_path)
    decisions = {}
    for frame in tracking["frames"]:
        for association in frame["associations"]:
            decision = association["decision"]
            decisions[decision] = decisions.get(decision, 0) + 1
    summary = {
        "frames": args.frames, "association_decision_counts": decisions,
        "tracking_path": str(tracking_path), "fusion_directory": str(fusion_directory),
        "tracking_preview": str(preview_directory / "tracking_contact_sheet.png"),
        "note": "决策计数不是关联准确率；总运行时间不是实时推理性能。",
    }
    save_json(run_directory / "summary.json", summary)
    print("\n序列处理完成：", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-directory", default="datasets/processed/Replica/office0")
    parser.add_argument("--dino-model", default="checkpoints/grounding-dino-tiny")
    parser.add_argument("--sam-model", default="checkpoints/sam2.1-hiera-tiny")
    parser.add_argument("--run-directory", default="outputs/experiments/office0_0000_0300_v1")
    parser.add_argument("--frames", type=int, nargs="+", default=list(range(0, 301, 10)))
    parser.add_argument("--classes", nargs="+", default=["computer monitor", "chair", "desk", "trash can", "door"])
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--max-box-area-fraction", type=float, default=0.40,
        help="丢弃面积占比超过该值的检测框（针对背景块误检）；1.0 关闭",
    )
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--erosion-iterations", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--reuse-baseline", action="store_true", help="复用旧目录中 0、10、20、30、40、50 帧的结果")
    parser.add_argument(
        "--encoder-weights",
        type=Path,
        default=None,
        help="Stage-5 点式 encoder 权重路径（提供则在 3D 提升时提取 shape_embedding）",
    )
    args = parser.parse_args()

    if args.frames[0] < 0 or any(b <= a for a, b in zip(args.frames, args.frames[1:])):
        parser.error("帧编号必须非负且严格递增")
    if args.pixel_stride < 1 or args.erosion_iterations < 0 or not 0 < args.voxel_size < float("inf"):
        parser.error("步长、腐蚀次数或体素大小无效")
    for name in ["scene_directory", "dino_model", "sam_model"]:
        value = absolute_path(getattr(args, name))
        if not value.is_dir():
            parser.error(f"目录不存在：{value}；请用对应参数指定实际路径")
        setattr(args, name, str(value))

    run_directory = absolute_path(args.run_directory)
    for protected in [absolute_path(args.scene_directory), ROOT / "outputs/perception", ROOT / "outputs/instances_3d"]:
        if run_directory == protected or protected in run_directory.parents or run_directory in protected.parents:
            parser.error("实验目录必须与原始数据及旧缓存目录隔离")
    for frame in args.frames:
        results = Path(args.scene_directory) / "results"
        for filename in [f"frame{frame:06d}.jpg", f"depth{frame:06d}.png"]:
            if not (results / filename).is_file():
                parser.error(f"缺少输入：{results / filename}")

    config = vars(args).copy()
    config["run_directory"] = str(run_directory)
    # Path 对象不可 JSON 序列化
    if config.get("encoder_weights") is not None:
        config["encoder_weights"] = str(config["encoder_weights"])
    config["cache_provenance_note"] = "旧缓存未记录完整生成配置，复用时由用户确认参数一致。"
    config_path = run_directory / "run_config.json"
    if config_path.is_file():
        if load_json(config_path) != config:
            parser.error("本次参数与该目录的 run_config.json 不一致，请使用新实验目录")
    else:
        if run_directory.exists() and any(run_directory.iterdir()):
            parser.error("实验目录非空且无配置记录，请使用新目录")
        save_json(config_path, config)

    if args.reuse_baseline:
        print("注意：旧六帧缓存的生成参数无法自动核实；本次记录其来源，不把缓存当作精度真值。", flush=True)
    for index, frame in enumerate(args.frames, start=1):
        print(f"\n[{index}/{len(args.frames)}] 准备 Frame {frame:06d}", flush=True)
        prepare_frame(args, frame, run_directory)
    finish_sequence(args, run_directory)


if __name__ == "__main__":
    main()
