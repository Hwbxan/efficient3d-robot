"""冻结 V2，因果回放 0～600 帧，只统计未调参的 310～600 帧。

前 31 帧复制旧缓存，后 30 帧沿用旧配置推理。支持原命令重跑续接；
所有新结果写入独立目录，不修改旧实验或关联算法。
"""

import argparse
import hashlib
import math
from pathlib import Path

from tools import run_instance_sequence as pipeline
from tools.summarize_heldout import make_heldout_report, make_manual_review


WARMUP = list(range(0, 301, 10))
EVALUATION = list(range(310, 601, 10))
FRAMES = WARMUP + EVALUATION
TRACKER_FILES = [
    "src/mapping/geometric_instance_tracker.py",
    "src/mapping/surface_instance_tracker.py",
    "tools/run_surface_tracking.py",
]
PROCESSING_KEYS = [
    "scene_directory", "dino_model", "sam_model", "classes", "box_threshold",
    "text_threshold", "pixel_stride", "erosion_iterations", "voxel_size",
]


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require_frames(tracking, expected):
    if [frame["frame_index"] for frame in tracking["frames"]] != expected:
        raise ValueError("关联文件的帧列表与实验协议不一致")


def verify_prefix(previous, replayed):
    """历史决策必须复现。最终 tracks 已含未来更新，不能用它验证前缀。"""
    require_frames(previous, WARMUP)
    require_frames(replayed, FRAMES)
    for old_frame, new_frame in zip(previous["frames"], replayed["frames"]):
        old_items, new_items = old_frame["associations"], new_frame["associations"]
        if len(old_items) != len(new_items):
            raise ValueError("历史复现失败：观测数量变化")
        for old, new in zip(old_items, new_items):
            for key in ["frame_index", "local_instance_id", "raw_label", "global_id", "decision"]:
                if old[key] != new[key]:
                    raise ValueError(f"历史复现失败：Frame {old['frame_index']}，字段 {key}")
            if "association_source" in old and old["association_source"] != new.get("association_source"):
                raise ValueError("历史复现失败：关联分支变化")
            before, after = old["association_cost"], new["association_cost"]
            if before is None or after is None:
                equal = before is after
            else:
                equal = math.isclose(before, after, rel_tol=1e-7, abs_tol=1e-7)
            if not equal:
                raise ValueError("历史复现失败：匹配代价变化")


def preflight(source_baseline, source_v2, output):
    config = pipeline.load_json(source_baseline / "run_config.json")
    comparison_path = source_v2 / "comparison.json"
    comparison = pipeline.load_json(comparison_path)
    original_tracking = source_baseline / "association/tracking.json"
    v2_tracking = source_v2 / "tracking.json"
    for path in [original_tracking, v2_tracking]:
        require_frames(pipeline.load_json(path), WARMUP)
    if not comparison["baseline_reproduced"] or comparison["processed_frames"] != WARMUP:
        raise ValueError("指定的 V2 不是已成功完成的 31 帧实验")

    recorded = {pipeline.absolute_path(key): value for key, value in comparison["source_sha256"].items()}
    for path in [original_tracking, *map(pipeline.absolute_path, TRACKER_FILES)]:
        if path not in recorded or digest(path) != recorded[path]:
            raise ValueError(f"冻结校验失败，代码或基线已变化：{path}。不要自动更新哈希绕过检查。")

    settings = {key: config[key] for key in PROCESSING_KEYS}
    for key in ["scene_directory", "dino_model", "sam_model"]:
        path = pipeline.absolute_path(settings[key])
        if not path.is_dir():
            raise FileNotFoundError(path)
        settings[key] = str(path)
    protected = [source_baseline, source_v2, pipeline.ROOT / "datasets", pipeline.ROOT / "checkpoints"]
    protected += [Path(settings[key]) for key in ["scene_directory", "dino_model", "sam_model"]]
    for path in protected:
        if output == path or path in output.parents or output in path.parents:
            raise ValueError("新实验目录必须与原实验、模型及数据目录隔离")
    scene = Path(settings["scene_directory"])
    if not (scene / "traj.txt").is_file():
        raise FileNotFoundError(scene / "traj.txt")
    for frame in FRAMES:
        for filename in [f"frame{frame:06d}.jpg", f"depth{frame:06d}.png"]:
            if not (scene / "results" / filename).is_file():
                raise FileNotFoundError(scene / "results" / filename)

    # 锁定本次所用配置、脚本和旧缓存，包括缓存引用的掩码及点云。
    inputs = {original_tracking, v2_tracking, comparison_path, source_baseline / "run_config.json"}
    inputs.update(map(pipeline.absolute_path, TRACKER_FILES))
    inputs.update(map(pipeline.absolute_path, [
        "tools/validate_heldout_sequence.py", "tools/summarize_heldout.py",
        "tools/run_instance_sequence.py", "tools/inspect_grounded_sam2.py",
        "tools/inspect_3d_instances.py", "tools/inspect_instance_tracking.py",
        "tools/audit_instance_tracking.py", "tools/replay_instance_fusion.py",
        "tools/preview_instance_tracking.py",
    ]))
    for frame in WARMUP:
        name = f"frame_{frame:06d}"
        masks = source_baseline / "segmentation" / f"{name}_instances.json"
        points = source_baseline / "instances_3d" / name / "instances_3d.json"
        pipeline.validate_frame(masks, points)
        inputs.update([masks, points])
        for metadata, field in [(masks, "mask_path"), (points, "point_cloud_path")]:
            inputs.update(pipeline.absolute_path(item[field]) for item in pipeline.load_json(metadata))
    return {
        "source_baseline": str(source_baseline), "source_v2": str(source_v2),
        "output_directory": str(output), "settings": settings,
        "warmup_frames": WARMUP, "evaluation_frames": EVALUATION,
        "surface_parameters": comparison["surface_parameters"],
        "input_sha256": {str(path): digest(path) for path in sorted(inputs)},
        "note": "旧六帧缓存的原始生成配置仍沿用上一轮来源声明；模型权重未重新哈希，请勿替换。",
    }


def freeze_output(output, frozen):
    config_path = output / "frozen_config.json"
    if config_path.is_file():
        if pipeline.load_json(config_path) != frozen:
            raise ValueError("续跑输入或配置已变化；保留该实验，不要覆盖冻结记录")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("输出目录非空且无冻结记录，请使用新目录")
        pipeline.save_json(config_path, frozen)


def copy_warmup(source, cache):
    for frame in WARMUP:
        name = f"frame_{frame:06d}"
        masks = cache / "segmentation" / f"{name}_instances.json"
        points = cache / "instances_3d" / name / "instances_3d.json"
        marker = cache / "progress" / f"{name}_warmup.json"
        if not marker.is_file():
            pipeline.copy_records(
                source / "segmentation" / masks.name, masks, "mask_path",
                cache / "segmentation" / f"{name}_masks",
            )
            pipeline.copy_records(
                source / "instances_3d" / name / "instances_3d.json", points,
                "point_cloud_path", points.parent / "individual",
            )
            pipeline.validate_frame(masks, points)
            pipeline.save_json(marker, {"source": str(source), "frame_index": frame})
        pipeline.validate_frame(masks, points)
    print("0～300 帧缓存已复制/核验，不运行模型。", flush=True)


def next_attempt(output, name):
    index = 1
    while (output / f"{name}_attempt_{index:02d}").exists():
        index += 1
    return output / f"{name}_attempt_{index:02d}"


def replay(frozen, output, cache):
    baseline_path = cache / "association/tracking.json"
    baseline_done = output / "progress/baseline.json"
    if not baseline_done.is_file():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.run_module("tools.inspect_instance_tracking", [
            "--instances-root", cache / "instances_3d", "--frames", *FRAMES,
            "--output", baseline_path,
        ], output / "logs/baseline.log")
    baseline = pipeline.load_json(baseline_path)
    verify_prefix(pipeline.load_json(Path(frozen["source_baseline"]) / "association/tracking.json"), baseline)
    pipeline.save_json(baseline_done, {"path": str(baseline_path)})

    v2_done = output / "progress/v2.json"
    if v2_done.is_file():
        v2_dir = Path(pipeline.load_json(v2_done)["path"])
    else:
        v2_dir = next_attempt(output, "v2")
        parameters = frozen["surface_parameters"]
        pipeline.run_module("tools.run_surface_tracking", [
            "--baseline-run", cache, "--output-directory", v2_dir,
            "--surface-distance", parameters["surface_distance"],
            "--minimum-coverage", parameters["minimum_coverage"],
        ], output / "logs/v2.log")
    updated = pipeline.load_json(v2_dir / "tracking.json")
    comparison = pipeline.load_json(v2_dir / "comparison.json")
    verify_prefix(pipeline.load_json(Path(frozen["source_v2"]) / "tracking.json"), updated)
    if comparison["surface_parameters"] != frozen["surface_parameters"]:
        raise ValueError("V2 实际参数与冻结版本不同")
    pipeline.save_json(v2_done, {"path": str(v2_dir)})
    return baseline, updated, comparison, v2_dir


def make_previews(frozen, output, cache, baseline, updated, review):
    selected = set(review["display_frames"])
    for name, tracking in [("baseline", baseline), ("v2", updated)]:
        directory = output / f"review_{name}"
        preview_json = output / f"review_{name}_tracking.json"
        pipeline.save_json(preview_json, {
            "frames": [frame for frame in tracking["frames"] if frame["frame_index"] in selected],
            "tracks": tracking["tracks"],
        })
        marker = output / "progress" / f"preview_{name}.json"
        image = directory / "tracking_contact_sheet.png"
        if not marker.is_file():
            pipeline.run_module("tools.preview_instance_tracking", [
                "--scene-directory", frozen["settings"]["scene_directory"],
                "--tracking-json", preview_json, "--mask-directory", cache / "segmentation",
                "--output-directory", directory,
            ], output / "logs" / f"preview_{name}.log")
        if not image.is_file():
            raise FileNotFoundError(image)
        pipeline.save_json(marker, {"path": str(image)})


def make_fusion(frozen, output, cache, v2_dir):
    marker = output / "progress/fusion.json"
    if marker.is_file():
        directory = Path(pipeline.load_json(marker)["path"])
    else:
        directory = next_attempt(output, "fusion")
        pipeline.run_module("tools.replay_instance_fusion", [
            "--tracking-json", v2_dir / "tracking.json", "--instances-root", cache / "instances_3d",
            "--voxel-size", frozen["settings"]["voxel_size"], "--output-directory", directory,
        ], output / "logs/fusion.log")
    report = pipeline.load_json(directory / "instance_map.json")
    if report["processed_frames"] != FRAMES:
        raise ValueError("融合的帧列表不正确")
    if not (directory / "instance_map_preview.png").is_file():
        raise FileNotFoundError(directory / "instance_map_preview.png")
    pipeline.save_json(marker, {"path": str(directory)})
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-baseline", default="outputs/experiments/office0_0000_0300_v1")
    parser.add_argument("--source-v2", default="outputs/experiments/office0_0000_0300_surface_v2")
    parser.add_argument("--output-directory", default="outputs/experiments/office0_heldout_0310_0600_v1")
    args = parser.parse_args()
    source = pipeline.absolute_path(args.source_baseline)
    source_v2 = pipeline.absolute_path(args.source_v2)
    output = pipeline.absolute_path(args.output_directory)
    print("正在核验冻结代码、旧配置及缓存……", flush=True)
    frozen = preflight(source, source_v2, output)
    freeze_output(output, frozen)
    cache = output / "cache"
    copy_warmup(source, cache)
    processing = argparse.Namespace(**frozen["settings"], reuse_baseline=False)
    for index, frame in enumerate(EVALUATION, start=1):
        print(f"\n新帧 [{index}/{len(EVALUATION)}] Frame {frame:06d}", flush=True)
        pipeline.prepare_frame(processing, frame, cache)

    baseline, updated, comparison, v2_dir = replay(frozen, output, cache)
    report = make_heldout_report(baseline, updated, comparison)
    pipeline.save_json(output / "heldout_report.json", report)
    review_path = output / "manual_review.json"
    review = make_manual_review(baseline, updated)
    # 人工填写的结果属于用户；重跑不能清空它。
    if not review_path.exists():
        pipeline.save_json(review_path, review)
    make_previews(frozen, output, cache, baseline, updated, review)
    fusion_dir = make_fusion(frozen, output, cache, v2_dir)
    pipeline.save_json(output / "result_paths.json", {
        "evaluation_report": str(output / "heldout_report.json"),
        "manual_review": str(review_path), "v2_directory": str(v2_dir),
        "baseline_preview": str(output / "review_baseline/tracking_contact_sheet.png"),
        "v2_preview": str(output / "review_v2/tracking_contact_sheet.png"),
        "fusion_directory": str(fusion_dir),
    })
    print("\n完成：只统计 310～600 帧，共 30 个采样帧。")
    print("基线决策：", report["baseline_decisions"])
    print("V2 决策：", report["v2_decisions"])
    print("待人工核验案例：", review["selected_case_count"])
    print("新建 ID 数减少不等于重复建档减少，更不等于准确率提升。")
    print("结果路径：", output / "result_paths.json")


if __name__ == "__main__":
    main()
