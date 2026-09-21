"""按已有的逐帧关联记录，增量融合每个全局实例的世界坐标点云。

在远程项目根目录运行：python -m tools.replay_instance_fusion
此脚本不重新推理、不估计位姿，也不修改原始观测或关联记录。
"""

import argparse
import json
from colorsys import hsv_to_rgb
from pathlib import Path

import numpy as np

from src.mapping.online_voxel_map import OnlineVoxelMap


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    """已有 JSON 中的相对路径以项目根目录为基准。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def instance_color(global_id):
    hue = (global_id * 0.61803398875) % 1.0
    return np.array(hsv_to_rgb(hue, 0.75, 1.0))


ALLOW_MISSING_CLOUD = True


def read_cloud(path, allow_missing=None):
    import open3d as o3d

    allow = ALLOW_MISSING_CLOUD if allow_missing is None else allow_missing
    if not path.is_file():
        if allow:
            return None, None
        raise FileNotFoundError(f"找不到实例点云：{path}")

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points).copy()
    colors = np.asarray(cloud.colors).copy()

    if len(points) == 0:
        raise ValueError(f"实例点云为空：{path}")
    if colors.shape != points.shape:
        raise ValueError(f"实例点云缺少完整 RGB 颜色：{path}")
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError(f"实例点云含 NaN 或无穷值：{path}")

    return points, colors


def write_cloud(path, points, colors):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"点云保存失败：{path}")


def replay_frames(tracking, instances_root, voxel_size):
    """每个全局实例每帧最多更新一次，避免重复计算观测帧数。"""
    frames = tracking["frames"]
    frame_indices = [frame["frame_index"] for frame in frames]
    if not frames or any(b <= a for a, b in zip(frame_indices, frame_indices[1:])):
        raise ValueError("关联记录必须非空，且帧编号严格递增")
    missing_cloud_count = 0

    instance_maps = {}
    source_frames = {}
    input_counts = {}
    first_labels = {}
    skipped_count = 0

    for frame in frames:
        frame_index = frame["frame_index"]
        metadata_path = instances_root / f"frame_{frame_index:06d}" / "instances_3d.json"
        observations = read_json(metadata_path)
        by_local_id = {item["local_instance_id"]: item for item in observations}
        if len(by_local_id) != len(observations):
            raise ValueError(f"局部实例 ID 重复：{metadata_path}")

        seen_local_ids = set()
        seen_global_ids = set()
        print(f"\nFrame {frame_index:06d}")

        for association in frame["associations"]:
            local_id = association["local_instance_id"]
            global_id = association["global_id"]
            if association["frame_index"] != frame_index:
                raise ValueError("关联项的帧编号与所属帧不一致")
            if local_id in seen_local_ids:
                raise ValueError(f"帧 {frame_index} 重复关联 Local {local_id}")
            seen_local_ids.add(local_id)

            if global_id is None:
                skipped_count += 1
                print(f"  Local {local_id:02d}：尚未分配全局 ID，跳过融合")
                continue

            if association["decision"] not in {"matched", "new_tentative"}:
                raise ValueError(f"不支持的融合决策：{association['decision']}")
            if global_id in seen_global_ids:
                raise ValueError(f"帧 {frame_index} 多次更新 G{global_id:03d}")
            seen_global_ids.add(global_id)

            observation = by_local_id[local_id]
            points, colors = read_cloud(project_path(observation["point_cloud_path"]))
            if points is None:
                # 该观测来自几何传播帧（未做 3D 提升、无独立点云），跳过融合。
                missing_cloud_count += 1
                print(f"  Local {local_id:02d} → G{global_id:03d} | 传播帧，无独立点云，跳过")
                continue

            if global_id not in instance_maps:
                instance_maps[global_id] = OnlineVoxelMap(voxel_size)
                source_frames[global_id] = []
                input_counts[global_id] = 0
                first_labels[global_id] = observation["label"]

            # PLY 中已经是世界坐标；这里不再乘相机位姿。
            result = instance_maps[global_id].update(points, colors, frame_index)
            source_frames[global_id].append(frame_index)
            input_counts[global_id] += len(points)

            print(
                f"  Local {local_id:02d} → G{global_id:03d} | "
                f"输入点={result.input_point_count} | "
                f"新增体素={result.new_voxel_count} | "
                f"累计体素={result.total_voxel_count}"
            )

    if not instance_maps:
        raise ValueError("没有可融合的实例")
    return instance_maps, source_frames, input_counts, first_labels, skipped_count


def save_preview(path, clouds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 10))
    axes = figure.add_subplot(111, projection="3d")

    for global_id, label, points in clouds:
        rng = np.random.default_rng(global_id)
        indices = rng.choice(len(points), min(len(points), 4000), replace=False)
        visible = points[indices]
        axes.scatter(
            visible[:, 0], visible[:, 1], visible[:, 2],
            color=instance_color(global_id), s=1, alpha=0.7,
            label=f"G{global_id:03d} {label}",
        )

    # 保留原始 XYZ，不假定 Y 或 Z 一定是重力方向。
    lower = np.min([points.min(axis=0) for _, _, points in clouds], axis=0)
    upper = np.max([points.max(axis=0) for _, _, points in clouds], axis=0)
    extent = np.maximum(upper - lower, 0.01)
    axes.set_xlim(lower[0] - 0.01, upper[0] + 0.01)
    axes.set_ylim(lower[1] - 0.01, upper[1] + 0.01)
    axes.set_zlim(lower[2] - 0.01, upper[2] + 0.01)
    axes.set_box_aspect(extent)
    axes.set_xlabel("World X (m)")
    axes.set_ylabel("World Y (m)")
    axes.set_zlabel("World Z (m)")
    axes.set_title("Fused 3D instances — original world coordinates")
    axes.legend(loc="upper left", fontsize=8, markerscale=5)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def export_maps(output_directory, tracking, replay_result, voxel_size):
    instance_maps, source_frames, input_counts, first_labels, skipped = replay_result
    final_tracks = {track["global_id"]: track for track in tracking.get("tracks", [])}

    # 默认拒绝覆盖已有目录，保留之前的实验结果。
    output_directory.mkdir(parents=True, exist_ok=False)
    individual_directory = output_directory / "individual"
    individual_directory.mkdir()
    metadata = []
    clouds = []
    combined_points = []
    combined_colors = []

    print("\n融合结果")
    for global_id, instance_map in sorted(instance_maps.items()):
        points, colors, frame_counts = instance_map.to_numpy(minimum_frame_observations=1)
        track = final_tracks.get(global_id, {})
        # 最终标签仅用于导出展示，不参与过去帧的关联或融合决策。
        label = track.get("label", first_labels[global_id])
        stem = f"G{global_id:03d}"
        cloud_path = individual_directory / f"{stem}.ply"
        stats_path = individual_directory / f"{stem}.npz"
        write_cloud(cloud_path, points, colors)
        np.savez_compressed(
            stats_path, points_world=points, colors=colors,
            frame_observations=frame_counts,
        )

        repeat_ratio = float(np.mean(frame_counts >= 2))
        metadata.append({
            "global_id": global_id,
            "label": label,
            "tracking_status": track.get("status", "unknown"),
            "source_frames": source_frames[global_id],
            "observation_count": len(source_frames[global_id]),
            "input_point_count": input_counts[global_id],
            "voxel_count": len(points),
            "multi_frame_voxel_ratio": repeat_ratio,
            "visible_surface_median_world": np.median(points, axis=0).tolist(),
            "bbox_min_world": points.min(axis=0).tolist(),
            "bbox_max_world": points.max(axis=0).tolist(),
            "point_cloud_path": str(cloud_path.relative_to(output_directory)),
            "voxel_statistics_path": str(stats_path.relative_to(output_directory)),
        })
        clouds.append((global_id, label, points))
        combined_points.append(points)
        combined_colors.append(np.tile(instance_color(global_id), (len(points), 1)))
        print(
            f"  {stem} {label:<18} | 观测={len(source_frames[global_id])} 帧 | "
            f"体素={len(points)} | 多帧观测体素占比={repeat_ratio:.1%}"
        )

    write_cloud(
        output_directory / "instances_global_ids.ply",
        np.concatenate(combined_points), np.concatenate(combined_colors),
    )
    report = {
        "voxel_size_m": voxel_size,
        "coordinate_frame": "original_world",
        "path_base": "directory_containing_this_json",
        "processed_frames": [frame["frame_index"] for frame in tracking["frames"]],
        "skipped_association_count": skipped,
        "instances": metadata,
    }
    (output_directory / "instance_map.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    save_preview(output_directory / "instance_map_preview.png", clouds)
    print(f"\n共融合 {len(metadata)} 个全局实例，输出：{output_directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-json", default="outputs/association/six_frame_tracking.json")
    parser.add_argument("--instances-root", default="outputs/instances_3d")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--output-directory", default="outputs/instance_map/six_frames_v1")
    args = parser.parse_args()

    if not np.isfinite(args.voxel_size) or args.voxel_size <= 0:
        parser.error("--voxel-size 必须是有限正数，单位为米")
    output_directory = project_path(args.output_directory)
    if output_directory.exists():
        parser.error("输出目录已存在，请更换 --output-directory，避免覆盖之前的实验")

    tracking = read_json(project_path(args.tracking_json))
    replay_result = replay_frames(tracking, project_path(args.instances_root), args.voxel_size)
    export_maps(output_directory, tracking, replay_result, args.voxel_size)


if __name__ == "__main__":
    main()
