from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from src.datasets.replica_sequence import ReplicaSequence
from src.geometry.backprojection import (
    backproject_rgbd,
    transform_points_to_world,
)


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--scene-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        default=300,
        help="结束帧，不包含该帧",
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="每隔多少帧融合一次",
    )

    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.03,
        help="体素降采样尺寸，单位为米",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/data_check/multiframe_map"),
    )

    return parser.parse_args()


def create_open3d_point_cloud(points, colors):
    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(
        points.astype(np.float64)
    )
    point_cloud.colors = o3d.utility.Vector3dVector(
        colors.astype(np.float64)
    )

    return point_cloud


def save_map_preview(
    output_path,
    points,
    colors,
    camera_positions,
):
    maximum_preview_points = 100_000

    if len(points) > maximum_preview_points:
        random_generator = np.random.default_rng(seed=0)

        selected_indices = random_generator.choice(
            len(points),
            size=maximum_preview_points,
            replace=False,
        )

        preview_points = points[selected_indices]
        preview_colors = colors[selected_indices]
    else:
        preview_points = points
        preview_colors = colors

    figure = plt.figure(figsize=(11, 9))
    axes = figure.add_subplot(111, projection="3d")

    # Replica 使用 Y 轴作为竖直方向。
    axes.scatter(
        preview_points[:, 0],
        preview_points[:, 2],
        preview_points[:, 1],
        c=preview_colors,
        s=0.4,
        linewidths=0,
    )

    # 绘制相机移动轨迹。
    axes.plot(
        camera_positions[:, 0],
        camera_positions[:, 2],
        camera_positions[:, 1],
        color="red",
        linewidth=2,
        label="Camera trajectory",
    )

    axes.scatter(
        camera_positions[:, 0],
        camera_positions[:, 2],
        camera_positions[:, 1],
        color="red",
        s=12,
    )

    axes.set_xlabel("World X")
    axes.set_ylabel("World Z")
    axes.set_zlabel("World Y")
    axes.set_title("Multi-frame RGB-D Map")
    axes.legend()

    axis_ranges = np.array(
        [
            np.ptp(preview_points[:, 0]),
            np.ptp(preview_points[:, 2]),
            np.ptp(preview_points[:, 1]),
        ]
    )

    axes.set_box_aspect(np.maximum(axis_ranges, 1e-3))
    axes.view_init(elev=28, azim=-55)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    arguments = parse_arguments()

    sequence = ReplicaSequence(arguments.scene_directory)

    end_frame = min(arguments.end_frame, len(sequence))

    frame_indices = list(
        range(
            arguments.start_frame,
            end_frame,
            arguments.frame_step,
        )
    )

    if not frame_indices:
        raise ValueError("没有可供融合的帧")

    world_points_buffer = []
    colors_buffer = []
    camera_positions = []

    for progress, frame_index in enumerate(frame_indices, start=1):
        frame = sequence[frame_index]

        points_camera, colors = backproject_rgbd(
            rgb=frame["rgb"],
            depth_m=frame["depth_m"],
            camera_matrix=frame["camera_matrix"],
            pixel_stride=arguments.pixel_stride,
        )

        points_world = transform_points_to_world(
            points_camera=points_camera,
            camera_to_world=frame["camera_to_world"],
        )

        world_points_buffer.append(points_world)
        colors_buffer.append(colors)

        camera_position = frame["camera_to_world"][:3, 3]
        camera_positions.append(camera_position)

        print(
            f"[{progress:02d}/{len(frame_indices):02d}] "
            f"已融合第 {frame_index} 帧，"
            f"新增 {len(points_world)} 个点"
        )

    world_points = np.concatenate(
        world_points_buffer,
        axis=0,
    )
    colors = np.concatenate(
        colors_buffer,
        axis=0,
    )
    camera_positions = np.asarray(
        camera_positions,
        dtype=np.float32,
    )

    print(f"\n体素降采样前：{len(world_points)} 个点")

    point_cloud = create_open3d_point_cloud(
        points=world_points,
        colors=colors,
    )

    point_cloud = point_cloud.voxel_down_sample(
        voxel_size=arguments.voxel_size
    )

    downsampled_points = np.asarray(point_cloud.points)
    downsampled_colors = np.asarray(point_cloud.colors)

    print(f"体素降采样后：{len(downsampled_points)} 个点")

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    point_cloud_path = (
        arguments.output_directory
        / "office0_frames_0000_0300.ply"
    )

    preview_path = (
        arguments.output_directory
        / "office0_frames_0000_0300_preview.png"
    )

    o3d.io.write_point_cloud(
        str(point_cloud_path),
        point_cloud,
        write_ascii=False,
    )

    save_map_preview(
        output_path=preview_path,
        points=downsampled_points,
        colors=downsampled_colors,
        camera_positions=camera_positions,
    )

    print(f"\n点云地图：{point_cloud_path.resolve()}")
    print(f"地图预览：{preview_path.resolve()}")


if __name__ == "__main__":
    main()