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
        "--frame-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=4,
        help="像素采样间隔，4 表示横纵方向每隔 4 个像素取一点",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/data_check/single_frame_pointcloud"),
    )

    return parser.parse_args()


def save_point_cloud(
    output_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
):
    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(
        points.astype(np.float64)
    )
    point_cloud.colors = o3d.utility.Vector3dVector(
        colors.astype(np.float64)
    )

    success = o3d.io.write_point_cloud(
        str(output_path),
        point_cloud,
        write_ascii=False,
    )

    if not success:
        raise RuntimeError(f"点云保存失败：{output_path}")


def save_point_cloud_preview(
    output_path: Path,
    points_world: np.ndarray,
    colors: np.ndarray,
    camera_position: np.ndarray,
):
    # Matplotlib 只负责生成静态预览图。
    # 最多绘制 60000 个点，避免图片生成过慢。
    maximum_preview_points = 60_000

    if len(points_world) > maximum_preview_points:
        random_generator = np.random.default_rng(seed=0)

        selected_indices = random_generator.choice(
            len(points_world),
            size=maximum_preview_points,
            replace=False,
        )

        preview_points = points_world[selected_indices]
        preview_colors = colors[selected_indices]
    else:
        preview_points = points_world
        preview_colors = colors

    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")

    # Replica 世界坐标系中 Y 轴是竖直方向。
    axes.scatter(
        preview_points[:, 0],
        preview_points[:, 2],
        preview_points[:, 1],
        c=preview_colors,
        s=0.6,
        linewidths=0,
    )

    axes.scatter(
        camera_position[0],
        camera_position[2],
        camera_position[1],
        c="red",
        s=80,
        marker="^",
        label="Camera",
    )

    axes.set_xlabel("World X")
    axes.set_ylabel("World Z")
    axes.set_zlabel("World Y")
    axes.set_title("Single-frame RGB-D Point Cloud")
    axes.legend()

    axis_ranges = (
        np.ptp(preview_points[:, 0]),
        np.ptp(preview_points[:, 2]),
        np.ptp(preview_points[:, 1]),
    )
    axes.set_box_aspect(axis_ranges)
    axes.view_init(elev=25, azim=-60)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    arguments = parse_arguments()

    sequence = ReplicaSequence(arguments.scene_directory)
    frame = sequence[arguments.frame_index]

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

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    ply_path = (
        arguments.output_directory
        / f"frame_{frame['frame_id']:06d}_world.ply"
    )

    preview_path = (
        arguments.output_directory
        / f"frame_{frame['frame_id']:06d}_preview.png"
    )

    save_point_cloud(
        output_path=ply_path,
        points=points_world,
        colors=colors,
    )

    camera_position = frame["camera_to_world"][:3, 3]

    save_point_cloud_preview(
        output_path=preview_path,
        points_world=points_world,
        colors=colors,
        camera_position=camera_position,
    )

    print(f"原始 RGB 像素数：{frame['rgb'].shape[0] * frame['rgb'].shape[1]}")
    print(f"生成的有效点数：{len(points_world)}")
    print(f"相机位置：{camera_position}")
    print(f"世界坐标最小值：{points_world.min(axis=0)}")
    print(f"世界坐标最大值：{points_world.max(axis=0)}")
    print(f"点云文件：{ply_path.resolve()}")
    print(f"预览图片：{preview_path.resolve()}")


if __name__ == "__main__":
    main()