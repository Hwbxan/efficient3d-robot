from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
import open3d as o3d

from src.datasets.replica_sequence import ReplicaSequence
from src.geometry.backprojection import (
    backproject_rgbd,
    transform_points_to_world,
)
from src.mapping.online_voxel_map import OnlineVoxelMap
from tools.fuse_replica_frames import (
    create_open3d_point_cloud,
    save_map_preview,
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
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
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
    )

    parser.add_argument(
        "--minimum-frame-observations",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/online_mapping"),
    )

    return parser.parse_args()


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
        raise ValueError("没有可以处理的帧")

    voxel_map = OnlineVoxelMap(
        voxel_size=arguments.voxel_size
    )

    camera_positions = []
    processing_times_ms = []

    for sequence_number, frame_index in enumerate(
        frame_indices,
        start=1,
    ):
        frame_start_time = perf_counter()

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

        update_result = voxel_map.update(
            points_world=points_world,
            colors=colors,
            frame_index=frame_index,
        )

        camera_positions.append(
            frame["camera_to_world"][:3, 3]
        )

        processing_time_ms = (
            perf_counter() - frame_start_time
        ) * 1000.0

        processing_times_ms.append(processing_time_ms)

        print(
            f"[{sequence_number:02d}/{len(frame_indices):02d}] "
            f"帧 {frame_index:04d} | "
            f"输入点 {update_result.input_point_count:6d} | "
            f"当前观测体素 {update_result.observed_voxel_count:6d} | "
            f"新增体素 {update_result.new_voxel_count:6d} | "
            f"地图体素 {update_result.total_voxel_count:7d} | "
            f"{processing_time_ms:7.2f} ms"
        )

    map_points, map_colors, frame_observations = voxel_map.to_numpy(
        minimum_frame_observations=(
            arguments.minimum_frame_observations
        )
    )

    camera_positions = np.asarray(
        camera_positions,
        dtype=np.float32,
    )

    point_cloud = create_open3d_point_cloud(
        points=map_points,
        colors=map_colors,
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scene_name = arguments.scene_directory.name

    output_name = (
        f"{scene_name}_online_"
        f"{arguments.start_frame:04d}_"
        f"{end_frame:04d}"
    )

    point_cloud_path = (
        arguments.output_directory
        / f"{output_name}.ply"
    )

    preview_path = (
        arguments.output_directory
        / f"{output_name}_preview.png"
    )

    observation_path = (
        arguments.output_directory
        / f"{output_name}_observations.npy"
    )

    o3d.io.write_point_cloud(
        str(point_cloud_path),
        point_cloud,
        write_ascii=False,
    )

    np.save(
        observation_path,
        frame_observations,
    )

    save_map_preview(
        output_path=preview_path,
        points=map_points,
        colors=map_colors,
        camera_positions=camera_positions,
    )

    average_time_ms = float(
        np.mean(processing_times_ms)
    )
    average_fps = 1000.0 / average_time_ms

    print("\n在线建图完成")
    print(f"处理帧数：{len(frame_indices)}")
    print(f"最终体素数：{len(map_points)}")
    print(f"平均单帧时间：{average_time_ms:.2f} ms")
    print(f"平均处理速度：{average_fps:.2f} FPS")
    print(f"点云地图：{point_cloud_path.resolve()}")
    print(f"预览图片：{preview_path.resolve()}")


if __name__ == "__main__":
    main()