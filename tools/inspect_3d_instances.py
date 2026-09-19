from argparse import ArgumentParser
from pathlib import Path
import json
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from src.datasets.replica_sequence import ReplicaSequence
from src.geometry.instance_lifting import lift_mask_to_world


INSTANCE_COLORS_RGB = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 160, 255),
    (255, 180, 80),
    (220, 80, 255),
    (80, 255, 220),
    (255, 100, 180),
    (180, 255, 80),
    (120, 120, 255),
    (255, 220, 80),
]


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
        "--instance-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--erosion-iterations",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/instances_3d"),
    )

    return parser.parse_args()


def safe_filename(label):
    cleaned = re.sub(
        pattern=r"[^a-zA-Z0-9_-]+",
        repl="_",
        string=label,
    )

    return cleaned.strip("_").lower()


def resolve_mask_path(mask_path_text, json_path):
    mask_path = Path(mask_path_text)

    if mask_path.is_absolute() and mask_path.exists():
        return mask_path

    if mask_path.exists():
        return mask_path.resolve()

    relative_to_json = json_path.parent / mask_path

    if relative_to_json.exists():
        return relative_to_json.resolve()

    raise FileNotFoundError(
        f"找不到实例掩码：{mask_path_text}"
    )


def save_point_cloud(output_path, points, colors):
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
        raise RuntimeError(
            f"点云保存失败：{output_path}"
        )


def draw_bbox(axes, bbox_min, bbox_max, color):
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max

    corners = np.array(
        [
            [x0, z0, y0],
            [x1, z0, y0],
            [x1, z1, y0],
            [x0, z1, y0],
            [x0, z0, y1],
            [x1, z0, y1],
            [x1, z1, y1],
            [x0, z1, y1],
        ]
    )

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for start, end in edges:
        axes.plot(
            corners[[start, end], 0],
            corners[[start, end], 1],
            corners[[start, end], 2],
            color=color,
            linewidth=1.2,
        )


def save_preview(
    output_path,
    observations,
    camera_position,
):
    figure = plt.figure(figsize=(11, 9))
    axes = figure.add_subplot(111, projection="3d")

    all_points = []

    for observation in observations:
        points = observation["geometry"].points_world
        instance_id = observation["local_instance_id"]
        label = observation["label"]

        color_rgb = np.array(
            INSTANCE_COLORS_RGB[
                instance_id % len(INSTANCE_COLORS_RGB)
            ]
        ) / 255.0

        if len(points) > 10_000:
            random_generator = np.random.default_rng(
                seed=instance_id
            )

            selected_indices = random_generator.choice(
                len(points),
                size=10_000,
                replace=False,
            )

            preview_points = points[selected_indices]
        else:
            preview_points = points

        all_points.append(preview_points)

        axes.scatter(
            preview_points[:, 0],
            preview_points[:, 2],
            preview_points[:, 1],
            color=color_rgb,
            s=1.0,
            linewidths=0,
        )

        geometry = observation["geometry"]

        draw_bbox(
            axes=axes,
            bbox_min=geometry.bbox_min,
            bbox_max=geometry.bbox_max,
            color=color_rgb,
        )

        centroid = geometry.centroid

        axes.text(
            centroid[0],
            centroid[2],
            centroid[1],
            f"{instance_id}: {label}",
            fontsize=8,
        )

    axes.scatter(
        camera_position[0],
        camera_position[2],
        camera_position[1],
        color="red",
        marker="^",
        s=80,
        label="Camera",
    )

    concatenated_points = np.concatenate(
        all_points,
        axis=0,
    )

    axis_ranges = np.array(
        [
            np.ptp(concatenated_points[:, 0]),
            np.ptp(concatenated_points[:, 2]),
            np.ptp(concatenated_points[:, 1]),
        ]
    )

    axes.set_box_aspect(
        np.maximum(axis_ranges, 1e-3)
    )

    axes.set_xlabel("World X")
    axes.set_ylabel("World Z")
    axes.set_zlabel("World Y")
    axes.set_title("Single-frame 3D Instance Observations")
    axes.legend()
    axes.view_init(elev=25, azim=-55)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    arguments = parse_arguments()

    sequence = ReplicaSequence(
        arguments.scene_directory
    )
    frame = sequence[arguments.frame_index]

    with arguments.instance_json.open(
        "r",
        encoding="utf-8",
    ) as file:
        instance_metadata = json.load(file)

    frame_output_directory = (
        arguments.output_directory
        / f"frame_{frame['frame_id']:06d}"
    )

    individual_directory = (
        frame_output_directory / "individual"
    )

    individual_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    observations = []
    output_metadata = []

    combined_points = []
    combined_instance_colors = []

    for metadata in instance_metadata:
        mask_path = resolve_mask_path(
            metadata["mask_path"],
            arguments.instance_json,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(
                f"无法读取掩码：{mask_path}"
            )

        geometry = lift_mask_to_world(
            rgb=frame["rgb"],
            depth_m=frame["depth_m"],
            mask=mask > 0,
            camera_matrix=frame["camera_matrix"],
            camera_to_world=frame["camera_to_world"],
            pixel_stride=arguments.pixel_stride,
            erosion_iterations=arguments.erosion_iterations,
        )

        instance_id = metadata["local_instance_id"]
        label = metadata["label"]

        observations.append(
            {
                "local_instance_id": instance_id,
                "label": label,
                "geometry": geometry,
            }
        )

        instance_name = (
            f"{instance_id:02d}_"
            f"{safe_filename(label)}"
        )

        individual_path = (
            individual_directory
            / f"{instance_name}.ply"
        )

        save_point_cloud(
            output_path=individual_path,
            points=geometry.points_world,
            colors=geometry.colors,
        )

        instance_color = np.array(
            INSTANCE_COLORS_RGB[
                instance_id % len(INSTANCE_COLORS_RGB)
            ],
            dtype=np.float32,
        ) / 255.0

        instance_colors = np.repeat(
            instance_color[None, :],
            len(geometry.points_world),
            axis=0,
        )

        combined_points.append(
            geometry.points_world
        )
        combined_instance_colors.append(
            instance_colors
        )

        output_metadata.append(
            {
                "local_instance_id": instance_id,
                "label": label,
                "detection_score": metadata["detection_score"],
                "sam_predicted_iou": metadata["sam_predicted_iou"],
                "point_count": len(geometry.points_world),
                "valid_depth_ratio": geometry.valid_depth_ratio,
                "median_depth_m": geometry.median_depth_m,
                "centroid_world": geometry.centroid.tolist(),
                "bbox_min_world": geometry.bbox_min.tolist(),
                "bbox_max_world": geometry.bbox_max.tolist(),
                "bbox_extent_m": geometry.bbox_extent.tolist(),
                "point_cloud_path": str(individual_path),
            }
        )

        print(
            f"{instance_id:02d} | "
            f"{label:18s} | "
            f"Points={len(geometry.points_world):6d} | "
            f"Depth={geometry.median_depth_m:.2f} m | "
            f"Valid={geometry.valid_depth_ratio:.3f} | "
            f"Extent={geometry.bbox_extent.round(2)}"
        )

    combined_points = np.concatenate(
        combined_points,
        axis=0,
    )
    combined_instance_colors = np.concatenate(
        combined_instance_colors,
        axis=0,
    )

    combined_path = (
        frame_output_directory
        / "instances_colored_by_id.ply"
    )

    preview_path = (
        frame_output_directory
        / "instances_3d_preview.png"
    )

    metadata_path = (
        frame_output_directory
        / "instances_3d.json"
    )

    save_point_cloud(
        output_path=combined_path,
        points=combined_points,
        colors=combined_instance_colors,
    )

    camera_position = (
        frame["camera_to_world"][:3, 3]
    )

    save_preview(
        output_path=preview_path,
        observations=observations,
        camera_position=camera_position,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n3D 实例数量：{len(observations)}")
    print(f"组合实例点云：{combined_path.resolve()}")
    print(f"3D 预览：{preview_path.resolve()}")
    print(f"3D 元数据：{metadata_path.resolve()}")


if __name__ == "__main__":
    main()