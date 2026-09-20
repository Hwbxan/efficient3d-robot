from dataclasses import dataclass

import cv2
import numpy as np

from src.geometry.backprojection import transform_points_to_world


@dataclass(frozen=True)
class InstancePointCloud3D:
    """单帧中的一个 3D 实例观测。"""

    points_world: np.ndarray
    colors: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bbox_extent: np.ndarray
    valid_depth_ratio: float
    median_depth_m: float


def voxel_indices(points_world, voxel_size=0.05):
    """把实例点云量化成体素索引，去重后返回 (N, 3) 的整数数组。

    跨帧实例关联用得上：世界坐标下把点量化成体素后，同一物体的不同视角会
    落到几乎相同的体素集合上，而相邻的不同物体几乎不重叠。
    """

    points = np.asarray(points_world, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.int64)

    return np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0)


def lift_mask_to_world(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    camera_to_world: np.ndarray,
    pixel_stride: int = 2,
    erosion_iterations: int = 1,
    min_depth_m: float = 0.1,
    max_depth_m: float = 10.0,
) -> InstancePointCloud3D:
    """将一个 2D 实例掩码反投影到世界坐标系。"""

    if mask.shape != depth_m.shape:
        raise ValueError(
            f"Mask 和 Depth 尺寸不一致："
            f"{mask.shape} 与 {depth_m.shape}"
        )

    if rgb.shape[:2] != depth_m.shape:
        raise ValueError(
            f"RGB 和 Depth 尺寸不一致："
            f"{rgb.shape[:2]} 与 {depth_m.shape}"
        )

    binary_mask = mask.astype(bool)

    # 腐蚀一个像素，减少物体边界混入背景深度。
    if erosion_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)

        binary_mask = cv2.erode(
            binary_mask.astype(np.uint8),
            kernel,
            iterations=erosion_iterations,
        ).astype(bool)

    mask_pixel_count = int(binary_mask.sum())

    if mask_pixel_count == 0:
        raise ValueError("腐蚀后的实例掩码为空")

    valid_depth_full = (
        np.isfinite(depth_m)
        & (depth_m > min_depth_m)
        & (depth_m < max_depth_m)
    )

    valid_depth_ratio = float(
        np.count_nonzero(binary_mask & valid_depth_full)
        / mask_pixel_count
    )

    image_height, image_width = depth_m.shape

    sampled_rgb = rgb[::pixel_stride, ::pixel_stride]
    sampled_depth = depth_m[::pixel_stride, ::pixel_stride]
    sampled_mask = binary_mask[::pixel_stride, ::pixel_stride]

    pixel_y, pixel_x = np.mgrid[
        0:image_height:pixel_stride,
        0:image_width:pixel_stride,
    ]

    valid_points = (
        sampled_mask
        & np.isfinite(sampled_depth)
        & (sampled_depth > min_depth_m)
        & (sampled_depth < max_depth_m)
    )

    candidate_depths = sampled_depth[valid_points]

    if len(candidate_depths) == 0:
        raise ValueError("实例掩码中没有有效深度")

    # 删除极少量深度边缘离群点。
    if len(candidate_depths) >= 20:
        depth_lower, depth_upper = np.percentile(
            candidate_depths,
            [1.0, 99.0],
        )

        valid_points &= (
            (sampled_depth >= depth_lower)
            & (sampled_depth <= depth_upper)
        )

    z = sampled_depth[valid_points]
    u = pixel_x[valid_points]
    v = pixel_y[valid_points]

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_camera = np.column_stack(
        (x, y, z)
    ).astype(np.float32)

    colors = (
        sampled_rgb[valid_points]
        .astype(np.float32)
        / 255.0
    )

    points_world = transform_points_to_world(
        points_camera=points_camera,
        camera_to_world=camera_to_world,
    )

    bbox_min = points_world.min(axis=0)
    bbox_max = points_world.max(axis=0)
    bbox_extent = bbox_max - bbox_min

    # 中位数比均值更不容易受到少量错误深度影响。
    centroid = np.median(
        points_world,
        axis=0,
    )

    return InstancePointCloud3D(
        points_world=points_world,
        colors=colors,
        centroid=centroid.astype(np.float32),
        bbox_min=bbox_min.astype(np.float32),
        bbox_max=bbox_max.astype(np.float32),
        bbox_extent=bbox_extent.astype(np.float32),
        valid_depth_ratio=valid_depth_ratio,
        median_depth_m=float(np.median(z)),
    )