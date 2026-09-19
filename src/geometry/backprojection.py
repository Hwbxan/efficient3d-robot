import numpy as np


def backproject_rgbd(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    pixel_stride: int = 4,
    min_depth_m: float = 0.1,
    max_depth_m: float = 10.0,
):
    """
    将 RGB-D 图像反投影为相机坐标系下的彩色点云。

    相机坐标系采用 OpenCV 约定：
        x：向右
        y：向下
        z：向前
    """

    if rgb.shape[:2] != depth_m.shape:
        raise ValueError(
            f"RGB 和 Depth 尺寸不一致："
            f"{rgb.shape[:2]} 与 {depth_m.shape}"
        )

    if pixel_stride < 1:
        raise ValueError("pixel_stride 必须大于或等于 1")

    image_height, image_width = depth_m.shape

    # 每隔 pixel_stride 个像素采样一次，避免点云过于密集。
    sampled_rgb = rgb[::pixel_stride, ::pixel_stride]
    sampled_depth = depth_m[::pixel_stride, ::pixel_stride]

    row_coordinates, column_coordinates = np.mgrid[
        0:image_height:pixel_stride,
        0:image_width:pixel_stride,
    ]

    valid_depth = (
        np.isfinite(sampled_depth)
        & (sampled_depth > min_depth_m)
        & (sampled_depth < max_depth_m)
    )

    z = sampled_depth[valid_depth]
    pixel_x = column_coordinates[valid_depth]
    pixel_y = row_coordinates[valid_depth]

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # 针孔相机模型：
    # pixel_x = fx * x / z + cx
    # pixel_y = fy * y / z + cy
    x = (pixel_x - cx) * z / fx
    y = (pixel_y - cy) * z / fy

    points_camera = np.column_stack((x, y, z)).astype(np.float32)

    colors = sampled_rgb[valid_depth].astype(np.float32) / 255.0

    return points_camera, colors


def transform_points_to_world(
    points_camera: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """将相机坐标系点云转换到世界坐标系。"""

    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]

    points_world = points_camera @ rotation.T + translation

    return points_world.astype(np.float32)