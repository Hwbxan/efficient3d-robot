from pathlib import Path

import cv2
import numpy as np


IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 680

FX = 600.0
FY = 600.0
CX = 599.5
CY = 339.5

DEPTH_SCALE = 6553.5


def create_camera_matrix() -> np.ndarray:
    """创建 Replica 相机内参矩阵。"""

    return np.array(
        [
            [FX, 0.0, CX],
            [0.0, FY, CY],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_camera_poses(pose_path: Path) -> np.ndarray:
    """
    读取 traj.txt 中的 Camera-to-World 位姿。

    traj.txt 的原始位姿可以直接配合 OpenCV 坐标系下的
    RGB-D 反投影使用，不需要翻转坐标轴。
    """

    pose_values = np.loadtxt(
        pose_path,
        dtype=np.float32,
    )

    if pose_values.ndim == 1:
        pose_values = pose_values[None, :]

    if pose_values.shape[1] != 16:
        raise ValueError(
            f"每个位姿应包含 16 个数，实际形状为 {pose_values.shape}"
        )

    camera_to_world = pose_values.reshape(-1, 4, 4).copy()

    expected_last_row = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )

    if not np.allclose(
        camera_to_world[:, 3, :],
        expected_last_row,
        atol=1e-5,
    ):
        raise ValueError("位姿矩阵最后一行不是 [0, 0, 0, 1]")

    return camera_to_world


def read_rgb_image(image_path: Path) -> np.ndarray:
    """读取 RGB 图像，返回 H×W×3 的 RGB 数组。"""

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError(f"无法读取 RGB 图像：{image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_depth_image(depth_path: Path) -> np.ndarray:
    """读取 16-bit 深度图，并转换为米。"""

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth_raw is None:
        raise RuntimeError(f"无法读取深度图：{depth_path}")

    if depth_raw.dtype != np.uint16:
        raise TypeError(
            f"深度图应为 uint16，实际类型为 {depth_raw.dtype}"
        )

    return depth_raw.astype(np.float32) / DEPTH_SCALE


class ReplicaSequence:
    """按帧读取 Replica RGB-D 序列。"""

    def __init__(self, scene_directory: str | Path):
        self.scene_directory = Path(scene_directory)
        self.results_directory = self.scene_directory / "results"
        self.pose_path = self.scene_directory / "traj.txt"

        self.rgb_paths = sorted(
            self.results_directory.glob("frame*.jpg")
        )
        self.depth_paths = sorted(
            self.results_directory.glob("depth*.png")
        )
        self.camera_to_world = load_camera_poses(self.pose_path)
        self.camera_matrix = create_camera_matrix()

        self._validate_sequence()

    def _validate_sequence(self) -> None:
        """检查 RGB、Depth 和 Pose 是否一一对应。"""

        rgb_count = len(self.rgb_paths)
        depth_count = len(self.depth_paths)
        pose_count = len(self.camera_to_world)

        if rgb_count == 0:
            raise RuntimeError(
                f"没有找到 RGB 图像：{self.results_directory}"
            )

        if not (rgb_count == depth_count == pose_count):
            raise RuntimeError(
                "数据数量不一致："
                f"RGB={rgb_count}, "
                f"Depth={depth_count}, "
                f"Pose={pose_count}"
            )

        rgb_ids = [path.stem[5:] for path in self.rgb_paths]
        depth_ids = [path.stem[5:] for path in self.depth_paths]

        if rgb_ids != depth_ids:
            raise RuntimeError("RGB 与 Depth 的帧编号不一致")

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, index: int) -> dict:
        rgb = read_rgb_image(self.rgb_paths[index])
        depth_m = read_depth_image(self.depth_paths[index])

        expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)

        if rgb.shape[:2] != expected_shape:
            raise RuntimeError(
                f"RGB 尺寸错误：{rgb.shape[:2]}，"
                f"预期为 {expected_shape}"
            )

        if depth_m.shape != expected_shape:
            raise RuntimeError(
                f"Depth 尺寸错误：{depth_m.shape}，"
                f"预期为 {expected_shape}"
            )

        frame_id = int(self.rgb_paths[index].stem[5:])

        return {
            "frame_id": frame_id,
            "rgb": rgb,
            "depth_m": depth_m,
            "camera_to_world": self.camera_to_world[index].copy(),
            "camera_matrix": self.camera_matrix.copy(),
        }