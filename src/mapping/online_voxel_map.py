from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelRecord:
    """一个体素内部累计的统计信息。"""

    point_sum: np.ndarray
    color_sum: np.ndarray
    sample_count: int
    frame_observation_count: int
    last_seen_frame: int


@dataclass(frozen=True)
class MapUpdateResult:
    """一次在线更新的统计结果。"""

    input_point_count: int
    observed_voxel_count: int
    new_voxel_count: int
    total_voxel_count: int


class OnlineVoxelMap:
    """
    使用哈希表维护在线彩色体素地图。

    每个体素保存：
        - 三维点坐标累加值
        - RGB 颜色累加值
        - 累计点数
        - 被多少帧观测过
        - 最后一次观测帧
    """

    def __init__(self, voxel_size: float):
        if voxel_size <= 0:
            raise ValueError("voxel_size 必须大于 0")

        self.voxel_size = float(voxel_size)
        self._voxels = {}

    def __len__(self) -> int:
        return len(self._voxels)

    def update(
        self,
        points_world: np.ndarray,
        colors: np.ndarray,
        frame_index: int,
    ) -> MapUpdateResult:
        """使用当前帧点云增量更新地图。"""

        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError(
                f"points_world 应为 N×3，实际为 {points_world.shape}"
            )

        if colors.shape != points_world.shape:
            raise ValueError(
                f"颜色与点云形状不一致："
                f"{colors.shape} 与 {points_world.shape}"
            )

        if len(points_world) == 0:
            return MapUpdateResult(
                input_point_count=0,
                observed_voxel_count=0,
                new_voxel_count=0,
                total_voxel_count=len(self),
            )

        voxel_coordinates = np.floor(
            points_world / self.voxel_size
        ).astype(np.int32)

        # 先在当前帧内部进行体素聚合。
        unique_voxels, inverse_indices = np.unique(
            voxel_coordinates,
            axis=0,
            return_inverse=True,
        )

        voxel_count = len(unique_voxels)

        point_sums = np.zeros(
            (voxel_count, 3),
            dtype=np.float64,
        )
        color_sums = np.zeros(
            (voxel_count, 3),
            dtype=np.float64,
        )

        np.add.at(
            point_sums,
            inverse_indices,
            points_world,
        )
        np.add.at(
            color_sums,
            inverse_indices,
            colors,
        )

        sample_counts = np.bincount(
            inverse_indices,
            minlength=voxel_count,
        )

        new_voxel_count = 0

        for voxel_index, voxel_coordinate in enumerate(unique_voxels):
            voxel_key = tuple(
                int(value) for value in voxel_coordinate
            )

            existing_record = self._voxels.get(voxel_key)

            if existing_record is None:
                self._voxels[voxel_key] = VoxelRecord(
                    point_sum=point_sums[voxel_index].copy(),
                    color_sum=color_sums[voxel_index].copy(),
                    sample_count=int(sample_counts[voxel_index]),
                    frame_observation_count=1,
                    last_seen_frame=frame_index,
                )

                new_voxel_count += 1

            else:
                existing_record.point_sum += point_sums[voxel_index]
                existing_record.color_sum += color_sums[voxel_index]
                existing_record.sample_count += int(
                    sample_counts[voxel_index]
                )
                existing_record.frame_observation_count += 1
                existing_record.last_seen_frame = frame_index

        return MapUpdateResult(
            input_point_count=len(points_world),
            observed_voxel_count=voxel_count,
            new_voxel_count=new_voxel_count,
            total_voxel_count=len(self),
        )

    def to_numpy(
        self,
        minimum_frame_observations: int = 1,
    ):
        """导出体素中心点、平均颜色和观测次数。"""

        selected_records = [
            record
            for record in self._voxels.values()
            if record.frame_observation_count
            >= minimum_frame_observations
        ]

        if not selected_records:
            raise RuntimeError("地图中没有满足条件的体素")

        points = np.stack(
            [
                record.point_sum / record.sample_count
                for record in selected_records
            ]
        ).astype(np.float32)

        colors = np.stack(
            [
                record.color_sum / record.sample_count
                for record in selected_records
            ]
        ).astype(np.float32)

        frame_observations = np.array(
            [
                record.frame_observation_count
                for record in selected_records
            ],
            dtype=np.int32,
        )

        colors = np.clip(colors, 0.0, 1.0)

        return points, colors, frame_observations