import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def load_observations(path):
    """读取并检查单帧 3D 实例观测。"""

    with path.open("r", encoding="utf-8") as file:
        observations = json.load(file)

    valid_observations = []

    for observation in observations:
        instance_id = observation["local_instance_id"]

        center = np.asarray(
            observation["centroid_world"],
            dtype=np.float64,
        )
        bbox_min = np.asarray(
            observation["bbox_min_world"],
            dtype=np.float64,
        )
        bbox_max = np.asarray(
            observation["bbox_max_world"],
            dtype=np.float64,
        )

        coordinates = np.concatenate(
            [center, bbox_min, bbox_max]
        )

        if not np.isfinite(coordinates).all():
            raise ValueError(
                f"{path}：实例 {instance_id} 包含无效坐标"
            )

        if np.any(bbox_max < bbox_min):
            raise ValueError(
                f"{path}：实例 {instance_id} 包围盒上下界错误"
            )

        point_count = observation["point_count"]

        if point_count < 30:
            print(
                f"跳过实例 {instance_id}："
                f"仅有 {point_count} 个点"
            )
            continue

        if observation["valid_depth_ratio"] < 0.8:
            print(
                f"提示：实例 {instance_id} 的有效深度比例较低"
            )

        valid_observations.append(observation)

    return valid_observations


def geometry_distances(first, second):
    """计算观测中心距离，以及两个 AABB 之间的最短距离。"""

    first_center = np.asarray(first["centroid_world"])
    second_center = np.asarray(second["centroid_world"])

    center_distance = np.linalg.norm(
        first_center - second_center
    )

    first_min = np.asarray(first["bbox_min_world"])
    first_max = np.asarray(first["bbox_max_world"])
    second_min = np.asarray(second["bbox_min_world"])
    second_max = np.asarray(second["bbox_max_world"])

    # 每个坐标轴上的区间间隔。
    # 区间重叠时，该轴上的间隔为 0。
    axis_gaps = np.maximum(
        np.maximum(
            first_min - second_max,
            second_min - first_max,
        ),
        0.0,
    )

    bbox_gap = np.linalg.norm(axis_gaps)

    return float(center_distance), float(bbox_gap)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--reference-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--current-json",
        type=Path,
        required=True,
    )

    arguments = parser.parse_args()

    reference = load_observations(arguments.reference_json)
    current = load_observations(arguments.current_json)

    print(f"\n参考帧有效实例：{len(reference)}")
    print(f"当前帧有效实例：{len(current)}")

    for observation in current:
        instance_id = observation["local_instance_id"]
        label = observation["label"]

        candidates = []

        for previous in reference:
            if previous["label"] != label:
                continue

            center_distance, bbox_gap = geometry_distances(
                observation,
                previous,
            )

            candidates.append(
                (
                    center_distance,
                    bbox_gap,
                    previous["local_instance_id"],
                )
            )

        candidates.sort(key=lambda candidate: candidate[0])

        print(f"\n当前实例 {instance_id:02d} | {label}")

        if not candidates:
            print("  参考帧没有同类别候选")
            continue

        for rank, candidate in enumerate(
            candidates[:2],
            start=1,
        ):
            center_distance, bbox_gap, previous_id = candidate

            print(
                f"  候选 {rank}：参考实例 {previous_id:02d} | "
                f"中心距离 {center_distance:.3f} m | "
                f"包围盒间距 {bbox_gap:.3f} m"
            )


if __name__ == "__main__":
    main()