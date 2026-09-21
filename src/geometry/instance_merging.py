"""合并单帧内重复 / 过分割的实例观测。

为什么需要这一步
----------------
Grounding DINO 会对同一物体给出多个高度重叠的框（NMS 之后仍有残留），SAM2
也会把一把椅子的坐垫与靠背切成两块。这不是跟踪器的问题，跟踪器再怎么调都
救不回来：这些重复观测在评估里必然算成假阳性，在跟踪里会把同一物体切成
多条同时活跃的轨道。

实测 room_2（200 帧）里有 73 帧存在同标签、体素 IoU > 0.05 的重复观测，
其中帧 10 的两个 chair 实例体素 IoU 0.79、质心只差 0.06 m——几乎完全重合。

判定依据
--------
用 3D 体素覆盖率（|A∩B| / min(|A|,|B|)）而不是 2D mask IoU：2D 上前后景物体
可能大面积重叠（站在椅子前的桌子在图像上会盖住椅子），3D 上不同物体几乎不
占同一批体素——同类相邻物体的体素覆盖率实测接近 0，而同一物体的重复观测在
0.7 以上，间隔非常干净。质心距离只作为附加约束，用来挡掉「覆盖率偶然偏高但
空间上分开很远」的极端情况。
"""

from __future__ import annotations

import numpy as np


def voxel_key_set(points_world, voxel_size):
    """点云 → 体素索引的 frozenset，供两两 IoU 计算复用。"""

    points = np.asarray(points_world, dtype=np.float64)
    if len(points) == 0:
        return frozenset()
    indices = np.unique(
        np.floor(points / voxel_size).astype(np.int64), axis=0
    )
    return frozenset(tuple(int(value) for value in row) for row in indices)


def set_iou(first, second):
    union = len(first | second)
    if union == 0:
        return 0.0
    return len(first & second) / union


def set_coverage(first, second):
    """覆盖率 = |A∩B| / min(|A|,|B|)。

    用覆盖率而不是 IoU 判定：同一物体被切成一大一小两块时，小块几乎完整落在
    大块里，覆盖率接近 1，而 IoU 会被大块的分母压得很低。实测 room_2 里同一
    把椅子的两个碎片覆盖率 1.000 但 IoU 只有 0.289——用 IoU 阈值 0.35 判不出
    来，用覆盖率 0.5 就干净地判出来了。
    """

    smaller = min(len(first), len(second))
    if smaller == 0:
        return 0.0
    return len(first & second) / smaller


def plan_merges(entries, iou_threshold, max_center_distance=0.0,
                allow_cross_label=False, known_labels=None):
    """把应合并的下标分到同一组。

    entries: [{"label": str, "voxels": frozenset, "centroid": array-like}, ...]
    返回：合并组列表，如 [[0, 3], [7, 9, 11]]；只含成员数 ≥ 2 的组。
    用并查集而不是贪心两两合并：三个互相重叠的实例应当一次合成一个。

    判定用覆盖率：两个碎片谁大谁小不影响结论，只要小的几乎完整落在大的里面
    就该合并。

    allow_cross_label=True 时允许不同标签的观测合并（用于「desk / table」这类
    同义提示词把同一物体检出两遍的情况）。安全性来自判据本身：3D 体素覆盖率
    高意味着两者占了同一批体素，而两个**不同**物体在 3D 上几乎不共享体素。
    known_labels 给出时只允许白名单内的标签参与跨标签合并，避免把检测器偶尔
    给出的杂标签并进正常实例。
    """

    count = len(entries)
    if known_labels is not None:
        known_labels = set(known_labels)
    parent = list(range(count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            # 始终让较小的下标当根，输出顺序稳定
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for i in range(count):
        for j in range(i + 1, count):
            same_label = entries[i]["label"] == entries[j]["label"]
            if not same_label:
                if not allow_cross_label:
                    continue
                if known_labels is not None and not (
                    entries[i]["label"] in known_labels
                    and entries[j]["label"] in known_labels
                ):
                    continue
            if set_coverage(
                entries[i]["voxels"], entries[j]["voxels"]
            ) < iou_threshold:
                continue
            if max_center_distance > 0:
                distance = float(np.linalg.norm(
                    np.asarray(entries[i]["centroid"], dtype=np.float64)
                    - np.asarray(entries[j]["centroid"], dtype=np.float64)
                ))
                if distance > max_center_distance:
                    continue
            union(i, j)

    groups = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    return [group for group in groups.values() if len(group) > 1]
