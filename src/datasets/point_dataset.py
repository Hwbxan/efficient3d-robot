"""点云数据集：从 `tools/build_mesh_point_dataset.py` 产出的 npz 里采样训练块。

两个关键设计：

1. **局部块而不是全局随机子集。** 随机抽 P 个点会得到一个"稀疏地覆盖整个房间"
   的样本，局部几何上下文很弱。这里以某个点为中心取最近的 P 个点，
   得到稠密局部块，语义分割才有意义。用 `argpartition` 一次 O(N) 完成，
   不需要 KD-tree。

2. **类别均衡采样。** Replica 里 wall / floor / ceiling 的**面积**占绝对多数
   （见 manifest 的 class_histogram），纯随机采样会让模型退化成"全猜地板"。
   这里按 `1/freq^alpha` 加权选中心点所属类别，再在该类点里随机取一个作中心，
   直接提升稀有类别的出现频率。

增强：世界系是 **Z 轴向上**（已用相机位姿验证过重力方向），
所以绕 Z 轴随机旋转是物理上合法的增强；倾斜旋转会把地板转到墙上，不能做。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# 类别 -1 表示"无标注"，训练时用 ignore_index 跳过。
IGNORE_INDEX = -1


class PointCloudPatchDataset(Dataset):
    """每个 `__getitem__` 返回一个 `(features, class_id, instance_id)` 三元组。

    features: `(num_points, 9)` float32，xyz 已中心化、normal 已归一、rgb 在 [0,1]
    """

    def __init__(
        self,
        dataset_root,
        split="train",
        num_points=8192,
        patches_per_scene=64,
        class_alpha=0.5,
        augment=True,
        seed=0,
        scene_limit=None,
        normalize=True,
    ):
        self.root = Path(dataset_root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"找不到 {manifest_path}；先跑 tools.build_mesh_point_dataset"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.class_names = self.manifest["class_names"]

        if split not in self.manifest["splits"]:
            raise ValueError(f"未知划分 {split}，可选 {sorted(self.manifest['splits'])}")
        self.split = split
        self.scenes = list(self.manifest["splits"][split])
        if scene_limit:
            self.scenes = self.scenes[:scene_limit]
        if not self.scenes:
            raise ValueError(f"划分 {split} 里没有场景；检查 manifest 的 splits")

        self.num_points = num_points
        self.patches_per_scene = patches_per_scene
        self.class_alpha = class_alpha
        self.augment = augment
        self.seed = seed
        self.normalize = normalize
        self.epoch = 0

        self._cache = {}
        self._class_pools = {}

    # ------------------------------------------------------------------ #

    def _arrays(self, scene):
        if scene not in self._cache:
            path = self.root / "scenes" / f"{scene}.npz"
            if not path.exists():
                raise FileNotFoundError(f"缺少场景文件 {path}")
            data = np.load(path)
            self._cache[scene] = {
                "xyz": data["xyz"].astype(np.float32),
                "normal": data["normal"].astype(np.float32),
                "rgb": data["rgb"].astype(np.float32),
                "instance": data["instance"].astype(np.int64),
                "class": data["class"].astype(np.int64),
            }
        return self._cache[scene]

    def _class_sampling_probabilities(self, scene):
        """按 `1/freq^alpha` 给类别加权，用于选中心点的类别。"""

        if scene in self._class_pools:
            return self._class_pools[scene]

        labels = self._arrays(scene)["class"]
        valid = labels >= 0
        unique, counts = np.unique(labels[valid], return_counts=True)

        weights = 1.0 / np.power(counts.astype(np.float64), self.class_alpha)
        weights = weights / weights.sum()
        # 每个类别对应的点索引池，供随机取中心点
        pools = {int(value): np.flatnonzero(labels == value) for value in unique}

        self._class_pools[scene] = (unique.astype(np.int64), weights, pools)
        return self._class_pools[scene]

    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.scenes) * self.patches_per_scene

    def set_epoch(self, epoch):
        """训练循环每个 epoch 调用一次，让采样序列随之变化。"""

        self.epoch = epoch

    def __getitem__(self, index):
        scene = self.scenes[index % len(self.scenes)]
        arrays = self._arrays(scene)
        rng = np.random.default_rng(self.seed + 100003 * self.epoch + index)

        points = arrays["xyz"]
        centre_index = self._pick_centre(scene, arrays, rng)

        # 以中心点为核心取最近的 num_points 个点：一次 O(N) 的 argpartition。
        distance = np.einsum("ij,ij->i", points - points[centre_index], points - points[centre_index])
        if points.shape[0] > self.num_points:
            picked = np.argpartition(distance, self.num_points - 1)[: self.num_points]
        else:
            picked = np.arange(points.shape[0])
        # 打乱顺序，否则 argpartition 的输出没有确定次序，会让 batch 内的空间分布偏置
        rng.shuffle(picked)

        xyz = points[picked].copy()
        normal = arrays["normal"][picked].copy()
        rgb = arrays["rgb"][picked].copy() / 255.0
        class_id = arrays["class"][picked].copy()
        instance = arrays["instance"][picked].copy()

        if self.normalize:
            xyz, normal = self._normalise(xyz, normal)
        if self.augment:
            xyz, normal, rgb = self._augment(xyz, normal, rgb, rng)

        features = np.concatenate([xyz, normal, rgb], axis=1).astype(np.float32)
        return (
            torch.from_numpy(features),
            torch.from_numpy(class_id.astype(np.int64)),
            torch.from_numpy(instance.astype(np.int64)),
        )

    # ------------------------------------------------------------------ #

    def _pick_centre(self, scene, arrays, rng):
        """按类别均衡权重选中心点，保证稀有类别也能被采到。"""

        labels = arrays["class"]
        unique, weights, pools = self._class_sampling_probabilities(scene)
        chosen = int(rng.choice(len(unique), p=weights))
        return int(rng.choice(pools[int(unique[chosen])]))

    @staticmethod
    def _normalise(xyz, normal):
        """把块平移到质心、按半径缩放，并归一化法线。"""

        xyz = xyz - xyz.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(xyz, axis=1).max()
        if scale > 1e-6:
            xyz = xyz / scale
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / np.maximum(norm, 1e-6)
        return xyz, normal

    @staticmethod
    def _augment(xyz, normal, rgb, rng):
        """绕 Z 轴旋转 + 轻微缩放/抖动。

        只绕 Z 轴：世界系 Z 向上，绕 Z 旋转不改变"哪边是地板"，
        而任意轴旋转会把地板转成墙，属于错误的增强。
        """

        angle = rng.uniform(0.0, 2.0 * np.pi)
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz = xyz @ rotation.T
        normal = normal @ rotation.T

        xyz = xyz * rng.uniform(0.9, 1.1) + rng.normal(0.0, 0.002, size=xyz.shape).astype(np.float32)
        rgb = np.clip(rgb * rng.uniform(0.9, 1.1) + rng.normal(0.0, 0.02, size=rgb.shape), 0.0, 1.0)
        return xyz.astype(np.float32), normal.astype(np.float32), rgb.astype(np.float32)


def describe_manifest(dataset_root):
    """打印数据集概况，训练前先看一眼类别分布。"""

    manifest = json.loads((Path(dataset_root) / "manifest.json").read_text(encoding="utf-8"))
    print(f"场景 {len(manifest['scenes'])} 个 / 点 {manifest['total_points']:,} / "
          f"类别 {manifest['num_classes']}")
    for split, scenes in manifest["splits"].items():
        print(f"  {split:5s} {len(scenes):2d} 场景  {scenes}")
    histogram = sorted(manifest["class_histogram"].items(), key=lambda kv: -kv[1])
    total = max(manifest["total_points"], 1)
    print("  类别分布（前 8）：")
    for name, count in histogram[:8]:
        print("    %-24s %10d  (%5.2f%%)" % (name, count, count / total * 100))
    if len(histogram) > 8:
        tail = sum(count for _, count in histogram[8:])
        print("    %-24s %10d  (%5.2f%%)" % ("其余 %d 类" % (len(histogram) - 8), tail, tail / total * 100))
    return manifest
