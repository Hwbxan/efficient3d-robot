"""为 RKNN 的 INT8 量化生成标定集。

rknn-toolkit2 要的是一个 `dataset.txt`，里面每行是一个 `.npy` 的路径，
每个 npy 是一份**输入**样本（形状与 ONNX 输入一致，含 batch 维）。

标定集的质量直接决定 INT8 掉多少点，所以这里**用真实数据**：
直接从训练集划分里采样局部点块，走和训练完全一样的归一化与特征拼接，
只是关掉增强（增强会引入训练时没有的分布）。

为什么不用随机噪声：随机噪声的激活范围与真实点云完全不同，
量化参数会按错误的动态范围去定，这是 INT8 掉点最常见的坑。

用法（项目根目录）：
    python -m tools.build_rknn_calibration_set \
        --dataset-root outputs/point_dataset \
        --output-directory outputs/rknn_calibration \
        --num-points 2048 --count 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.point_dataset import PointCloudPatchDataset  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", default="outputs/point_dataset")
    parser.add_argument("--output-directory", default="outputs/rknn_calibration")
    parser.add_argument("--split", default="train",
                        help="从哪个划分采样。**必须用 train**，用 val/test 等于偷看")
    parser.add_argument("--num-points", type=int, default=2048,
                        help="必须与要转换的 ONNX 输入长度一致")
    parser.add_argument("--count", type=int, default=200,
                        help="标定样本数。官方建议 100–500，太少量化参数不稳")
    parser.add_argument("--patches-per-scene", type=int, default=64)
    parser.add_argument("--class-alpha", type=float, default=0.5,
                        help="与训练同口径的类别均衡采样")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_arguments()
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("生成 RKNN INT8 标定集")
    print("=" * 78)

    dataset = PointCloudPatchDataset(
        args.dataset_root,
        split=args.split,
        num_points=args.num_points,
        patches_per_scene=args.patches_per_scene,
        class_alpha=args.class_alpha,
        augment=False,          # 标定必须关增强
        seed=args.seed,
        normalize=True,         # 与训练、与部署图一致的归一化
    )
    print(f"  采样划分：{args.split}（{len(dataset.scenes)} 个场景）")
    print(f"  每样本点数：{args.num_points}   目标样本数：{args.count}")

    if args.split != "train":
        print(f"  ⚠️  警告：从 {args.split} 采样做标定，等于用验证/测试数据定量化参数。")
        print("      正式结果请用 --split train。")

    lines = []
    shapes = set()
    for index in range(args.count):
        # 循环取，样本数超过数据集长度时继续复用（每轮 rng 种子不同，内容会变）
        features, _, _ = dataset[index % len(dataset)]
        array = features.numpy().astype(np.float32)[None, ...]   # 补 batch 维
        shapes.add(tuple(array.shape))
        path = out / f"calib_{index:04d}.npy"
        np.save(path, array)
        lines.append(str(path.resolve()))

    if len(shapes) != 1:
        print(f"  [FAIL] 采样出的形状不一致：{sorted(shapes)}")
        return 1
    shape = shapes.pop()

    dataset_list = out / "dataset.txt"
    dataset_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 顺手记下特征统计 —— 量化掉点严重时，先看这里是不是分布偏了
    sample = np.load(lines[0])
    stats = {
        "count": len(lines),
        "input_shape": list(shape),
        "split": args.split,
        "num_points": args.num_points,
        "feature_mean": [round(float(v), 5) for v in sample.reshape(-1, shape[-1]).mean(axis=0)],
        "feature_std": [round(float(v), 5) for v in sample.reshape(-1, shape[-1]).std(axis=0)],
        "feature_min": [round(float(v), 5) for v in sample.reshape(-1, shape[-1]).min(axis=0)],
        "feature_max": [round(float(v), 5) for v in sample.reshape(-1, shape[-1]).max(axis=0)],
        "note": "特征顺序为 xyz(3) + normal(3) + rgb(3)，xyz 已按训练口径中心化并缩放到单位球",
    }
    (out / "calibration_manifest.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  形状：{shape}   特征维：{shape[-1]}（xyz + normal + rgb）")
    print(f"  已写出 {len(lines)} 个 npy + dataset.txt")
    print(f"  特征范围（每维 min）：{stats['feature_min']}")
    print(f"  特征范围（每维 max）：{stats['feature_max']}")
    print(f"\n标定清单：{dataset_list}")
    print("下一步：")
    print(f"  python -m tools.rknn_toolkit_probe --onnx deploy/point_encoder_rknn_deploy.onnx \\")
    print(f"      --output deploy/point_encoder_rk3588_int8.rknn \\")
    print(f"      --calibration-dataset {dataset_list}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
