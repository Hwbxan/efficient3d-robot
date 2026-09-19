from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.replica_sequence import ReplicaSequence


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--scene-directory",
        type=Path,
        required=True,
        help="Replica 场景目录",
    )

    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="需要检查的帧序号",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data_check/frame_preview.png"),
        help="预览图保存位置",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    sequence = ReplicaSequence(arguments.scene_directory)
    frame = sequence[arguments.frame_index]

    rgb = frame["rgb"]
    depth_m = frame["depth_m"]
    camera_to_world = frame["camera_to_world"]

    valid_depth_mask = np.isfinite(depth_m) & (depth_m > 0)
    valid_depth = depth_m[valid_depth_mask]

    print(f"序列帧数：{len(sequence)}")
    print(f"当前帧编号：{frame['frame_id']}")
    print(f"RGB 形状：{rgb.shape}")
    print(f"Depth 形状：{depth_m.shape}")
    print(f"Depth 类型：{depth_m.dtype}")
    print(
        f"有效深度范围："
        f"{valid_depth.min():.3f} ～ {valid_depth.max():.3f} 米"
    )

    print("\n相机内参：")
    print(frame["camera_matrix"])

    print("\nCamera-to-World 位姿：")
    print(camera_to_world)

    depth_display_max = np.percentile(valid_depth, 99)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    depth_plot = axes[1].imshow(
        depth_m,
        cmap="turbo",
        vmin=0,
        vmax=depth_display_max,
    )
    axes[1].set_title("Depth (meters)")
    axes[1].axis("off")

    colorbar = figure.colorbar(depth_plot, ax=axes[1])
    colorbar.set_label("Depth (m)")

    figure.tight_layout()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=160)
    plt.close(figure)

    print(f"\n预览图已保存：{arguments.output.resolve()}")


if __name__ == "__main__":
    main()