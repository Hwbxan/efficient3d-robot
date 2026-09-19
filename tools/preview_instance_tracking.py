import colorsys
import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from src.datasets.replica_sequence import ReplicaSequence


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def global_id_color(global_id):
    """相同全局 ID 始终返回相同的 RGB 颜色。"""

    if global_id is None:
        return np.array([180, 180, 180], dtype=np.uint8)

    hue = (global_id * 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.75, 1.0)

    return np.round(np.asarray(rgb) * 255).astype(np.uint8)


def read_mask(metadata, json_path, expected_shape):
    mask_path = Path(metadata["mask_path"])

    # 兼容之前保存的“相对项目根目录”路径。
    if not mask_path.is_absolute() and not mask_path.exists():
        mask_path = json_path.parent / mask_path

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"无法读取掩码：{mask_path}")

    if mask.shape != expected_shape:
        raise ValueError(
            f"掩码尺寸不匹配：{mask.shape} 与 {expected_shape}"
        )

    return mask > 0


def draw_frame(rgb, metadata, associations, json_path):
    association_by_local_id = {
        item["local_instance_id"]: item
        for item in associations
    }

    overlay = rgb.astype(np.float32).copy()
    annotations = []

    for item in metadata:
        local_id = item["local_instance_id"]
        association = association_by_local_id.get(local_id)

        if association is None:
            global_id = None
            id_text = "SKIPPED"
        else:
            global_id = association["global_id"]
            id_text = (
                f"G{global_id:03d}"
                if global_id is not None
                else "PENDING"
            )

        color_rgb = global_id_color(global_id)
        mask = read_mask(item, json_path, rgb.shape[:2])

        overlay[mask] = (
            0.60 * overlay[mask]
            + 0.40 * color_rgb
        )

        annotations.append(
            (item, mask, color_rgb, id_text)
        )

    image_bgr = cv2.cvtColor(
        overlay.astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    image_height, image_width = rgb.shape[:2]

    for item, mask, color_rgb, id_text in annotations:
        color_bgr = tuple(int(value) for value in color_rgb[::-1])

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            image_bgr,
            contours,
            -1,
            color_bgr,
            2,
        )

        # 展示当前帧原始标签，方便发现标签变化。
        caption = f"{id_text} {item['label']}"

        (text_width, text_height), baseline = cv2.getTextSize(
            caption,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )

        x = int(item["box_xyxy"][0])
        y = int(item["box_xyxy"][1]) - 8

        x = max(4, min(x, image_width - text_width - 8))
        y = max(text_height + 8, min(y, image_height - baseline - 8))

        cv2.rectangle(
            image_bgr,
            (x - 3, y - text_height - 4),
            (x + text_width + 3, y + baseline + 3),
            (25, 25, 25),
            -1,
        )

        cv2.putText(
            image_bgr,
            caption,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color_bgr,
            2,
            cv2.LINE_AA,
        )

    return image_bgr


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"图片保存失败：{path}")


def main():
    parser = ArgumentParser()

    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--tracking-json", type=Path, required=True)

    parser.add_argument(
        "--mask-directory",
        type=Path,
        default=Path("outputs/perception/grounded_sam2"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/association/preview"),
    )

    arguments = parser.parse_args()

    sequence = ReplicaSequence(arguments.scene_directory)
    tracking = load_json(arguments.tracking_json)

    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    tiles = []

    for frame_record in tracking["frames"]:
        frame_index = frame_record["frame_index"]
        frame = sequence[frame_index]

        json_path = (
            arguments.mask_directory
            / f"frame_{frame['frame_id']:06d}_instances.json"
        )

        image = draw_frame(
            rgb=frame["rgb"],
            metadata=load_json(json_path),
            associations=frame_record["associations"],
            json_path=json_path,
        )

        output_path = (
            arguments.output_directory
            / f"frame_{frame['frame_id']:06d}_global_ids.png"
        )
        save_image(output_path, image)

        # 每帧缩小后添加标题，用于生成对比图。
        tile = cv2.resize(image, (600, 340))
        tile = cv2.copyMakeBorder(
            tile, 36, 0, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )

        cv2.putText(
            tile,
            f"Frame {frame_index:06d}",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        tiles.append(tile)
        print(f"已保存：{output_path.resolve()}")

    if not tiles:
        raise ValueError("关联记录中没有帧")

    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))

    rows = [
        np.concatenate(tiles[index:index + 2], axis=1)
        for index in range(0, len(tiles), 2)
    ]

    contact_sheet = np.concatenate(rows, axis=0)
    contact_sheet_path = (
        arguments.output_directory / "tracking_contact_sheet.png"
    )

    save_image(contact_sheet_path, contact_sheet)
    print(f"\n对比图：{contact_sheet_path.resolve()}")


if __name__ == "__main__":
    main()