from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json
import re

import cv2
import numpy as np

from src.datasets.replica_sequence import ReplicaSequence
from src.perception.grounding_dino_detector import (
    GroundingDinoDetector,
)
from src.perception.sam2_box_segmenter import (
    Sam2BoxSegmenter,
)


MASK_COLORS_RGB = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 160, 255),
    (255, 180, 80),
    (220, 80, 255),
    (80, 255, 220),
    (255, 100, 180),
    (180, 255, 80),
]


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--scene-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--dino-model",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--sam-model",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "computer monitor",
            "chair",
            "desk",
            "trash can",
            "door",
        ],
    )

    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/perception/grounded_sam2"),
    )

    return parser.parse_args()


def create_mask_overlay(
    rgb,
    detections,
    instance_masks,
):
    overlay = rgb.astype(np.float32).copy()

    for instance_index, instance_mask in enumerate(instance_masks):
        color_rgb = np.array(
            MASK_COLORS_RGB[
                instance_index % len(MASK_COLORS_RGB)
            ],
            dtype=np.float32,
        )

        mask = instance_mask.mask

        overlay[mask] = (
            0.55 * overlay[mask]
            + 0.45 * color_rgb
        )

    annotated_bgr = cv2.cvtColor(
        overlay.astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    for instance_index, (detection, instance_mask) in enumerate(
        zip(detections, instance_masks)
    ):
        color_rgb = MASK_COLORS_RGB[
            instance_index % len(MASK_COLORS_RGB)
        ]
        color_bgr = tuple(reversed(color_rgb))

        x_min, y_min, x_max, y_max = map(
            int,
            detection.box_xyxy,
        )

        mask_uint8 = instance_mask.mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            annotated_bgr,
            contours,
            contourIdx=-1,
            color=color_bgr,
            thickness=2,
        )

        cv2.rectangle(
            annotated_bgr,
            (x_min, y_min),
            (x_max, y_max),
            color_bgr,
            thickness=2,
        )

        caption = (
            f"{instance_index:02d} "
            f"{detection.label} "
            f"D:{detection.score:.2f} "
            f"S:{instance_mask.predicted_iou:.2f}"
        )

        text_y = max(y_min - 8, 22)

        cv2.putText(
            annotated_bgr,
            caption,
            (x_min, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_bgr,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return annotated_bgr


def safe_filename(label):
    cleaned_label = re.sub(
        pattern=r"[^a-zA-Z0-9_-]+",
        repl="_",
        string=label,
    )

    return cleaned_label.strip("_").lower()


def main():
    arguments = parse_arguments()

    sequence = ReplicaSequence(arguments.scene_directory)
    frame = sequence[arguments.frame_index]

    detector = GroundingDinoDetector(
        model_id=arguments.dino_model
    )

    segmenter = Sam2BoxSegmenter(
        model_id_or_path=arguments.sam_model
    )

    detection_start = perf_counter()

    detections = detector.predict(
        rgb=frame["rgb"],
        text_queries=arguments.classes,
        box_threshold=arguments.box_threshold,
        text_threshold=arguments.text_threshold,
    )

    detection_time_ms = (
        perf_counter() - detection_start
    ) * 1000.0

    if not detections:
        raise RuntimeError(
            "Grounding DINO 没有产生检测框"
        )

    boxes_xyxy = np.stack(
        [
            detection.box_xyxy
            for detection in detections
        ]
    )

    segmentation_start = perf_counter()

    instance_masks = segmenter.predict(
        rgb=frame["rgb"],
        boxes_xyxy=boxes_xyxy,
    )

    segmentation_time_ms = (
        perf_counter() - segmentation_start
    ) * 1000.0

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    masks_directory = (
        arguments.output_directory
        / f"frame_{frame['frame_id']:06d}_masks"
    )
    masks_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay_path = (
        arguments.output_directory
        / f"frame_{frame['frame_id']:06d}_instances.jpg"
    )

    json_path = (
        arguments.output_directory
        / f"frame_{frame['frame_id']:06d}_instances.json"
    )

    annotated_image = create_mask_overlay(
        rgb=frame["rgb"],
        detections=detections,
        instance_masks=instance_masks,
    )

    cv2.imwrite(
        str(overlay_path),
        annotated_image,
    )

    serialized_instances = []

    for instance_index, (detection, instance_mask) in enumerate(
        zip(detections, instance_masks)
    ):
        label_name = safe_filename(detection.label)

        mask_path = (
            masks_directory
            / f"{instance_index:02d}_{label_name}.png"
        )

        cv2.imwrite(
            str(mask_path),
            instance_mask.mask.astype(np.uint8) * 255,
        )

        serialized_instances.append(
            {
                "local_instance_id": instance_index,
                "label": detection.label,
                "detection_score": detection.score,
                "sam_predicted_iou": (
                    instance_mask.predicted_iou
                ),
                "area_pixels": instance_mask.area_pixels,
                "box_xyxy": detection.box_xyxy.tolist(),
                "mask_path": str(mask_path),
            }
        )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            serialized_instances,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n检测实例数：{len(detections)}")
    print(f"Grounding DINO：{detection_time_ms:.2f} ms")
    print(f"SAM2：{segmentation_time_ms:.2f} ms")

    for instance_index, (detection, instance_mask) in enumerate(
        zip(detections, instance_masks)
    ):
        print(
            f"{instance_index:02d} | "
            f"{detection.label:18s} | "
            f"DINO={detection.score:.3f} | "
            f"SAM={instance_mask.predicted_iou:.3f} | "
            f"Area={instance_mask.area_pixels}"
        )

    print(f"\n实例预览：{overlay_path.resolve()}")
    print(f"实例信息：{json_path.resolve()}")
    print(f"二值掩码目录：{masks_directory.resolve()}")


if __name__ == "__main__":
    main()