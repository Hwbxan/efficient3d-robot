from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import cv2

from src.datasets.replica_sequence import ReplicaSequence
from src.perception.grounding_dino_detector import (
    GroundingDinoDetector,
)


BOX_COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 160, 255),
    (255, 180, 80),
    (220, 80, 255),
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
        default=0.25,
    )

    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/perception"),
    )

    parser.add_argument(
        "--model-id-or-path",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face 模型名称或本地模型目录",
    )

    return parser.parse_args()


def draw_detections(rgb, detections):
    annotated_bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    for detection_index, detection in enumerate(detections):
        color = BOX_COLORS[
            detection_index % len(BOX_COLORS)
        ]

        x_min, y_min, x_max, y_max = detection.box_xyxy
        x_min, y_min, x_max, y_max = map(
            int,
            [x_min, y_min, x_max, y_max],
        )

        caption = (
            f"{detection.label}: "
            f"{detection.score:.2f}"
        )

        cv2.rectangle(
            annotated_bgr,
            (x_min, y_min),
            (x_max, y_max),
            color,
            thickness=3,
        )

        text_size, _ = cv2.getTextSize(
            caption,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )

        text_width, text_height = text_size
        text_top = max(y_min - text_height - 12, 0)

        cv2.rectangle(
            annotated_bgr,
            (x_min, text_top),
            (x_min + text_width + 8, y_min),
            color,
            thickness=-1,
        )

        cv2.putText(
            annotated_bgr,
            caption,
            (x_min + 4, y_min - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return annotated_bgr


def main():
    arguments = parse_arguments()

    sequence = ReplicaSequence(arguments.scene_directory)
    frame = sequence[arguments.frame_index]

    detector = GroundingDinoDetector(
    model_id=arguments.model_id_or_path
)

    start_time = perf_counter()

    detections = detector.predict(
        rgb=frame["rgb"],
        text_queries=arguments.classes,
        box_threshold=arguments.box_threshold,
        text_threshold=arguments.text_threshold,
    )

    inference_time_ms = (
        perf_counter() - start_time
    ) * 1000.0

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path = (
        arguments.output_directory
        / f"frame_{frame['frame_id']:06d}_detection.jpg"
    )

    json_path = (
        arguments.output_directory
        / f"frame_{frame['frame_id']:06d}_detection.json"
    )

    annotated_image = draw_detections(
        rgb=frame["rgb"],
        detections=detections,
    )

    cv2.imwrite(
        str(image_path),
        annotated_image,
    )

    serialized_detections = [
        {
            "label": detection.label,
            "score": detection.score,
            "box_xyxy": detection.box_xyxy.tolist(),
        }
        for detection in detections
    ]

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            serialized_detections,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n检测类别：{arguments.classes}")
    print(f"检测数量：{len(detections)}")
    print(f"推理时间：{inference_time_ms:.2f} ms")

    for detection in detections:
        print(
            f"{detection.label:20s} | "
            f"{detection.score:.3f} | "
            f"{detection.box_xyxy.round(1)}"
        )

    print(f"\n检测图片：{image_path.resolve()}")
    print(f"检测结果：{json_path.resolve()}")


if __name__ == "__main__":
    main()