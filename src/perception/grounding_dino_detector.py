from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)


@dataclass(frozen=True)
class Detection:
    """一个开放词汇检测结果。"""

    label: str
    score: float
    box_xyxy: np.ndarray


class GroundingDinoDetector:
    """Grounding DINO 开放词汇目标检测器。"""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model_id = model_id

        print(f"正在加载模型：{model_id}")
        print(f"推理设备：{self.device}")

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def predict(
        self,
        rgb: np.ndarray,
        text_queries: List[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> List[Detection]:
        """根据文本类别检测图像中的目标。"""

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"RGB 图像应为 H×W×3，实际为 {rgb.shape}"
            )

        if not text_queries:
            raise ValueError("text_queries 不能为空")

        image = Image.fromarray(rgb)

        # 外层列表表示 batch 中只有一张图像。
        batched_text_queries = [text_queries]

        inputs = self.processor(
            images=image,
            text=batched_text_queries,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        outputs = self.model(**inputs)

        processed_results = (
            self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )
        )

        result = processed_results[0]
        detections = []

        for box, score, label in zip(
            result["boxes"],
            result["scores"],
            result["labels"],
        ):
            detections.append(
                Detection(
                    label=str(label),
                    score=float(score.item()),
                    box_xyxy=(
                        box.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    ),
                )
            )

        return detections