from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor


@dataclass(frozen=True)
class InstanceMask:
    """SAM2 生成的单个实例掩码。"""

    mask: np.ndarray
    predicted_iou: float
    area_pixels: int


class Sam2BoxSegmenter:
    """使用检测框提示生成实例掩码。"""

    def __init__(
        self,
        model_id_or_path: str,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        print(f"正在加载 SAM2：{model_id_or_path}")
        print(f"SAM2 推理设备：{self.device}")

        self.processor = Sam2Processor.from_pretrained(
            model_id_or_path
        )

        self.model = (
            Sam2Model
            .from_pretrained(model_id_or_path)
            .to(self.device)
            .eval()
        )

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )

        return nullcontext()

    @torch.inference_mode()
    def predict(
        self,
        rgb: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> List[InstanceMask]:
        """为每个 XYXY 检测框生成一个掩码。"""

        if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
            raise ValueError(
                f"boxes_xyxy 应为 N×4，实际为 {boxes_xyxy.shape}"
            )

        if len(boxes_xyxy) == 0:
            return []

        image = Image.fromarray(rgb)

        # 维度结构：[图像][物体][x1, y1, x2, y2]
        input_boxes = [
            boxes_xyxy.astype(np.float32).tolist()
        ]

        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        with self._autocast_context():
            outputs = self.model(
                **inputs,
                multimask_output=False,
            )

        processed_masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )[0]

        predicted_ious = (
            outputs.iou_scores[0, :, 0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        instance_masks = []

        for object_index in range(len(boxes_xyxy)):
            mask = (
                processed_masks[object_index, 0]
                .numpy()
                .astype(bool)
            )

            instance_masks.append(
                InstanceMask(
                    mask=mask,
                    predicted_iou=float(
                        predicted_ious[object_index]
                    ),
                    area_pixels=int(mask.sum()),
                )
            )

        return instance_masks