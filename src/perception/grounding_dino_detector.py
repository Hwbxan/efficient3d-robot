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
    """Grounding DINO 开放词汇目标检测器。

    默认启用"逐类别解析标签"：processor 自带的短语解码会把跨多个类别的
    token 跨度拼成一个复合标签（如 "chair desk trash"），这种标签无法与任何
    已知类别比较。这里改为在同一份 logits 上按类别取最大概率再取 argmax，
    得到单一类别，且不增加前向传播次数。
    """

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

    def category_token_groups(self, prompt: str, categories: List[str]):
        """把 prompt 中的 token 归属到各个类别。

        prompt 由 ". ".join(categories) + "." 构造，因此每个类别的字符区间
        是已知的；再用分词器的 offset_mapping 把 token 映射到字符区间。
        """

        encoded = self.processor.tokenizer(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = encoded["offset_mapping"]

        if any(offset is None for offset in offsets):
            raise RuntimeError("分词器未返回 offset_mapping，无法解析类别 token")

        ranges = {}
        cursor = 0
        for name in categories:
            ranges[name] = (cursor, cursor + len(name))
            cursor += len(name) + 2

        groups = []
        for name, (start, end) in ranges.items():
            indices = [
                index for index, (token_start, token_end) in enumerate(offsets)
                if token_start < end and token_end > start
            ]
            if not indices:
                raise RuntimeError(f"类别 {name!r} 未匹配到任何 token")
            groups.append(indices)

        return encoded["input_ids"], groups

    @staticmethod
    def normalized_to_xyxy(normalized_boxes, image_size):
        """把归一化的 cxcywh 转成像素 xyxy。"""

        width, height = image_size
        center_x, center_y, box_width, box_height = normalized_boxes.unbind(-1)
        return torch.stack(
            [
                (center_x - box_width / 2.0) * width,
                (center_y - box_height / 2.0) * height,
                (center_x + box_width / 2.0) * width,
                (center_y + box_height / 2.0) * height,
            ],
            dim=-1,
        )

    def detections_from_phrases(
        self, outputs, inputs, image, box_threshold, text_threshold,
        max_box_area_fraction=1.0,
    ) -> List[Detection]:
        """原有路径：使用 processor 的短语解码（保留作为回退）。"""

        processed = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        detections = [
            Detection(
                label=str(label),
                score=float(score.item()),
                box_xyxy=box.detach().cpu().numpy().astype(np.float32),
            )
            for box, score, label in zip(
                processed["boxes"], processed["scores"], processed["labels"]
            )
        ]

        if max_box_area_fraction < 1.0:
            image_area = float(image.size[0] * image.size[1])
            detections = [
                detection for detection in detections
                if (
                    (detection.box_xyxy[2] - detection.box_xyxy[0])
                    * (detection.box_xyxy[3] - detection.box_xyxy[1])
                ) / image_area <= max_box_area_fraction
            ]
        return detections

    @torch.inference_mode()
    def predict(
        self,
        rgb: np.ndarray,
        text_queries: List[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        resolve_labels: bool = True,
        max_box_area_fraction: float = 0.40,
    ) -> List[Detection]:
        """根据文本类别检测图像中的目标。

        resolve_labels=True 时，每个框的标签取"该框在各类别 token 上
        最大概率"最高的类别，避免出现复合标签。

        max_box_area_fraction 用于剔除"背景块"误检：Grounding DINO 对
        短类别词（desk / door / computer monitor）会给出覆盖半张图的大框，
        且分数不低（0.36~0.78）、类别间间隔也不小，靠分数阈值分不掉
        ——实测提高分数阈值反而单调降低 AP。

        在 Replica office0 上（61 帧、209 个 GT 实例）实测：
        阈值 0.50 丢 8 个 FP、0 个 TP；0.40 丢 26 个 FP、0 个 TP
        （AP@0.25 0.5821->0.6252，AP@0.50 0.2581->0.2843，召回不变）；
        0.30 起开始丢 TP（6 个）。故 0.40 是"零召回代价"的最大过滤强度。
        设为 1.0 关闭该过滤。
        """

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

        if not resolve_labels:
            return self.detections_from_phrases(
                outputs, inputs, image, box_threshold, text_threshold,
                max_box_area_fraction,
            )

        categories = [query.lower() for query in text_queries]
        if any("." in query for query in categories):
            raise ValueError("类别名不能包含句点，否则无法定位 token 区间")
        prompt = ". ".join(categories) + "."

        prompt_ids, token_groups = self.category_token_groups(prompt, categories)
        if prompt_ids != inputs["input_ids"][0].tolist():
            raise RuntimeError(
                "自行构造的 prompt 与 processor 的 input_ids 不一致，"
                "无法安全解析类别标签；请改用 resolve_labels=False"
            )

        probabilities = outputs.logits[0].sigmoid()
        masked = torch.where(
            probabilities >= text_threshold,
            probabilities,
            torch.zeros_like(probabilities),
        )

        box_scores = masked.max(dim=-1).values
        category_scores = torch.stack(
            [masked[:, indices].max(dim=-1).values for indices in token_groups],
            dim=-1,
        )
        best_score, best_index = category_scores.max(dim=-1)

        # best_score 为 0 表示没有任何类别 token 通过 text_threshold。
        keep = (box_scores > box_threshold) & (best_score > 0)

        boxes = self.normalized_to_xyxy(outputs.pred_boxes[0], image.size)

        if max_box_area_fraction < 1.0:
            image_area = float(image.size[0] * image.size[1])
            box_width = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
            box_height = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
            area_fraction = (box_width * box_height) / image_area
            keep = keep & (area_fraction <= max_box_area_fraction)

        detections = []
        for index in torch.nonzero(keep).flatten().tolist():
            detections.append(
                Detection(
                    label=text_queries[int(best_index[index])],
                    score=float(box_scores[index]),
                    box_xyxy=boxes[index].detach().cpu().numpy().astype(np.float32),
                )
            )

        return detections
