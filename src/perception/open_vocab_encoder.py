"""开放词汇语义编码器：CLIP 图文双塔的统一封装。

供「实例嵌入提取」与「文本查询」两个工具复用，保证图文走同一套
预处理与归一化，余弦相似度才可比。

关键约束：服务器 torch 为 2.5.1，低于 transformers 5.x 所要求的 2.6
（CVE-2025-32434），因此加载权重必须显式 use_safetensors=True，
否则任何只提供 pytorch_model.bin 的仓库都会直接抛 ValueError。
"""

from __future__ import annotations

import torch
from transformers import CLIPModel, CLIPProcessor


def as_embedding_tensor(output):
    """transformers 各版本 get_*_features 的返回类型不一致，统一取出嵌入张量。"""

    if isinstance(output, torch.Tensor):
        return output
    for name in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError("无法从 %r 中取出嵌入张量" % type(output))


class OpenVocabularyEncoder:
    """CLIP 封装：图像与文本编码到同一 512 维空间，输出均已 L2 归一化。"""

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = (
            CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device).eval()
        )

    @property
    def embedding_dim(self):
        return int(self.model.config.projection_dim)

    @staticmethod
    def _normalize(features):
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.inference_mode()
    def encode_images(self, images, batch_size=32):
        """images: PIL.Image 序列，返回 (N, D) 的 CPU float32 张量。"""

        images = list(images)
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            features = as_embedding_tensor(self.model.get_image_features(**inputs)).float()
            chunks.append(self._normalize(features).cpu())
        if not chunks:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    @torch.inference_mode()
    def encode_texts(self, texts, batch_size=32):
        """texts: 字符串序列，返回 (N, D) 的 CPU float32 张量。"""

        texts = list(texts)
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(self.device)
            features = as_embedding_tensor(self.model.get_text_features(**inputs)).float()
            chunks.append(self._normalize(features).cpu())
        if not chunks:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float32)
        return torch.cat(chunks, dim=0)
