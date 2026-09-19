"""轻量点式 3D 骨干：逐点语义 + 实例判别嵌入 + 开放词汇投影。

设计取舍（对应 `stage5_design.md` §4）：

1. **不引入稀疏卷积。** 服务器没装 spconv / torchsparse / MinkowskiEngine，
   在 CUDA 12.8 + torch 2.5.1 上编译它们风险高、收益不确定。点式方案零额外依赖，
   而且天然处理在线融合点云的**变密度**问题（近处密、远处疏），
   体素卷积还得额外处理空体素。
2. **不用 FPS（最远点采样）。** 纯 PyTorch 的 FPS 是 O(N·M)，
   20k 点上要跑好几秒。这里改用 **Morton（Z-order）序 + 等距抽稀**：
   把点按空间填充曲线排序后每隔固定步长取一个，空间覆盖同样均匀，
   但代价只有一次排序，且**点数精确可控**（利于固定形状的 batch）。
3. **固定点数样本。** 每个样本固定 P 个点，各级降采样点数整除，
   于是 batch 是稠密张量 `(B, P, C)`——不需要 padding/mask，
   kNN 也不必担心跨样本串门。整场景推理时按块切分（见文末）。

注意：解码器（FP 层）会把特征上采样回**全部原始点**，
所以降采样只影响上下文聚合，不会永久丢点。

用法：
    from src.models.point_encoder import PointEncoderConfig, build_point_encoder
    model = build_point_encoder(num_classes=40)
    out = model(features)          # features: (B, P, 9) = xyz + normal + rgb
    out.semantic_logits            # (B, P, num_classes)
    out.instance_embedding         # (B, P, 16)
    out.text_embedding             # (B, P, 512)，已 L2 归一化
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 邻域搜索 / 采样工具：全部纯 PyTorch，无第三方扩展依赖
# --------------------------------------------------------------------------- #


def knn_indices(support, query, k, chunk=512):
    """为每个 query 点在 support 中找 k 个最近邻，返回 `(B, Q, k)` 索引。

    **必须分块**：B=4、Q=8192、N=8192 时，全量 `cdist` 会生成
    2.7e8 个 float（>1 GB），显存直接爆掉。按 query 分块后峰值降到 1/chunk。
    """

    if support.shape[0] != query.shape[0]:
        raise ValueError("support 与 query 的 batch 维必须一致")
    num_support = support.shape[1]
    k = min(k, num_support)

    collected = []
    for start in range(0, query.shape[1], chunk):
        block = query[:, start:start + chunk]                     # (B, q, 3)
        distance = torch.cdist(block, support)                    # (B, q, N)
        collected.append(distance.topk(k, dim=-1, largest=False).indices)
    return torch.cat(collected, dim=1)


def gather_features(features, indices):
    """按索引收集特征：`(B, N, C)` + `(B, Q, k)` → `(B, Q, k, C)`。"""

    B, _, C = features.shape
    Q, k = indices.shape[1], indices.shape[2]
    flat = indices.reshape(B, -1).unsqueeze(-1).expand(-1, -1, C)
    return torch.gather(features, 1, flat).reshape(B, Q, k, C)


def _morton_codes(xyz, bits=10):
    """把坐标量化后交错比特，得到 Morton（Z-order）码 `(B, N)`。"""

    low = xyz.amin(dim=1, keepdim=True)
    high = xyz.amax(dim=1, keepdim=True)
    span = (high - low).clamp(min=1e-9)
    limit = (1 << bits) - 1
    quantised = ((xyz - low) / span * limit).round().long().clamp(0, limit)

    code = torch.zeros_like(quantised[..., 0])
    for axis in range(3):
        axis_bits = quantised[..., axis]
        for bit in range(bits):
            code = code | (((axis_bits >> bit) & 1) << (3 * bit + axis))
    return code


def morton_stride_indices(xyz, count):
    """按 Morton 序等距抽取 `count` 个点，返回 `(B, count)` 索引。

    比随机采样空间覆盖更均匀，比 FPS 便宜几个数量级，
    而且**点数精确等于 count**，使各级张量形状固定、可整除。
    """

    total = xyz.shape[1]
    if count >= total:
        return torch.arange(total, device=xyz.device).unsqueeze(0).expand(xyz.shape[0], -1)

    order = torch.argsort(_morton_codes(xyz), dim=1)              # (B, N)
    picks = torch.linspace(0, total - 1, count, device=xyz.device).round().long()
    return torch.gather(order, 1, picks.unsqueeze(0).expand(xyz.shape[0], -1))


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass
class PointEncoderConfig:
    """默认配置约 1.0 M 参数（对应设计文档的 1–2 M 目标）。

    索引约定：`level_channels[k]` 是第 k 级 SA 的输出宽度（0 最细）；
    `decoder_channels[s]` 是第 s 个 FP 层的输出宽度，**s=0 从最粗一级开始**。
    所以默认值 (256, 128, 128) 对应 FP3→256、FP2→128、FP1→128，
    最终逐点特征宽度是 `decoder_channels[-1]`。
    """

    in_channels: int = 9                 # xyz(3) + normal(3) + rgb(3)
    num_classes: int = 40
    stem_channels: int = 64
    level_channels: Sequence[int] = (128, 256, 512)
    level_k: Sequence[int] = (16, 16, 12)
    downsample_stride: int = 4
    decoder_channels: Sequence[int] = (256, 128, 128)
    instance_dim: int = 16
    text_dim: int = 512
    dropout: float = 0.0

    def __post_init__(self):
        if len(self.level_channels) != len(self.level_k):
            raise ValueError("level_channels 与 level_k 长度必须一致")
        if len(self.decoder_channels) != len(self.level_channels):
            raise ValueError("decoder_channels 必须与 level_channels 等长")


# --------------------------------------------------------------------------- #
# 模块
# --------------------------------------------------------------------------- #


class SetAbstraction(nn.Module):
    """降采样 + kNN 邻域聚合（PointNet++ SA，两层 MLP + max pool）。"""

    def __init__(self, in_channels, out_channels, k, stride):
        super().__init__()
        self.k = k
        self.stride = stride
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),             # +3 是相对坐标
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, xyz, features):
        total = xyz.shape[1]
        count = max(1, total // self.stride)
        picked = morton_stride_indices(xyz, count)                # (B, M)
        centres = torch.gather(xyz, 1, picked.unsqueeze(-1).expand(-1, -1, 3))

        neighbours = knn_indices(xyz, centres, self.k)            # (B, M, k)
        neighbour_xyz = gather_features(xyz, neighbours)           # (B, M, k, 3)
        neighbour_feat = gather_features(features, neighbours)     # (B, M, k, C)

        relative = neighbour_xyz - centres.unsqueeze(2)
        hidden = self.mlp(torch.cat([relative, neighbour_feat], dim=-1))
        return centres, hidden.max(dim=2).values, picked


class FeaturePropagation(nn.Module):
    """上采样 + skip 融合（PointNet++ FP，3-NN 反距离插值）。"""

    def __init__(self, coarse_channels, skip_channels, out_channels, k=3, dropout=0.0):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(coarse_channels + skip_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, xyz_coarse, feat_coarse, xyz_fine, feat_skip):
        neighbours = knn_indices(xyz_coarse, xyz_fine, self.k)     # (B, Q, k)
        neighbour_xyz = gather_features(xyz_coarse, neighbours)
        neighbour_feat = gather_features(feat_coarse, neighbours)

        distance = (neighbour_xyz - xyz_fine.unsqueeze(2)).norm(dim=-1)
        weight = 1.0 / (distance + 1e-8)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        interpolated = (neighbour_feat * weight.unsqueeze(-1)).sum(dim=2)

        return self.mlp(torch.cat([interpolated, feat_skip], dim=-1))


@dataclass
class PointEncoderOutput:
    semantic_logits: torch.Tensor          # (B, P, num_classes)
    instance_embedding: torch.Tensor       # (B, P, instance_dim)
    text_embedding: torch.Tensor           # (B, P, text_dim)，L2 归一化
    point_features: torch.Tensor           # (B, P, decoder_channels[-1])


class PointEncoder(nn.Module):
    """点式骨干：3 级 SA 编码 + 3 级 FP 解码 + 三个任务头。"""

    def __init__(self, config: PointEncoderConfig | None = None):
        super().__init__()
        self.config = config or PointEncoderConfig()
        cfg = self.config

        self.stem = nn.Sequential(
            nn.Linear(cfg.in_channels, cfg.stem_channels),
            nn.LayerNorm(cfg.stem_channels),
            nn.GELU(),
        )

        self.encoders = nn.ModuleList()
        in_channels = cfg.stem_channels
        for out_channels, k in zip(cfg.level_channels, cfg.level_k):
            self.encoders.append(
                SetAbstraction(in_channels, out_channels, k, cfg.downsample_stride)
            )
            in_channels = out_channels

        self.decoders = nn.ModuleList()
        coarse_channels = cfg.level_channels[-1]
        # step=0 从最粗一级开始上采样，skip 取该级编码器的输入（最细一级取 stem）
        for step in range(len(cfg.level_channels)):
            encoder_index = len(cfg.level_channels) - 1 - step
            skip_channels = (
                cfg.level_channels[encoder_index - 1]
                if encoder_index > 0
                else cfg.stem_channels
            )
            out_channels = cfg.decoder_channels[step]
            self.decoders.append(
                FeaturePropagation(coarse_channels, skip_channels, out_channels, dropout=cfg.dropout)
            )
            coarse_channels = out_channels

        final_channels = cfg.decoder_channels[-1]
        self.semantic_head = nn.Linear(final_channels, cfg.num_classes)
        self.instance_head = nn.Linear(final_channels, cfg.instance_dim)
        self.text_head = nn.Linear(final_channels, cfg.text_dim)

    def forward(self, features, xyz=None):
        """features: `(B, P, in_channels)`，前 3 通道视为 xyz。"""

        if features.dim() != 3:
            raise ValueError(f"期望 (B, P, C) 的输入，收到 {tuple(features.shape)}")
        if xyz is None:
            xyz = features[..., :3]

        stem_features = self.stem(features)

        # centres[k] / level_features[k]：k=0 是最细一级（stem 的输出），
        # k=1..L 依次是各级 SA 的输出。两者长度都是 L+1。
        centres = [xyz]
        level_features = [stem_features]
        current_xyz, current_features = xyz, stem_features
        for encoder in self.encoders:
            current_xyz, current_features, _ = encoder(current_xyz, current_features)
            centres.append(current_xyz)
            level_features.append(current_features)

        decoded = level_features[-1]
        for step, decoder in enumerate(self.decoders):
            encoder_index = len(self.decoders) - 1 - step
            decoded = decoder(
                centres[encoder_index + 1], decoded,          # 粗：上一级 SA 的输出
                centres[encoder_index], level_features[encoder_index],  # 细：该级 SA 的输入
            )

        return PointEncoderOutput(
            semantic_logits=self.semantic_head(decoded),
            instance_embedding=self.instance_head(decoded),
            text_embedding=F.normalize(self.text_head(decoded), dim=-1),
            point_features=decoded,
        )


def build_point_encoder(**overrides):
    """便捷构造：`build_point_encoder(num_classes=40, dropout=0.1)`。"""

    return PointEncoder(PointEncoderConfig(**overrides))


def parameter_count(model):
    """返回 (全部参数, 可训练参数)，便于 Stage 6–9 记录压缩基线。"""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------------------- #
# 整场景推理
# --------------------------------------------------------------------------- #
#
# 训练用固定 P 个点的样本；推理时整场景可能有几十万点，不能一次性喂进去
# （kNN 的 (B, Q, N) 会爆显存）。这里按块切分：
#
#   1. 把场景点云按 Morton 序排序（天然让空间相邻的点落在相邻索引）；
#   2. 切成 P 点的块，块间留重叠；
#   3. 每块前向，**先写先得**——后一块只填还没被写过的点；
#   4. 语义 logits / 嵌入按原索引写回。
#
# "先写先得"而不是"后写覆盖"：排序后每块的**开头**是紧接上一块结尾的点，
# 让先来的块负责它们，可以保证每个点都由"以它附近为中心"的块处理，
# 而不是由它恰好落在尾部的那个块处理（尾部的点邻域信息最少）。


@torch.no_grad()
def encode_cloud_in_blocks(model, features, num_points=8192, overlap=0.1,
                           device=None, batch_size=1):
    """整场景分块推理。

    `features`: `(N, C)` 或 `(1, N, C)` 的 tensor/ndarray，前 3 通道是 xyz。
    返回 `(semantic_logits, instance_embedding, text_embedding)`，形状 `(N, ...)`。
    返回 None 的那一路说明模型没有该头。
    """

    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap 必须在 [0, 1) 内")

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    if features.dim() == 3:
        if features.shape[0] != 1:
            raise ValueError("分块推理只接受单个场景（batch=1）")
        features = features[0]
    if features.dim() != 2:
        raise ValueError(f"期望 (N, C) 或 (1, N, C)，收到 {tuple(features.shape)}")

    device = device or next(model.parameters()).device
    features = features.to(device).float()
    total = features.shape[0]
    num_points = min(num_points, total)

    # 按 Morton 序排序，让空间相邻的点落在相邻索引上
    order = torch.argsort(_morton_codes(features[:, :3].unsqueeze(0))[0])

    outputs = {}
    filled = torch.zeros(total, dtype=torch.bool, device=device)
    step = max(1, int(num_points * (1.0 - overlap)))

    model.eval()
    for start in range(0, total, step):
        block = order[start:start + num_points]
        if block.numel() == 0:
            break
        output = model(features[block].unsqueeze(0))
        pending = ~filled[block]
        if not pending.any():
            continue
        target = block[pending]

        for name in ("semantic_logits", "instance_embedding", "text_embedding"):
            value = getattr(output, name)
            if value is None:
                continue
            outputs.setdefault(name, torch.zeros(
                (total,) + value.shape[2:], dtype=value.dtype, device=device
            ))
            outputs[name][target] = value[0][pending]
        filled[target] = True

        if filled.all():
            break

    if not filled.all():
        raise RuntimeError(f"仍有 {int((~filled).sum())} 个点没有被任何块覆盖")

    return (
        outputs.get("semantic_logits"),
        outputs.get("instance_embedding"),
        outputs.get("text_embedding"),
    )
