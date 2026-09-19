"""Stage 5 的三项训练损失：语义、实例判别嵌入、开放词汇文本对齐。

三项都只在**已标注**点上生效（类别 -1 表示无标注，实例 0 表示无标注）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_CLASS = -1
IGNORE_INSTANCE = 0


def semantic_loss(logits, target, class_weights=None, ignore_index=IGNORE_CLASS):
    """逐点交叉熵。

    `class_weights` 用来对抗结构性类别（wall/floor/ceiling）的面积优势；
    由 `compute_class_weights` 从数据集直方图算出。

    显式升 float32：AMP 下 logits 可能是 half，而 softmax 的归一化项对精度敏感；
    autocast 本来也会为 cross_entropy 做这个升位，这里写出来是为了不依赖隐式行为。
    """

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        target.reshape(-1),
        weight=class_weights,
        ignore_index=ignore_index,
    )


def discriminative_loss(
    embedding,
    instance,
    delta_v=0.5,
    delta_d=1.5,
    alpha=1.0,
    beta=1.0,
    gamma=0.001,
    ignore_instance=IGNORE_INSTANCE,
):
    """实例判别嵌入的 pull / push 损失（SGPN 那一套）。

    - `L_var`：同实例内的点要聚到实例均值附近（半径 delta_v 以内不罚）；
    - `L_dist`：不同实例的均值要拉开（2*delta_d 以外不罚）；
    - `L_reg`：把均值拉向原点，防止嵌入整体漂移导致数值爆炸。

    这里按"块内出现的实例"计算，而不是全局实例——一个 8192 点的块里
    通常只有几个实例，正好是希望分开的那几个。

    ⚠️ `ignore_instance=0` 沿用 Replica 的约定：**object_id 0 表示无标注**。
    如果换成从 0 开始编号的实例数据集，必须显式传 `ignore_instance=None`
    之类的哨兵值，否则编号 0 的那个实例会被静默丢掉。

    ⚠️ 内部统一升到 **float32** 再算。AMP 下 encoder 输出可能是 half，而这个损失
    全是"按实例累加 / 求均值"的归约：half 既丢精度也容易在点数多时溢出，
    而且 `.norm()` 在 autocast 下会升到 float32，与 half 的累加器混用会直接抛
    `index_add_(): self (Half) and source (Float) must have the same scalar type`。
    （这个 bug 只在 `--amp` 下出现，CPU 冒烟测试测不到。）
    """

    embedding = embedding.float()

    if embedding.numel() == 0:
        zero = embedding.sum() * 0.0
        return zero, zero, zero

    valid = instance != ignore_instance
    if valid.sum() < 2:
        zero = embedding.sum() * 0.0
        return zero, zero, zero

    embedding = embedding[valid]
    labels = instance[valid]

    unique, inverse = torch.unique(labels, return_inverse=True)
    instance_count = unique.numel()
    if instance_count < 1:
        zero = embedding.sum() * 0.0
        return zero, zero, zero

    counts = torch.bincount(inverse, minlength=instance_count).clamp(min=1).to(embedding.dtype)
    means = torch.zeros(instance_count, embedding.shape[-1], device=embedding.device, dtype=embedding.dtype)
    means.index_add_(0, inverse, embedding)
    means = means / counts.unsqueeze(1)

    distance_to_mean = (embedding - means[inverse]).norm(dim=1)
    variance_per_point = F.relu(distance_to_mean - delta_v).pow(2)
    variance_sum = torch.zeros(instance_count, device=embedding.device, dtype=embedding.dtype)
    variance_sum.index_add_(0, inverse, variance_per_point)
    variance_loss = (variance_sum / counts).mean()

    # 块内只有 1 个实例时没有配对可推，distance 项为 0；
    # 但 pull 项（同实例聚拢）依然成立，丢掉它等于让这类块白跑。
    if instance_count < 2:
        zero = embedding.sum() * 0.0
        return alpha * variance_loss, zero, gamma * means.norm(dim=1).mean()

    pair_distance = torch.cdist(means, means)
    upper = torch.triu_indices(instance_count, instance_count, offset=1, device=embedding.device)
    distance_loss = F.relu(2.0 * delta_d - pair_distance[upper[0], upper[1]]).pow(2).mean()

    regularisation_loss = means.norm(dim=1).mean()

    return alpha * variance_loss, beta * distance_loss, gamma * regularisation_loss


def text_alignment_loss(text_embedding, class_id, text_bank, temperature=0.07,
                        ignore_index=IGNORE_CLASS):
    """把逐点嵌入对齐到 CLIP 文本空间（InfoNCE）。

    `text_embedding` 与 `text_bank` 都应是 L2 归一化的，于是内积即余弦相似度。
    预测时 `argmax(text_embedding @ text_bank.T)` 就是开放词汇的分类结果，
    所以这个损失**直接优化推理时用的那个量**。

    用 InfoNCE 而不是朴素的 `1 - cos`：后者存在退化解——把所有点映到
    一个对**所有**文本都中等相似的位置即可。对比形式逼模型在同批次的
    类别之间做区分，避免塌缩。

    显式升 float32：相似度矩阵是 `(N, 类别数)`，softmax 要在这个矩阵上做
    归一化，half 精度的累积误差会直接改变梯度方向。
    """

    if text_embedding.numel() == 0 or text_bank.numel() == 0:
        return text_embedding.sum() * 0.0

    logits = (text_embedding.float() @ text_bank.float().t()) / temperature
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        class_id.reshape(-1),
        ignore_index=ignore_index,
    )


def compute_class_weights(class_histogram, class_names, mode="inverse_sqrt", max_weight=20.0):
    """由数据集直方图算类别权重，返回 `(num_classes,)` 的 float32 张量。

    `inverse_sqrt` 比完全逆频率温和：完全逆频率会让 0.01% 的类别拿到
    10000 倍权重，梯度被极少数点主导，训练反而不稳。
    """

    counts = torch.tensor(
        [float(class_histogram.get(name, 0.0)) for name in class_names], dtype=torch.float64
    )
    present = counts > 0
    if not present.any():
        return torch.ones(len(class_names), dtype=torch.float32)

    weights = torch.ones_like(counts)
    if mode == "inverse":
        weights[present] = 1.0 / counts[present]
    elif mode == "inverse_sqrt":
        weights[present] = 1.0 / counts[present].sqrt()
    elif mode == "none":
        return torch.ones(len(class_names), dtype=torch.float32)
    else:
        raise ValueError(f"未知的 mode：{mode}")

    weights[present] = weights[present] / weights[present].mean()
    weights = weights.clamp(max=max_weight)
    # 数据集里没出现过的类别权重置 0，避免它被误当成"稀有但重要"
    weights[~present] = 0.0
    return weights.to(torch.float32)
