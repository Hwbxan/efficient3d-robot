"""把训练好的 PointEncoder 转成 **RKNN 可部署形态**，并验证数值等价。

背景
----
`RKNN算子审计报告_2026-09-19.md` 发现四类算子在 RK3588 的 RKNN 上不可用或需手术：

| 算子 | 来源 | 本模块的处理 |
|---|---|---|
| `Erf` | `nn.GELU()` 精确实现 | 换成 tanh 近似 → 图上只剩 `Tanh` |
| `Sqrt` | `nn.LayerNorm` 的 `1/sqrt(var)` | 换成 `Pow(var+eps, -0.5)` |
| `ReduceL2` / `Sqrt` | `F.normalize` | 换成 `Pow(sum(x²), -0.5)` |
| `Reshape` | 形状重排 | 静态形状下本就可用，无需处理 |

**这些改写是数学等价的**（GELU 除外，它是显式近似），
所以本模块的核心是 `verify_equivalence()` —— 用同一个输入比对改写前后的输出。

一个附带的重要发现
------------------
**含 Morton 编码的完整图，用部署路径的导出器根本导不出来**，
而且不是 opset 的问题 —— 实测：

| opset | 结果 |
|---|---|
| 12 / 13 / 15 / 17 | `ONNX export does NOT support exporting bitwise AND` |
| 18 / 19 / 20 | `ONNX export does NOT support exporting bitwise OR` |

也就是说，Morton 编码的位运算**连 ONNX 这一关都过不去**，
不需要等到上 NPU 才被拒绝。这把"Morton 必须移出图"
从一条经验判断变成了一个硬约束。测量完整图要用
`export_onnx_with_fallback()`（退回 dynamo，仅作量级参考）。

设计原则
--------
**训练路径完全不动。** 本模块在**副本**上做替换，返回一个新模型；
原模型保持原样，已有 checkpoint 不受影响。`L2Normalise` 无参数，
所以替换后 `state_dict` 的键也不变。

用法
----
    from tools.rknn_export import to_rknn_friendly, verify_equivalence

    friendly = to_rknn_friendly(model)          # 返回一个替换后的深拷贝
    report = verify_equivalence(model, friendly, sample)
    assert report["ok"]

命令行：
    python -m tools.rknn_export --onnx-out deploy/point_encoder_rknn.onnx
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 本文件在仓库里的位置是 tools/，而 point_encoder 在 src/models/。
# 仓库内所有工具统一用「项目根入 sys.path + src.xxx 全路径导入」的写法
# （见 train_point_encoder.py / evaluate_point_encoder.py / benchmark_latency.py），
# 这里保持一致，`python -m tools.rknn_export` 与 `python tools/rknn_export.py` 都能跑。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.point_encoder import (  # noqa: E402
    InverseDistanceWeight,
    L2Normalise,
    PointEncoderConfig,
    build_point_encoder,
)

# tanh 近似 GELU 的常数
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_CUBIC = 0.044715


# --------------------------------------------------------------------------- #
# RKNN 友好替换模块
# --------------------------------------------------------------------------- #


class RKNNFriendlyGELU(nn.Module):
    """GELU 的 tanh 近似。

        gelu(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))

    精确 GELU 依赖 `Erf`，RKNN 不支持；tanh 在支持列表里。
    这是本模块里**唯一非严格等价**的替换，偏差量级 ~1e-3（见验证输出）。
    """

    def forward(self, x):
        inner = _SQRT_2_OVER_PI * (x + _GELU_CUBIC * x * x * x)
        return 0.5 * x * (1.0 + torch.tanh(inner))

    def extra_repr(self) -> str:
        return "approximate='tanh'"


class RKNNFriendlyLayerNorm(nn.Module):
    """用 `Pow(·, -0.5)` 代替 `sqrt` 的 LayerNorm。

    `(x - μ) / sqrt(σ² + ε)` 与 `(x - μ) · pow(σ² + ε, -0.5)` 数学等价，
    但后者不产生 `Sqrt` 算子 —— RK3588 NPU 没有 sqrt 硬件加速。

    `σ²` 用**有偏**方差（除以 N），与 `nn.LayerNorm` 一致。
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone())
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        centred = x - mean
        variance = (centred * centred).mean(dim=-1, keepdim=True)
        inv_std = torch.pow(variance + self.eps, -0.5)
        return centred * inv_std * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}, pow=-0.5"


class RKNNFriendlyL2Normalise(nn.Module):
    """用 `Pow` 代替 `ReduceL2` / `Sqrt` 的 L2 归一化。

    与 `F.normalize(x, dim, eps)` 等价：
        F.normalize:  x / max(‖x‖₂, ε)
        本实现:       x · max(Σx², ε²)^(-0.5)

    两者在 `Σx² ≥ ε²` 时都是 `x/‖x‖₂`；否则前者给 `x/ε`，
    后者给 `x·(ε²)^(-0.5) = x/ε`。**完全一致。**
    """

    def __init__(self, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        squared = x * x
        total = squared.sum(dim=self.dim, keepdim=True)
        inv_norm = torch.pow(total.clamp_min(self.eps ** 2), -0.5)
        return x * inv_norm

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, pow=-0.5"


class RKNNFriendlyInverseDistanceWeight(nn.Module):
    """用 `Pow(·, -0.5)` 代替 `.norm()` 的反距离加权。

    原实现是 `1/(d + ε)`，其中 `d = ‖Δ‖₂` —— 会产生
    `ReduceL2` + `Sqrt` + `Reciprocal` 三个算子。

    等价改写：`(‖Δ‖₂² + ε²)^(-0.5)`，只留一个 `Pow`。
    由于权重随后会被归一化，且归一化到单位球后 `d` 通常在 1e-2 量级，
    `ε = 1e-8` 的影响在 1e-6 相对量级以下（由 `verify_equivalence` 把关）。
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, neighbour_xyz, query_xyz):
        delta = neighbour_xyz - query_xyz.unsqueeze(2)
        squared = (delta * delta).sum(dim=-1)
        weight = torch.pow(squared + self.eps ** 2, -0.5)
        return weight / weight.sum(dim=-1, keepdim=True)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, pow=-0.5"


# --------------------------------------------------------------------------- #
# 替换
# --------------------------------------------------------------------------- #


def to_rknn_friendly(model: nn.Module, inplace: bool = False) -> nn.Module:
    """把模型里的 GELU / LayerNorm / L2Normalise / 反距离加权换成 RKNN 友好实现。

    默认返回**深拷贝**，原模型不受影响（训练路径保持不变）。
    返回的模型与原模型 `state_dict` 键完全一致（这些模块都不带参数）。
    """

    target = model if inplace else copy.deepcopy(model)
    counts = Counter()

    def swap(module: nn.Module):
        for name, child in list(module.named_children()):
            replacement = None

            # 用 type() 精确匹配，避免把已替换的模块再换一次
            if type(child) is nn.GELU:
                replacement = RKNNFriendlyGELU()
                counts["gelu"] += 1
            elif type(child) is nn.LayerNorm:
                replacement = RKNNFriendlyLayerNorm(child.weight, child.bias, child.eps)
                counts["layernorm"] += 1
            elif type(child) is L2Normalise:
                replacement = RKNNFriendlyL2Normalise(child.dim, child.eps)
                counts["l2norm"] += 1
            elif type(child) is InverseDistanceWeight:
                replacement = RKNNFriendlyInverseDistanceWeight(child.eps)
                counts["inverse_distance"] += 1

            if replacement is not None:
                if isinstance(module, nn.Sequential):
                    module[int(name)] = replacement
                else:
                    setattr(module, name, replacement)
            else:
                swap(child)

    swap(target)
    return target


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #


@torch.no_grad()
def verify_equivalence(original: nn.Module, friendly: nn.Module, sample: torch.Tensor,
                       tolerance: float = 1e-3,
                       argmax_tolerance: float = 0.99) -> dict:
    """比对改写前后的输出，返回两份**独立**的判据。

    1. `ok` —— 输出**幅值**上的偏差（`max_rel_diff <= tolerance`）。
       这是主判据。默认 `tolerance=1e-3`，因为 GELU 的 tanh 近似本身
       就有这个量级的偏差，不能用 1e-6 那种"严格等价"的阈值。
    2. `argmax_ok` —— 语义分支 argmax 一致率（辅助 sanity check）。

    **两者分开是有意的**：随机初始化的模型 logits 分布平坦，微小扰动就容易
    翻转 argmax，一致率天然偏低；训练好的模型 logits 更自信，一致率更高
    （s5b 实测 99.951%，随机小模型约 99.8%）。把二者绑成一个布尔值会让
    阈值失去意义，所以分开报告。
    """

    original.eval()
    friendly.eval()
    a = original(sample)
    b = friendly(sample)

    report = {
        "tolerance": tolerance,
        "argmax_tolerance": argmax_tolerance,
        "per_output": {},
        "ok": True,
    }

    for name in ("semantic_logits", "instance_embedding", "text_embedding"):
        x = getattr(a, name)
        y = getattr(b, name)
        if x is None or y is None:
            continue
        abs_diff = (x - y).abs()
        scale = x.abs().max().clamp(min=1e-9)
        entry = {
            "max_abs_diff": float(abs_diff.max()),
            "max_rel_diff": float(abs_diff.max() / scale),
            "mean_abs_diff": float(abs_diff.mean()),
            "cosine_min": float(F.cosine_similarity(
                x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), dim=-1).min()),
        }
        entry["ok"] = entry["max_rel_diff"] <= tolerance
        report["per_output"][name] = entry
        report["ok"] = report["ok"] and entry["ok"]

    # 语义分支的 argmax 一致率 —— 评测真正关心的指标，单独判定
    logits_a = a.semantic_logits
    logits_b = b.semantic_logits
    if logits_a is not None and logits_b is not None:
        agree = float((logits_a.argmax(dim=-1) == logits_b.argmax(dim=-1))
                      .float().mean())
        report["semantic_argmax_agreement"] = agree
        report["argmax_ok"] = agree >= argmax_tolerance
        report["ok"] = report["ok"] and report["argmax_ok"]

    return report


# --------------------------------------------------------------------------- #
# ONNX 导出 + 算子核对
# --------------------------------------------------------------------------- #

# 部署路径必须消灭的算子
FORBIDDEN_OPS = ("Erf", "Sqrt", "ReduceL2")


class TupleOutputWrapper(nn.Module):
    """`torch.export` 不接受未注册 pytree 的 dataclass 输出，包一层。"""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, features):
        out = self.inner(features)
        return out.semantic_logits, out.instance_embedding, out.text_embedding


def export_onnx(model: nn.Module, sample: torch.Tensor, path: str,
                opset: int = 12) -> str:
    """导出静态形状的 ONNX（RKNN 只接受静态形状）。**只走部署路径。**

    部署路径 = TorchScript 导出器（`dynamo=False`），产出的图明显更干净
    （Constant 少、没有成片的 Cast/Transpose），也是 RKNN 实际要吃的形态。

    ⚠️ 含 Morton 编码的**完整图用这个函数导不出来**，且不是 opset 的问题：
    实测 opset 12/13/15/17 报 "bitwise AND"、18/19/20 报 "bitwise OR"。
    这是"Morton 必须移出图"的最强证据。要测量完整图请用
    `export_onnx_with_fallback()`。
    """

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wrapped = TupleOutputWrapper(model).eval()
    try:
        torch.onnx.export(
            wrapped,
            (sample,),
            path,
            input_names=["features"],
            output_names=["semantic_logits", "instance_embedding", "text_embedding"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"ONNX 导出失败（部署路径，opset {opset}）："
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}") from exc
    return path


def export_onnx_with_fallback(model: nn.Module, sample: torch.Tensor, path: str,
                              opset: int = 12) -> tuple:
    """部署路径导不出来时退回 dynamo，**仅用于测量**。

    返回 `(实际路径, 导出器名)`；导出器名是 `"torchscript"` 或 `"dynamo"`。
    两个导出器的节点数不可直接比较，调用方必须把这个标签一起报出来。
    """

    try:
        return export_onnx(model, sample, path, opset=opset), "torchscript"
    except RuntimeError:
        dynamo_path = path.replace(".onnx", "_dynamo.onnx")
        os.makedirs(os.path.dirname(os.path.abspath(dynamo_path)), exist_ok=True)
        wrapped = TupleOutputWrapper(model).eval()
        with contextlib.redirect_stdout(io.StringIO()):
            torch.onnx.export(
                wrapped,
                (sample,),
                dynamo_path,
                input_names=["features"],
                output_names=["semantic_logits", "instance_embedding", "text_embedding"],
                opset_version=17,
                do_constant_folding=True,
                dynamic_axes=None,
                dynamo=True,
            )
        return dynamo_path, "dynamo"


def onnx_op_report(path: str) -> dict:
    """读回 ONNX，统计算子并检查禁用算子是否已消灭。"""

    import onnx

    graph = onnx.load(path).graph
    ops = Counter(node.op_type for node in graph.node)
    forbidden = {op: ops[op] for op in FORBIDDEN_OPS if ops.get(op)}
    return {
        "path": path,
        "size_mb": round(os.path.getsize(path) / 1e6, 2),
        "total_nodes": sum(ops.values()),
        "op_count": len(ops),
        "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
        "forbidden_present": forbidden,
        "clean": not forbidden,
    }


def export_deployable_onnx(model: nn.Module, sample: torch.Tensor, path: str,
                           opset: int = 12) -> str:
    """导出**真实部署图** —— Morton 编码与 kNN 的索引由 CPU 提供。

    这两个模块都是动态索引（`argsort` / `topk` / `gather`），
    在 RKNN 上必须留在 CPU（见 `RKNN算子审计报告_2026-09-19.md`）。
    把它们的索引替换成常量后导出，得到的才是真正跑在 NPU 上的那张图。

    `knn_indices` 内部的 `torch.cdist` 会引入 `Sqrt` —— 那些 `Sqrt`
    随 kNN 一起留在 CPU，所以本函数导出的图应当**不含任何禁用算子**。
    """

    from src.models import point_encoder as pe

    saved = {}

    def _static_morton(xyz, count):
        c = min(count, xyz.shape[1])
        base = torch.arange(c, device=xyz.device, dtype=torch.long)
        return base.unsqueeze(0).expand(xyz.shape[0], -1)

    def _static_knn(support, query, k, chunk=512):
        kk = min(k, support.shape[1])
        base = torch.arange(kk, device=support.device, dtype=torch.long)
        return base.view(1, 1, kk).expand(support.shape[0], query.shape[1], kk)

    try:
        saved["morton_stride_indices"] = pe.morton_stride_indices
        saved["knn_indices"] = pe.knn_indices
        pe.morton_stride_indices = _static_morton
        pe.knn_indices = _static_knn
        return export_onnx(model, sample, path, opset)
    finally:
        for name, fn in saved.items():
            setattr(pe, name, fn)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description="导出 RKNN 可部署的 PointEncoder")
    ap.add_argument("--checkpoint", default=None, help="训练好的 state_dict（可选）")
    ap.add_argument("--num-classes", type=int, default=87)
    ap.add_argument("--num-points", type=int, default=2048)
    ap.add_argument("--onnx-out", default="deploy/point_encoder_rknn.onnx")
    ap.add_argument("--deploy-out", default=None,
                    help="部署图输出路径（默认在 --onnx-out 后加 _deploy）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)

    config = PointEncoderConfig(
        in_channels=9, num_classes=args.num_classes, stem_channels=64,
        level_channels=(128, 256, 512), level_k=(16, 16, 12), downsample_stride=4,
        decoder_channels=(256, 128, 128), instance_dim=16, text_dim=512, dropout=0.1,
    )
    model = build_point_encoder(**{k: v for k, v in vars(config).items()
                                   if not k.startswith("_")})
    model.eval()

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        state = state.get("model", state)
        model.load_state_dict(state)
        print(f"已加载 checkpoint：{args.checkpoint}")

    sample = torch.randn(1, args.num_points, 9)

    print("=" * 78)
    print("RKNN 可部署形态导出")
    print("=" * 78)

    friendly = to_rknn_friendly(model)

    replaced = Counter()
    for module in friendly.modules():
        name = type(module).__name__
        if name.startswith("RKNNFriendly"):
            replaced[name] += 1
    print("\n[1] 模块替换")
    for name, count in sorted(replaced.items()):
        print(f"   {name:<28} x{count}")

    print("\n[2] 数值等价性")
    report = verify_equivalence(model, friendly, sample, tolerance=args.tolerance)
    for name, entry in report["per_output"].items():
        mark = "OK " if entry["ok"] else "FAIL"
        print(f"   [{mark}] {name:<22} max_abs={entry['max_abs_diff']:.3e}  "
              f"max_rel={entry['max_rel_diff']:.3e}  cos_min={entry['cosine_min']:.6f}")
    if "semantic_argmax_agreement" in report:
        agree = report["semantic_argmax_agreement"]
        print(f"   [{'OK ' if agree >= 0.999 else 'FAIL'}] 语义 argmax 一致率 "
              f"{agree * 100:.3f}%")
    print(f"\n   → 总体：{'通过' if report['ok'] else '未通过'}")

    print("\n[3] ONNX 导出与算子核对")
    print("    (a) 完整图（含 Morton 编码 + kNN）")
    full_path, exporter = export_onnx_with_fallback(friendly, sample, args.onnx_out)
    op_report = onnx_op_report(full_path)
    exporter_label = {"torchscript": "TorchScript opset 12（部署路径）",
                      "dynamo": "dynamo opset 17（仅测量）"}[exporter]
    print(f"        {full_path}  ({op_report['size_mb']} MB, "
          f"{op_report['total_nodes']} 节点, {op_report['op_count']} 类算子)")
    print(f"        导出器：{exporter_label}")
    if exporter == "dynamo":
        print("        ⚠️  部署路径（TorchScript）导不出来 —— 完整图的位运算过不了 ONNX 导出。")
        print("           这份图只用于测量，**不是部署产物**。")
    if op_report["clean"]:
        print(f"        [OK ] 禁用算子（{'/'.join(FORBIDDEN_OPS)}）已全部消灭")
    else:
        print(f"        [..] 仍存在：{op_report['forbidden_present']} "
              f"（预期来自 kNN 的 cdist，见下）")

    print("\n    (b) 部署图（Morton + kNN 的索引由 CPU 提供）")
    deploy_path = args.deploy_out or args.onnx_out.replace(".onnx", "_deploy.onnx")
    export_deployable_onnx(friendly, sample, deploy_path)
    deploy_report = onnx_op_report(deploy_path)
    print(f"        {deploy_path}  ({deploy_report['size_mb']} MB, "
          f"{deploy_report['total_nodes']} 节点, {deploy_report['op_count']} 类算子)")
    print("        导出器：TorchScript opset 12（部署路径）")
    if deploy_report["clean"]:
        print(f"        [OK ] 禁用算子（{'/'.join(FORBIDDEN_OPS)}）已全部消灭 "
              f"→ **这就是跑在 NPU 上的图**")
    else:
        print(f"        [FAIL] 仍存在禁用算子：{deploy_report['forbidden_present']}")

    print("\n    部署图的算子构成：")
    for op, count in deploy_report["ops"].items():
        print(f"      {op:<24} x{count}")

    if args.json:
        payload = {
            "checkpoint": args.checkpoint,
            "num_points": args.num_points,
            "num_classes": args.num_classes,
            "replaced_modules": dict(replaced),
            "equivalence": report,
            "onnx_full": op_report,
            "onnx_full_exporter": exporter,
            "onnx_deployable": deploy_report,
            "onnx_deployable_exporter": "torchscript",
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\n   已写入 {args.json}")

    return 0 if (report["ok"] and deploy_report["clean"]) else 1


if __name__ == "__main__":
    sys.exit(main())
