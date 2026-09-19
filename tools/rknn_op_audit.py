"""RKNN 算子兼容性审计：把 PointEncoder 的前向图拆成算子，逐个判定能否上 RK3588 NPU。

背景
----
M2/M3 的前提是"我们的点式骨干能部署到 RK3588 NPU"。这个前提必须先验证，
否则等到硬件到位才发现 kNN / Gather 无法手术，就是重大返工。

本脚本**不需要任何硬件**，纯静态分析，在本地 CPU 上就能跑。

方法
----
1. 用 `torch.fx.symbolic_trace` 拿到前向图，统计算子（含算子实例化后的实际参数）；
2. 尝试 ONNX 导出，作为权威口径（fx 看不到 torch 内部的算子分解）；
3. 对着 RKNN 支持表给每个算子打标：
     NPU        —— 有对应支持，静态形状下可上 NPU
     SURGERY    —— 不支持，但可等价改写成支持的算子
     CPU_ONLY   —— 必须留 CPU（动态索引 / 数据相关控制流）
     UNKNOWN    —— 需实测（工具链版本相关）

用法
----
    python -m tools.rknn_op_audit                       # 用 s5b 的真实配置
    python -m tools.rknn_op_audit --num-points 2048 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn

# 本文件在仓库里的位置是 tools/，而 point_encoder 在 src/models/。
# 与 train_point_encoder.py / evaluate_point_encoder.py 保持同一种导入写法。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.point_encoder import (  # noqa: E402
    PointEncoder,
    PointEncoderConfig,
    build_point_encoder,
)


# --------------------------------------------------------------------------- #
# RKNN 支持表（依据 rknn-toolkit 算子支持文档 + RK3588 实测记录整理）
# --------------------------------------------------------------------------- #
#
# 证据等级说明：
#   - 官方支持列表来自 rknn-toolkit 算子支持参考（v1.7.5，早于 RK3588 的 toolkit2 线）
#   - "实测"结论来自公开的 RK3588 PointNet 部署记录（博客级证据，非论文）
#   - 凡标 UNKNOWN 的，必须在真机上用 toolkit2 的 RKNN_OP_Support_And_Limit.xlsx 复核

RKNN_SUPPORT = {
    # ---- 稠密算子：支持，静态形状 ----
    "Gemm": ("NPU", "全连接，需静态形状；BN 应提前融合"),
    "MatMul": ("NPU", "需静态形状"),
    "Add": ("NPU", ""),
    "Sub": ("NPU", ""),
    "Mul": ("NPU", ""),
    "Div": ("NPU", "注意除零"),
    "Concat": ("NPU", "沿通道维最省"),
    "Reshape": ("NPU", "本图已全静态（dynamic_axes=None），可直接上 NPU；"
                       "但若 ONNX 里残留 ? 维，RKNN 会报 invalid input shape"),
    "Transpose": ("NPU", "布局转换有带宽代价"),
    "Flatten": ("NPU", ""),
    "Relu": ("NPU", ""),
    "LeakyRelu": ("NPU", ""),
    "Sigmoid": ("NPU", ""),
    "Tanh": ("NPU", ""),
    "Softmax": ("NPU", ""),
    "MaxPool": ("NPU", ""),
    "AveragePool": ("NPU", ""),
    "GlobalAveragePool": ("NPU", ""),
    "ReduceMax": ("NPU", "沿非连续维做规约时需内存布局配合"),
    "ReduceMean": ("NPU", ""),
    "ReduceSum": ("NPU", ""),
    "Slice": ("NPU", ""),
    "Split": ("NPU", ""),
    "Pad": ("NPU", ""),
    "Clip": ("NPU", ""),
    "Exp": ("NPU", ""),
    "Log": ("NPU", ""),
    "Pow": ("NPU", ""),
    "Sqrt": ("SURGERY", "NPU 无 sqrt 硬件加速；用平方距离可完全规避"),
    "BatchNormalization": ("SURGERY", "INT8 下易溢出，应在导出前融合进 Conv/Gemm"),
    "Constant": ("NPU", "常量折叠"),
    "Identity": ("NPU", ""),
    "Dropout": ("NPU", "推理态即 Identity"),
    "L2Normalize": ("NPU", "TFLite 有 L2_NORMALIZATION"),
    "Cast": ("NPU", "支持；但大量 Cast 说明图里混了 CPU 侧预处理"),
    "Unsqueeze": ("NPU", ""),
    "Squeeze": ("NPU", ""),
    "Expand": ("NPU", ""),
    "Less": ("NPU", "比较算子"),
    "Greater": ("NPU", ""),
    "Equal": ("NPU", ""),
    "Where": ("NPU", "等价 TFLite SELECT"),
    "ReduceMin": ("NPU", ""),
    "ReduceL2": ("SURGERY", "可拆成 Pow+ReduceSum+Sqrt；F.normalize 的产物"),
    "Reciprocal": ("NPU", "等价 1/x"),
    "Round": ("UNKNOWN", "需实测；也可改用 Clip+Cast 近似"),
    "Tanh": ("NPU", "GELU 的 tanh 近似可用它规避 Erf"),
    "Shape": ("NPU", "静态形状下会被常量折叠"),
    "ConstantOfShape": ("NPU", ""),
    "Neg": ("NPU", ""),
    "Sigmoid": ("NPU", ""),
    "Silu": ("UNKNOWN", "需实测；RKNN 新版可能支持 Swish"),

    # ---- 位运算：Morton 编码的产物，RKNN 完全不支持 ----
    "BitShift": ("CPU_ONLY", "RKNN 不支持位运算；Morton 编码必须整体移出图"),
    "BitwiseOr": ("CPU_ONLY", "同上"),
    "BitwiseAnd": ("CPU_ONLY", "同上"),
    "BitwiseNot": ("CPU_ONLY", "同上"),

    # ---- 动态索引：RKNN 的核心短板 ----
    "Gather": ("CPU_ONLY", "NPU 张量引擎不支持非连续内存索引；可尝试 topk+expand 静态化，但索引本身仍动态"),
    "GatherElements": ("CPU_ONLY", "RKNN 明确不支持"),
    "GatherND": ("CPU_ONLY", "未列入支持表"),
    "ScatterND": ("CPU_ONLY", "未列入支持表"),
    "ScatterElements": ("CPU_ONLY", "RKNN 明确不支持"),
    "IndexSelect": ("CPU_ONLY", "等价 Gather"),
    "TopK": ("CPU_ONLY", "未列入支持表；kNN 的核心算子"),
    "ArgMax": ("CPU_ONLY", "常用于 CPU 侧后处理"),
    "ArgMin": ("CPU_ONLY", "同上"),
    "Sort": ("CPU_ONLY", "未支持"),

    # ---- 需实测 ----
    "LayerNorm": ("UNKNOWN", "旧版 toolkit 未列出；toolkit2 可能支持。可拆成 ReduceMean+Sub+Pow+ReduceMean+Sqrt+Div"),
    "LayerNormalization": ("UNKNOWN", "ONNX 原生写法。**这是可行性的关键判据之一**：若不支持，需手工拆解或留 CPU"),
    "Gelu": ("UNKNOWN", "未列出；Erf 不支持，需换 tanh 近似或 ReLU/SiLU"),
    "Erf": ("SURGERY", "不支持；GELU 的精确实现依赖它 → 换 tanh 近似即可彻底规避"),
    "Loop": ("CPU_ONLY", "RKNN v1.6 完全不支持"),
    "If": ("CPU_ONLY", "不支持"),
    "NonMaxSuppression": ("CPU_ONLY", "未支持"),
    "GridSample": ("CPU_ONLY", "未支持"),
    "Cdist": ("SURGERY", "由 MatMul+广播 Sub+Pow+ReduceSum+Sqrt 组合而成；输出 (B,Q,N) 过大，通常留 CPU"),
}


def classify(op_name: str):
    return RKNN_SUPPORT.get(op_name, ("UNKNOWN", "不在已知支持表中，需查 toolkit2 的 xlsx"))


# --------------------------------------------------------------------------- #
# 1. ONNX 图普查
# --------------------------------------------------------------------------- #
#
# 关于两个导出器：
#   - **TorchScript 导出器**（`dynamo=False`，opset 12）是**部署路径**用的那个，
#     产出的图明显更干净（Constant 少、没有成片的 Cast/Transpose）。
#   - **dynamo 导出器**（`torch.export`，opset 17）分解得更碎，
#     但能容纳 TorchScript 导出器直接拒绝的图。
#
# 两者的节点数**不可直接比较**，所以每个档位都要记录用的是哪个导出器。


def onnx_census(model: nn.Module, sample: torch.Tensor, path: str, opset: int,
                 dynamo: bool):
    """导出并统计算子。返回 (ops, error)。"""

    try:
        import onnx
    except ImportError:
        return None, "onnx 未安装"

    import contextlib
    import io
    import logging

    wrapped = TupleOutputWrapper(model).eval()
    # dynamo 导出器会把 "[torch.onnx] Obtain model graph ..." 直接打到 stdout，
    # 淹没审计报告，这里临时压掉（logging 与 stdout 两条路都堵）。
    onnx_logger = logging.getLogger("torch.onnx")
    previous_level = onnx_logger.level
    onnx_logger.setLevel(logging.ERROR)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            torch.onnx.export(
                wrapped,
                (sample,),
                path,
                input_names=["features"],
                output_names=["semantic_logits", "instance_embedding", "text_embedding"],
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes=None,
                dynamo=dynamo,
            )
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    finally:
        onnx_logger.setLevel(previous_level)

    graph = onnx.load(path).graph
    return Counter(node.op_type for node in graph.node), None


def export_variant(model: nn.Module, sample: torch.Tensor, path: str, level: str):
    """在"把某些 CPU 侧算子拿掉"的前提下导出，测出真实的 NPU 侧图。

    level:
      "full"   —— 不改，导出完整图（含 Morton 编码 + kNN）
      "morton" —— Morton 编码换成常量索引（模拟"索引由 CPU 提供"）
      "knn"    —— 再把 kNN 也换成常量索引（模拟"邻域搜索在 CPU 完成"）

    说明：索引变成常量后，ONNX 的常量折叠会把随之而来的 Gather 折成 Slice/Identity，
    所以 "knn" 档测出的是 **NPU 侧的下界**（只剩稠密算子）。
    真实部署里索引是数据相关的，Gather 仍要留在 CPU —— 但那些算子本来就很小。

    返回 dict：
      ops        —— Counter，导出失败时为 None
      exporter   —— "torchscript" / "dynamo"，失败时为 None
      path       —— 实际写出 ONNX 的路径（两个导出器路径不同）
      error      —— 部署路径（TorchScript）导出失败时的原因
      error_dynamo —— dynamo 也失败时的原因
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
        if level in ("morton", "knn"):
            saved["morton_stride_indices"] = pe.morton_stride_indices
            pe.morton_stride_indices = _static_morton
        if level == "knn":
            saved["knn_indices"] = pe.knn_indices
            pe.knn_indices = _static_knn

        # 先走部署路径（TorchScript，opset 12）
        ops, error = onnx_census(model, sample, path, opset=12, dynamo=False)
        if ops is not None:
            return {"ops": ops, "exporter": "torchscript", "path": path,
                    "error": None, "error_dynamo": None}

        # 部署路径导不出来时，退回 dynamo 只为**测量**，不作为部署产物。
        # 完整图就是这种情况：Morton 的位运算过不了 TorchScript 导出器。
        dynamo_path = path.replace(".onnx", "_dynamo.onnx")
        ops_dynamo, error_dynamo = onnx_census(
            model, sample, dynamo_path, opset=17, dynamo=True)
        return {"ops": ops_dynamo,
                "exporter": "dynamo" if ops_dynamo is not None else None,
                "path": dynamo_path if ops_dynamo is not None else None,
                "error": error, "error_dynamo": error_dynamo}
    finally:
        for name, fn in saved.items():
            setattr(pe, name, fn)


def module_census(model: nn.Module):
    """按模块类型统计（不依赖 fx 追踪）。

    `PointEncoder.forward` 里有形状断言（`if features.dim() != 3`），
    fx 的符号追踪会在这句上抛 TraceError。这里不做图追踪 ——
    **图级别的算子普查交给 ONNX**，因为 ONNX 图才是真正要转成 RKNN 的产物。
    本函数只补充 ONNX 看不到的模块级细节（如 Gemm 的 in/out 维度）。
    """

    ops = Counter()
    details = defaultdict(list)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            params = module.weight.numel()
            if module.bias is not None:
                params += module.bias.numel()
            ops["Gemm"] += 1
            details["Gemm"].append(
                f"{name}: in={module.in_features} out={module.out_features} params={params:,}"
            )
        elif isinstance(module, nn.LayerNorm):
            ops["LayerNorm"] += 1
            details["LayerNorm"].append(
                f"{name}: shape={tuple(module.normalized_shape)} eps={module.eps}"
            )
        elif isinstance(module, nn.GELU):
            ops["Gelu"] += 1
            details["Gelu"].append(name)
        elif isinstance(module, nn.Dropout):
            ops["Dropout"] += 1
            details["Dropout"].append(name)
        elif isinstance(module, nn.Identity):
            ops["Identity"] += 1
            details["Identity"].append(name)

    return ops, details


# --------------------------------------------------------------------------- #
# 2. ONNX 导出（权威口径）
# --------------------------------------------------------------------------- #


class TupleOutputWrapper(nn.Module):
    """把 `PointEncoderOutput` 拆成 tuple。

    `torch.export` 不接受未注册 pytree 的自定义 dataclass 作为输出，
    所以导出前必须包一层。这层不影响计算图。
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, features):
        out = self.inner(features)
        return out.semantic_logits, out.instance_embedding, out.text_embedding


# --------------------------------------------------------------------------- #
# 3. 报告
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-points", type=int, default=2048,
                    help="导出用的静态点数（推理时是 8192，这里用小一点加快导出）")
    ap.add_argument("--num-classes", type=int, default=87, help="s5b 的真实类别数")
    ap.add_argument("--json", default=None)
    ap.add_argument("--onnx-out", default=None)
    args = ap.parse_args()

    config = PointEncoderConfig(
        in_channels=9,
        num_classes=args.num_classes,
        stem_channels=64,
        level_channels=(128, 256, 512),
        level_k=(16, 16, 12),
        downsample_stride=4,
        decoder_channels=(256, 128, 128),
        instance_dim=16,
        text_dim=512,
        dropout=0.1,
    )
    model = PointEncoder(config)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print("=" * 78)
    print("RKNN 算子兼容性审计 —— PointEncoder")
    print("=" * 78)
    print(f"配置：num_classes={args.num_classes}  level_k={config.level_k}  "
          f"stride={config.downsample_stride}")
    print(f"参数量：{total:,}")
    print(f"静态输入：features (1, {args.num_points}, 9)")
    down = [args.num_points]
    for _ in config.level_channels:
        down.append(max(1, down[-1] // config.downsample_stride))
    print(f"各级点数：{' -> '.join(str(d) for d in down)}")

    sample = torch.randn(1, args.num_points, 9)

    with torch.no_grad():
        out = model(sample)
    print(f"前向输出：semantic_logits {tuple(out.semantic_logits.shape)}  "
          f"instance_embedding {tuple(out.instance_embedding.shape)}  "
          f"text_embedding {tuple(out.text_embedding.shape)}")

    # ---- ONNX 优先：这才是真正要转成 RKNN 的产物 ----
    print(f"\n{'=' * 78}\n[1] ONNX 导出（权威口径）\n{'=' * 78}")
    onnx_out = args.onnx_out or os.path.join(
        os.environ.get("TEMP", "/tmp"), "point_encoder_audit.onnx")
    full = export_variant(model, sample, onnx_out, "full")
    onnx_ops = full["ops"]
    if onnx_ops is None:
        print("   部署路径（TorchScript 导出器，opset 12）导出失败：")
        print(f"      {full['error']}")
        if full["error_dynamo"]:
            print("      dynamo 导出器也失败：")
            print(f"      {full['error_dynamo']}")
        print("   → 退回模块级普查（不依赖追踪）")
    else:
        exporter_name = {"torchscript": "TorchScript 导出器 opset 12",
                         "dynamo": "dynamo 导出器 opset 17"}[full["exporter"]]
        print(f"   导出成功（{exporter_name}）：{full['path']}  "
              f"{os.path.getsize(full['path']) / 1e6:.1f} MB")
        if full["error"]:
            print(f"   注意：部署路径（TorchScript opset 12）**导不出来**——")
            print(f"      {full['error']}")
            print("      上面这份图是 dynamo 导出的，只用于测量，不作为部署产物。")
        onnx_buckets = defaultdict(list)
        for op, count in sorted(onnx_ops.items(), key=lambda kv: -kv[1]):
            verdict, note = classify(op)
            onnx_buckets[verdict].append((op, count, note))
        for verdict, title in [
            ("NPU", "可上 NPU"),
            ("SURGERY", "需手术"),
            ("UNKNOWN", "需实测"),
            ("CPU_ONLY", "必须留 CPU"),
        ]:
            items = onnx_buckets.get(verdict, [])
            if not items:
                continue
            print(f"\n-- {title} --")
            for op, count, note in items:
                print(f"   {op:<22} x{count:<4} {note}")

    # ---- 模块级细节（不依赖追踪） ----
    print(f"\n{'=' * 78}\n[2] 模块级细节\n{'=' * 78}")
    ops, details = module_census(model)
    for op in ("Gemm", "LayerNorm", "Gelu", "Dropout", "Identity"):
        if details.get(op):
            print(f"\n   {op} x{len(details[op])}")
            for d in details[op][:10]:
                print(f"      {d}")
            if len(details[op]) > 10:
                print(f"      ... 另有 {len(details[op]) - 10} 个")

    # ---- 结论 ----
    print(f"\n{'=' * 78}\n[3] 结论（模块级普查口径）\n{'=' * 78}")
    print("   说明：这一节用模块级普查，口径与导出器无关；")
    print("         图级的真实切分在 [4] 节。")
    source = ops
    counts = Counter()
    for op, count in source.items():
        counts[classify(op)[0]] += count
    grand = sum(counts.values()) or 1
    for verdict, title in [
        ("NPU", "可上 NPU"),
        ("SURGERY", "需手术"),
        ("UNKNOWN", "需实测"),
        ("CPU_ONLY", "必须留 CPU"),
    ]:
        c = counts.get(verdict, 0)
        print(f"   {title:<16} {c:>4} / {grand}  节点 ({100.0 * c / grand:.1f}%)")

    # 参数量口径：Gemm 占绝对多数，是压缩的主战场
    gemm_params = sum(
        m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
        for m in model.modules() if isinstance(m, nn.Linear)
    )
    print(f"\n   Gemm 参数 {gemm_params:,} / 总参数 {total:,} "
          f"({100.0 * gemm_params / total:.1f}%)  ← 剪枝/量化的主战场")

    # ---- 图切分：实测三种导出档位 ----
    print(f"\n{'=' * 78}\n[4] 图切分实测：NPU 侧到底剩多少\n{'=' * 78}")
    print("   每个档位都先试**部署路径**（TorchScript 导出器，opset 12）；")
    print("   导不出来时才退回 dynamo 导出器（仅用于测量）。两个导出器的节点数不可比。")

    variants = [
        ("full", "完整图（含 Morton 编码 + kNN）"),
        ("morton", "Morton 移出（索引由 CPU 提供）"),
        ("knn", "Morton + kNN 都移出（只剩稠密算子）"),
    ]
    variant_results = {}
    variant_meta = {}
    tmpdir = os.environ.get("TEMP", "/tmp")
    for level, label in variants:
        vpath = os.path.join(tmpdir, f"pe_audit_{level}.onnx")
        result = export_variant(model, sample, vpath, level)
        vops = result["ops"]
        variant_meta[level] = {"exporter": result["exporter"],
                               "error": result["error"]}
        if vops is None:
            print(f"\n   {label}：两个导出器都失败")
            print(f"      {result['error']}")
            continue
        variant_results[level] = dict(vops)
        vtotal = sum(vops.values()) or 1
        vcounts = Counter()
        for op, c in vops.items():
            vcounts[classify(op)[0]] += c
        exporter_name = {"torchscript": "TorchScript opset 12（部署路径）",
                         "dynamo": "dynamo opset 17（仅测量）"}[result["exporter"]]
        print(f"\n   {label}  ——  {vtotal} 节点   [{exporter_name}]")
        for verdict, title in [
            ("NPU", "可上 NPU"),
            ("SURGERY", "需手术"),
            ("UNKNOWN", "需实测"),
            ("CPU_ONLY", "必须留 CPU"),
        ]:
            c = vcounts.get(verdict, 0)
            print(f"      {title:<12} {c:>5} ({100.0 * c / vtotal:5.1f}%)")
        if result["error"]:
            print(f"      注：部署路径导不出来 —— {result['error']}")

    if "knn" in variant_results:
        final = variant_results["knn"]
        cpu_left = sorted((op, c) for op, c in final.items()
                          if classify(op)[0] == "CPU_ONLY")
        print(f"\n   剥离 Morton + kNN 之后，图里剩下的 CPU_ONLY 算子：")
        if cpu_left:
            for op, c in cpu_left:
                print(f"      {op:<22} x{c}")
        else:
            print("      （无）—— 全部动态算子都已被剥离")

        print(f"\n   NPU 主图的完整算子构成：")
        for op, c in sorted(final.items(), key=lambda kv: -kv[1]):
            print(f"      {op:<22} x{c:<5} {classify(op)[0]}")

    # 兼容旧口径：按"必须留 CPU"集合估算
    cpu_ops = {op for op, count in source.items() if classify(op)[0] == "CPU_ONLY"}
    cpu_nodes = sum(source[op] for op in cpu_ops)
    residual = {op: c for op, c in source.items() if op not in cpu_ops}
    resid_counts = Counter()
    for op, c in residual.items():
        resid_counts[classify(op)[0]] += c

    blocking = sorted(
        (op, c) for op, c in residual.items() if classify(op)[0] in ("SURGERY", "UNKNOWN")
    )
    if blocking:
        print(f"\n   挡住可行性的算子（按节点数）：")
        for op, c in sorted(blocking, key=lambda kv: -kv[1]):
            print(f"      {op:<22} x{c:<5} {classify(op)[1]}")

    # ---- 结论 ----
    print(f"\n{'=' * 78}\n[5] 这个审计给出的三个结论\n{'=' * 78}")

    full_ops = variant_results.get("full") or {}
    full_meta = variant_meta.get("full", {})
    bitwise_family = ("BitShift", "BitwiseOr", "BitwiseAnd", "BitwiseNot")
    bitwise_nodes = sum(full_ops.get(op, 0) for op in bitwise_family)
    cast_nodes = full_ops.get("Cast", 0)

    if full_meta.get("error"):
        line1 = (
            f"      最直接的证据：用**部署路径的导出器**（TorchScript，opset 12）\n"
            f"      完整图**根本导不出来** ——\n"
            f"        {full_meta['error']}\n"
            f"      也就是说，Morton 的位运算连 ONNX 这一关都过不去，"
            f"更谈不上上 NPU。\n"
        )
        if full_meta.get("exporter") == "dynamo" and bitwise_nodes:
            line1 += (
                f"      换成 dynamo 导出器可以导出来（{sum(full_ops.values())} 节点），\n"
                f"      其中位运算族 BitShift/BitwiseOr/BitwiseAnd/BitwiseNot 共 "
                f"{bitwise_nodes} 个，另有 Cast {cast_nodes} 个。\n"
                f"      注意 dynamo 的节点数与部署路径不可比，这里只作为量级参考。\n"
            )
    else:
        line1 = (
            f"      完整图 {sum(full_ops.values())} 节点，其中位运算族 "
            f"{bitwise_nodes} 个、Cast {cast_nodes} 个。RKNN 不支持任何位运算。\n"
        )

    morton_cpu = {}
    if "morton" in variant_results:
        morton_cpu = {op: c for op, c in variant_results["morton"].items()
                      if classify(op)[0] == "CPU_ONLY"}

    surgery_ops = {}
    if "knn" in variant_results:
        surgery_ops = {op: c for op, c in variant_results["knn"].items()
                       if classify(op)[0] == "SURGERY"}

    print(f"""
   1) Morton 编码必须整体移出 ONNX 图。
{line1}      → 这本来就是设计意图（CPU 侧排序），现在有了量化依据。

   2) kNN 与邻域收集（TopK / Gather / GatherElements）同样必须留 CPU。
      把 Morton 移出后，图里还剩 {sum(morton_cpu.values())} 个 CPU_ONLY 节点：
      {', '.join(f'{op} x{c}' for op, c in sorted(morton_cpu.items())) or '（无）'}
      好在它们本来就在 `knn_indices` / `gather_features` 里，边界清晰。

   3) 切完之后，NPU 主图是一个**纯稠密 MLP 栈**（MatMul + LayerNorm + 激活 + 逐元素）。
      剥离 Morton 与 kNN 后，剩下的"需手术"算子恰好是：
      {', '.join(f'{op} x{c}' for op, c in sorted(surgery_ops.items())) or '（无）'}
      → 这三个族就是 `tools/rknn_export.py` 要消灭的全部对象：
        Erf(GELU) → Tanh 近似；Sqrt(LayerNorm) / ReduceL2(F.normalize) → Pow(·, -0.5)。
      → 改写之后唯一需要真机确认的只剩 **LayerNorm 的算子组合**（Pow 与 Tanh 是否都在 NPU 上）。
""")

    payload = {
        "config": {
            "num_classes": args.num_classes,
            "num_points": args.num_points,
            "level_k": list(config.level_k),
            "level_channels": list(config.level_channels),
            "downsample_stride": config.downsample_stride,
            "parameters": total,
            "gemm_parameters": gemm_params,
        },
        "onnx_ops": dict(onnx_ops) if onnx_ops is not None else None,
        "onnx_exporter": full["exporter"],
        "onnx_deploy_path_error": full["error"],
        "module_ops": dict(ops),
        "verdicts": {op: {"verdict": classify(op)[0], "note": classify(op)[1]}
                     for op in source},
        "summary": dict(counts),
        "cpu_side_ops": sorted(cpu_ops),
        "residual_npu_graph": dict(residual),
        "residual_summary": dict(resid_counts),
        "variants": variant_results,
        "variant_exporters": variant_meta,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"   已写入 {args.json}")


if __name__ == "__main__":
    main()
