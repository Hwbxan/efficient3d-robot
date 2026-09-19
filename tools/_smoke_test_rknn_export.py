"""RKNN 导出路径的冒烟测试。

验证 `rknn_export.py` 的四条核心保证：

1. **不改动原模型** —— `to_rknn_friendly` 默认返回深拷贝；
2. **state_dict 键不变** —— 替换的模块都不带参数，已有 checkpoint 仍能加载；
3. **数值等价** —— 三个替换模块各自与 PyTorch 原实现对齐到预期精度；
4. **部署图干净** —— 导出的 ONNX 不含 Erf / Sqrt / ReduceL2。

运行：
    python -m tools._smoke_test_rknn_export
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 与仓库内其他工具一致：项目根入 sys.path，用 src.xxx / tools.xxx 全路径导入。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.point_encoder import (  # noqa: E402
    InverseDistanceWeight,
    L2Normalise,
    PointEncoderConfig,
    build_point_encoder,
)
from tools.rknn_export import (  # noqa: E402
    FORBIDDEN_OPS,
    RKNNFriendlyGELU,
    RKNNFriendlyInverseDistanceWeight,
    RKNNFriendlyL2Normalise,
    RKNNFriendlyLayerNorm,
    export_deployable_onnx,
    onnx_op_report,
    to_rknn_friendly,
    verify_equivalence,
)

TMP = os.path.join(tempfile.gettempdir(), "smoke_rknn_export")
os.makedirs(TMP, exist_ok=True)

# RKNN 支持列表里我们确认可用的算子（依据审计报告）
SUPPORTED_IN_DEPLOY_GRAPH = {
    "Constant", "ConstantOfShape", "Mul", "Add", "Sub", "Div", "Pow", "MatMul",
    "ReduceMean", "ReduceSum", "ReduceMax",
    "Expand", "Reshape", "Unsqueeze", "Concat", "Slice", "Clip", "Identity",
    "Equal", "Where", "Tanh",
    # 已知需留 CPU 或需真机确认的（单独列出，不计入失败）
    "GatherElements",
}


def _tiny_model(num_classes: int = 5, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    model = build_point_encoder(
        in_channels=9, num_classes=num_classes, stem_channels=32,
        level_channels=(64, 128), level_k=(8, 8), downsample_stride=4,
        decoder_channels=(64, 32), instance_dim=8, text_dim=64, dropout=0.0,
    )
    model.eval()
    return model


# --------------------------------------------------------------------------- #


def check_original_untouched():
    print("--- 原模型不被改动 ---")

    model = _tiny_model()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    before_types = [type(m).__name__ for m in model.modules()]

    friendly = to_rknn_friendly(model)

    after = model.state_dict()
    assert set(before) == set(after), "state_dict 键发生了变化"
    for key in before:
        assert torch.equal(before[key], after[key]), f"{key} 被改动了"
    assert before_types == [type(m).__name__ for m in model.modules()], \
        "原模型的模块类型被改动了"

    # friendly 必须是不同的对象
    assert friendly is not model, "返回了同一个对象，说明没做深拷贝"

    friendly_types = {type(m).__name__ for m in friendly.modules()}
    assert "RKNNFriendlyGELU" in friendly_types, "GELU 没有被替换"
    assert "RKNNFriendlyLayerNorm" in friendly_types, "LayerNorm 没有被替换"
    assert "RKNNFriendlyL2Normalise" in friendly_types, "L2Normalise 没有被替换"
    assert "RKNNFriendlyInverseDistanceWeight" in friendly_types, \
        "反距离加权没有被替换"

    print(f"  原模型 {len(before)} 个 state_dict 项全部未变")
    print(f"  替换后模块类型：{sorted(t for t in friendly_types if t.startswith('RKNNFriendly'))}")
    print("  OK")


def check_state_dict_compatible():
    print("--- state_dict 双向兼容 ---")

    model = _tiny_model(seed=1)
    friendly = to_rknn_friendly(model)

    # 原 checkpoint 能加载进 friendly
    friendly.load_state_dict(model.state_dict(), strict=True)
    # 反之亦然
    model.load_state_dict(friendly.state_dict(), strict=True)

    print(f"  strict=True 双向加载成功，{len(model.state_dict())} 项")
    print("  OK")


def check_module_equivalence():
    print("--- 各替换模块的数值等价 ---")

    torch.manual_seed(0)

    # LayerNorm
    x = torch.randn(4, 32, 128) * 3.0
    ref = nn.LayerNorm(128, eps=1e-5)
    mine = RKNNFriendlyLayerNorm(ref.weight, ref.bias, ref.eps)
    diff = (ref(x) - mine(x)).abs().max().item()
    assert diff < 1e-5, f"LayerNorm 偏差过大：{diff:.3e}"
    print(f"  LayerNorm        最大偏差 {diff:.3e}")

    # L2Normalise
    y = torch.randn(4, 32, 64) * 5.0
    ref_n = L2Normalise(dim=-1)
    mine_n = RKNNFriendlyL2Normalise(dim=-1)
    diff = (ref_n(y) - mine_n(y)).abs().max().item()
    assert diff < 1e-6, f"L2Normalise 偏差过大：{diff:.3e}"
    print(f"  L2Normalise      最大偏差 {diff:.3e}")

    # 极小输入下的 eps 行为（F.normalize 的 clamp 语义）
    tiny = torch.randn(1, 4, 8) * 1e-9
    diff = (ref_n(tiny) - mine_n(tiny)).abs().max().item()
    assert diff < 1e-6, f"极小输入下 L2Normalise 偏差过大：{diff:.3e}"
    print(f"  L2Normalise(极小) 最大偏差 {diff:.3e}")

    # GELU tanh 近似（唯一非严格等价）
    z = torch.randn(4, 32, 128) * 2.0
    diff = (F.gelu(z) - RKNNFriendlyGELU()(z)).abs().max().item()
    assert diff < 5e-3, f"GELU 近似偏差过大：{diff:.3e}"
    print(f"  GELU(tanh 近似)  最大偏差 {diff:.3e}  ← 唯一非严格等价项")

    # 反距离加权
    nbr = torch.randn(2, 16, 8, 3)
    qry = torch.randn(2, 16, 3)
    diff = (InverseDistanceWeight()(nbr, qry)
            - RKNNFriendlyInverseDistanceWeight()(nbr, qry)).abs().max().item()
    assert diff < 1e-6, f"反距离加权偏差过大：{diff:.3e}"
    print(f"  反距离加权       最大偏差 {diff:.3e}")

    print("  OK")


def check_model_level_equivalence():
    print("--- 整模型等价性 ---")

    model = _tiny_model(seed=2)
    friendly = to_rknn_friendly(model)
    sample = torch.randn(1, 512, 9)

    report = verify_equivalence(model, friendly, sample, tolerance=1e-3)
    for name, entry in report["per_output"].items():
        mark = "OK " if entry["ok"] else "FAIL"
        print(f"  [{mark}] {name:<20} max_rel={entry['max_rel_diff']:.3e} "
              f"cos_min={entry['cosine_min']:.6f}")

    # 关键判据是**输出幅值**上的偏差（max_rel < 1e-3）。
    # argmax 一致率只是辅助 sanity check，而且**随机初始化的模型会放大它** ——
    # 权重随机时 logits 分布平坦，微小扰动就容易翻转 argmax。
    # 真实训练好的模型 logits 更自信：s5b 上实测 99.951%，这里的小模型约 99.8%。
    assert report["ok"], "输出幅值等价性未通过（max_rel 超过 tolerance）"
    agree = report.get("semantic_argmax_agreement")
    if agree is not None:
        ok = report.get("argmax_ok", agree >= 0.99)
        print(f"  [{'OK ' if ok else 'FAIL'}] 语义 argmax 一致率 {agree * 100:.3f}%"
              f"（随机权重下预期略低于训练模型）")
        assert ok, f"argmax 一致率过低：{agree:.4f}"

    print("  OK")


def check_deployable_graph_is_clean():
    print("--- 部署图不含禁用算子 ---")

    model = _tiny_model(seed=3)
    friendly = to_rknn_friendly(model)
    sample = torch.randn(1, 512, 9)
    path = os.path.join(TMP, "deploy.onnx")

    export_deployable_onnx(friendly, sample, path)
    report = onnx_op_report(path)

    print(f"  {report['total_nodes']} 节点 / {report['op_count']} 类算子 / {report['size_mb']} MB")
    assert report["clean"], \
        f"部署图仍含禁用算子：{report['forbidden_present']}"
    print(f"  禁用算子 {'/'.join(FORBIDDEN_OPS)}：0 个")

    unknown = set(report["ops"]) - SUPPORTED_IN_DEPLOY_GRAPH
    assert not unknown, f"部署图出现未登记算子：{sorted(unknown)}"
    print(f"  全部 {report['op_count']} 类算子都在已登记的支持表内")
    print("  OK")


def main():
    print("=" * 70)
    print("RKNN 导出路径冒烟测试")
    print("=" * 70)

    check_original_untouched()
    check_state_dict_compatible()
    check_module_equivalence()
    check_model_level_equivalence()
    check_deployable_graph_is_clean()

    print("\n全部通过")


if __name__ == "__main__":
    main()
