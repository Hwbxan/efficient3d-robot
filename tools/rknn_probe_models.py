"""生成用于 RKNN 转换探测的小 ONNX —— 逐个测「改写后的模块」能不能过 RKNN。

背景
----
`RKNN算子审计报告_2026-09-19.md` 把 `Pow(·,-0.5)` 与 `Tanh` 列为**整个方案的支点**：
`Pow` 若不支持，LayerNorm 就得留 CPU，改写方案要重做。这份脚本把支点拆成
最小可验证单元，每个模块单独导出一个 ONNX，交给 `rknn_toolkit_probe.py` 去转。

用**真实模块**而不是手搓 ONNX：这样测的就是我们真正要部署的那几个类，
而不是"我以为等价"的近似物。

用法（在项目根目录，用**有 torch 的环境**跑）：
    python -m tools.rknn_probe_models --output-directory /tmp/rknn_probe

产出（每个都是一个独立的小 ONNX）：
    pow_neg_half.onnx            —— Pow(x+1, -0.5)，LayerNorm 的支点
    tanh.onnx                    —— Tanh，GELU 近似的支点
    layernorm_rknn.onnx          —— RKNNFriendlyLayerNorm（完整 LayerNorm）
    gelu_tanh_rknn.onnx          —— RKNNFriendlyGELU（完整 GELU tanh 近似）
    l2norm_rknn.onnx             —— RKNNFriendlyL2Normalise
    idw_rknn.onnx                —— RKNNFriendlyInverseDistanceWeight
    block_mlp.onnx               —— Gemm + LayerNorm + GELU 的典型块（真实结构）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.rknn_export import (  # noqa: E402
    RKNNFriendlyGELU,
    RKNNFriendlyInverseDistanceWeight,
    RKNNFriendlyL2Normalise,
    RKNNFriendlyLayerNorm,
)


def export_simple(model: nn.Module, sample: torch.Tensor, path: str, opset: int = 12):
    """给探测模块用的最小导出。

    **不能复用 `rknn_export.export_onnx`** —— 那个会把输出包成
    `PointEncoderOutput` 的 tuple，而这里的探测模块直接返回张量。
    """

    torch.onnx.export(
        model.eval(),
        (sample,),
        path,
        input_names=["features"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    return path


class PowNegHalf(nn.Module):
    """LayerNorm 的核心：`Pow(var + eps, -0.5)`。

    单独抽出来测，是因为**整个改写方案都压在这个算子上**。
    指数必须是 Python float —— 写成张量或换成 `rsqrt` 会分解成 Sqrt/Div。
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = (x * x).mean(dim=-1, keepdim=True)
        return x * torch.pow(variance + self.eps, -0.5)


class TanhOnly(nn.Module):
    """GELU tanh 近似里的那个 tanh。"""

    def forward(self, x):
        return torch.tanh(x)


class IDWWrapper(nn.Module):
    """反距离加权要两个输入，包一层固定 query。"""

    def __init__(self, neighbours: int = 8, queries: int = 16):
        super().__init__()
        self.register_buffer("query", torch.randn(1, queries, 3) * 0.1)

    def forward(self, neighbour_xyz):
        return RKNNFriendlyInverseDistanceWeight()(neighbour_xyz, self.query)


class BlockMLP(nn.Module):
    """真实结构的最小复刻：Linear → LayerNorm → GELU → Linear。

    这一条链把 §6.1 的三个替换模块串起来，是「NPU 主图」的基本单元。
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        # 注意：RKNNFriendlyLayerNorm 的签名是 (weight, bias, eps)，
        # 不是 nn.LayerNorm 的 (normalized_shape)。直接传 int 会 TypeError。
        self.norm = RKNNFriendlyLayerNorm(
            torch.ones(channels), torch.zeros(channels), 1e-5
        )
        self.act = RKNNFriendlyGELU()
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x):
        return self.fc2(self.act(self.norm(self.fc1(x))))


PROBES = {
    # 名字 -> (模块工厂, 输入形状, 说明)
    #
    # 注意 RKNNFriendlyLayerNorm 的构造签名是 (weight, bias, eps) ——
    # 它不是像 nn.LayerNorm 那样自己造参数，而是在 to_rknn_friendly 里
    # 从已有的 nn.LayerNorm 搬参数过来（这样 state_dict 键才不变）。
    # 所以这里显式传一组单位参数。
    #
    # 另一个坑：它在 **最后一维** 做归一化，所以输入最后一维必须等于
    # weight 的长度。写成 (1, 64, 32) 而 weight 长 64 会在 forward 里炸
    # （broadcast 失败），故这里用 (1, 32, 64)。
    "pow_neg_half": (lambda: PowNegHalf(), (1, 64, 32), "Pow(x+eps, -0.5) —— LayerNorm 支点"),
    "tanh": (lambda: TanhOnly(), (1, 64, 32), "Tanh —— GELU 近似支点"),
    "layernorm_rknn": (
        lambda: RKNNFriendlyLayerNorm(torch.ones(64), torch.zeros(64), 1e-5),
        (1, 32, 64), "完整 LayerNorm 改写"),
    "gelu_tanh_rknn": (lambda: RKNNFriendlyGELU(), (1, 64, 32), "完整 GELU tanh 近似"),
    "l2norm_rknn": (lambda: RKNNFriendlyL2Normalise(dim=-1), (1, 64, 32), "L2 归一化改写"),
    "idw_rknn": (lambda: IDWWrapper(), (1, 16, 8, 3), "反距离加权改写"),
    "block_mlp": (lambda: BlockMLP(64), (1, 64, 64), "Linear+LayerNorm+GELU+Linear"),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-directory", default="/tmp/rknn_probe")
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_arguments()
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("生成 RKNN 转换探测模型")
    print("=" * 78)

    manifest = []
    for name, (factory, shape, note) in PROBES.items():
        torch.manual_seed(0)
        module = factory().eval()
        sample = torch.randn(*shape)
        path = out / f"{name}.onnx"
        try:
            export_simple(module, sample, str(path), opset=args.opset)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name:<16} 导出失败：{type(exc).__name__}: {exc}")
            continue

        with torch.no_grad():
            reference = module(sample)
        if isinstance(reference, (tuple, list)):
            reference = reference[0]
        np.save(out / f"{name}.npy", reference.numpy())
        # 输入也要存：rknn-toolkit2 的模拟器比对需要它
        np.save(out / f"{name}_input.npy", sample.numpy())

        size_kb = path.stat().st_size / 1024
        print(f"  [ OK ] {name:<16} {str(path):<44} {size_kb:7.1f} KB   {note}")
        manifest.append({
            "name": name,
            "onnx": str(path),
            "reference_npy": str(out / f"{name}.npy"),
            "input_npy": str(out / f"{name}_input.npy"),
            "input_shape": list(shape),
            "note": note,
        })

    print(f"\n共生成 {len(manifest)} 个探测模型，输出目录：{out}")
    print("下一步（用装了 rknn-toolkit2 的环境跑）：")
    print(f"  python -m tools.rknn_toolkit_probe --probe-directory {out}")
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
