"""把 PointEncoder 拆成 NPU 子图导出。

RKNN 不支持 GatherElements（kNN gather），所以把 kNN 搜索+收集留在 CPU，
NPU 只跑稠密 MLP（Linear+LayerNorm+Pow+Tanh）。

拆出的子图：
  sa0, sa1, sa2    — SetAbstraction 的 MLP+maxpool
  fp0, fp1, fp2    — FeaturePropagation 的 MLP
  head             — semantic+instance+text 三个头
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.point_encoder import PointEncoderConfig, PointEncoder, make_activation


class SA_NPU(nn.Module):
    """NPU 端：输入已收集好的邻居包，输出 max-pool 后特征。"""
    def __init__(self, mlp: nn.Sequential):
        super().__init__()
        self.mlp = mlp

    def forward(self, neighbour_package):
        # neighbour_package: (B, Q, k, C_in+3)
        hidden = self.mlp(neighbour_package)   # (B, Q, k, C_out)
        return hidden.max(dim=2).values        # (B, Q, C_out)


class FP_NPU(nn.Module):
    """NPU 端：输入插值特征 + skip，输出 MLP 结果。"""
    def __init__(self, mlp: nn.Sequential):
        super().__init__()
        self.mlp = mlp

    def forward(self, interpolated, skip):
        # interpolated: (B, Q, C_coarse)
        # skip: (B, Q, C_skip)
        return self.mlp(torch.cat([interpolated, skip], dim=-1))


class Head_NPU(nn.Module):
    """NPU 端：三个头。"""
    def __init__(self, semantic, instance, text, text_norm):
        super().__init__()
        self.semantic = semantic
        self.instance = instance
        self.text = text
        self.text_norm = text_norm

    def forward(self, features):
        return (
            self.semantic(features),
            self.instance(features),
            self.text_norm(self.text(features)),
        )


def export_one(model, dummy_inputs, path, opset=12):
    """导出单个子图为 ONNX。"""
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(path),
            input_names=[f'input_{i}' for i in range(len(dummy_inputs) if isinstance(dummy_inputs, tuple) else 1)],
            output_names=['output'],
            opset_version=opset,
            do_constant_folding=True,
        )
    size_kb = path.stat().st_size / 1024
    print(f'  [ONNX] {path.name:30s} {size_kb:8.1f} KB')
    return size_kb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', default='/tmp/rknn_subgraphs')
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = PointEncoderConfig(**ckpt['config'])
    encoder = PointEncoder(cfg)
    encoder.load_state_dict(ckpt['model'])
    encoder.eval()

    report = {'subgraphs': {}}
    B = args.batch

    # ---- SA 子图 ----
    sa_shapes = [
        # (Q, k, C_in)  for neighbour_package (C_in+3 channels)
        (2048, 16, cfg.stem_channels),      # sa0: 64 -> 128
        (512,  16, cfg.level_channels[0]),  # sa1: 128 -> 256
        (128,  12, cfg.level_channels[1]),  # sa2: 256 -> 512
    ]
    for i, (Q, k, C_in) in enumerate(sa_shapes):
        mlp = encoder.encoders[i].mlp
        sa_npu = SA_NPU(mlp)
        dummy = torch.randn(B, Q, k, C_in + 3)
        path = out_dir / f'sa{i}_npu.onnx'
        export_one(sa_npu, dummy, path)
        report['subgraphs'][f'sa{i}'] = {'Q': Q, 'k': k, 'C_in': C_in, 'C_out': cfg.level_channels[i], 'path': str(path)}

    # ---- FP 子图 ----
    fp_cfgs = [
        # (coarse, skip, out)
        (cfg.level_channels[2], cfg.level_channels[1], cfg.decoder_channels[0]),  # fp0: 512+256->256
        (cfg.decoder_channels[0], cfg.level_channels[0], cfg.decoder_channels[1]), # fp1: 256+128->128
        (cfg.decoder_channels[1], cfg.stem_channels, cfg.decoder_channels[2]),     # fp2: 128+64->128
    ]
    fp_Qs = [128, 512, 2048]  # 对应上采样后的点数
    for i, (C_coarse, C_skip, C_out) in enumerate(fp_cfgs):
        mlp = encoder.decoders[i].mlp
        fp_npu = FP_NPU(mlp)
        Q = fp_Qs[i]
        dummy = (torch.randn(B, Q, C_coarse), torch.randn(B, Q, C_skip))
        path = out_dir / f'fp{i}_npu.onnx'
        export_one(fp_npu, dummy, path)
        report['subgraphs'][f'fp{i}'] = {'Q': Q, 'C_coarse': C_coarse, 'C_skip': C_skip, 'C_out': C_out, 'path': str(path)}

    # ---- Head ----
    head_npu = Head_NPU(encoder.semantic_head, encoder.instance_head,
                        encoder.text_head, encoder.text_normalise)
    dummy = torch.randn(B, 2048, cfg.decoder_channels[-1])
    path = out_dir / 'head_npu.onnx'
    export_one(head_npu, dummy, path)
    report['subgraphs']['head'] = {'Q': 2048, 'C_in': cfg.decoder_channels[-1], 'path': str(path)}

    report_path = out_dir / 'subgraph_report.json'
    report_path.write_text(json.dumps(report, indent=2))
    print(f'\n报告：{report_path}')
    print(f'共 {len(report["subgraphs"])} 个子图 ONNX')


if __name__ == '__main__':
    main()
