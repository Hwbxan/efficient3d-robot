"""在 RK3588 板子上测 .rknn 的延迟与精度。

**这个脚本要在板子上跑**，不是服务器上。板子只需要 `rknn-toolkit-lite2` +
`librknnrt.so`，**不需要** `rknn-toolkit2`（后者是 x86 上的编译器）。

用法（板子上）：
    # 1. 先确认环境
    python3 rk3588_benchmark.py --check-only

    # 2. 测延迟（3 个 NPU 核都试一遍）
    python3 rk3588_benchmark.py --model point_encoder_rk3588_int8.rknn \
        --input calib_0000.npy --iterations 200 --warmup 20

    # 3. 顺便验精度（给 PyTorch 的参考输出）
    python3 rk3588_benchmark.py --model ... --input ... \
        --reference reference_output.npy

注意 RK3588 有 3 个 NPU 核，`--core-mask` 决定用几个：
    0 = 自动（默认，按负载挑核）    1 / 2 / 4 = 单独用 core0/1/2
    7 = 三个核一起（吞吐最高，但单次延迟可能变差）
多核并行只对**批处理**有意义；单帧推理通常用 auto 或单核延迟更低。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def check_environment():
    """先确认板子上的运行时是不是齐的。缺 librknnrt 是最常见的坑。"""

    report = {
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }

    try:
        from rknnlite.api import RKNNLite
        report["rknnlite"] = "OK"
        report["rknnlite_version"] = getattr(RKNNLite, "__version__", "未知")
    except Exception as exc:  # noqa: BLE001
        report["rknnlite"] = f"缺失：{type(exc).__name__}: {exc}"

    # librknnrt.so 是运行时本体，版本必须与编译用的 toolkit2 版本一致
    candidates = [
        "/usr/lib/librknnrt.so",
        "/usr/lib/aarch64-linux-gnu/librknnrt.so",
        "/usr/local/lib/librknnrt.so",
    ]
    found = [path for path in candidates if os.path.exists(path)]
    report["librknnrt"] = found or "未在常见路径找到（可能装在别处）"

    # 版本不匹配会直接报 "librknnrt version mismatch"
    report["note"] = (
        "librknnrt.so 的版本必须与编译 .rknn 用的 rknn-toolkit2 一致。"
        "不一致时报 version mismatch，换 so 或重转模型都行。"
    )
    return report


def load_rknn(model_path: str, core_mask: int):
    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(model_path) != 0:
        raise RuntimeError(f"load_rknn 失败：{model_path}")
    if rknn.init_runtime(core_mask=core_mask) != 0:
        raise RuntimeError(f"init_runtime 失败（core_mask={core_mask}）")
    return rknn


def benchmark(model_path: str, inputs, iterations: int, warmup: int, core_mask: int):
    rknn = load_rknn(model_path, core_mask)
    try:
        # warmup：第一次推理包含权重搬运与 NPU 初始化，不丢会严重高估
        for _ in range(warmup):
            rknn.inference(inputs=inputs)

        latencies = []
        outputs = None
        for _ in range(iterations):
            start = time.perf_counter()
            outputs = rknn.inference(inputs=inputs)
            latencies.append((time.perf_counter() - start) * 1000.0)
    finally:
        rknn.release()

    latencies.sort()
    return {
        "iterations": iterations,
        "warmup": warmup,
        "core_mask": core_mask,
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 2),
        "min_ms": round(latencies[0], 2),
        "max_ms": round(latencies[-1], 2),
        "fps_median": round(1000.0 / statistics.median(latencies), 2),
    }, outputs


def compare(reference_path: str, outputs):
    reference = np.load(reference_path)
    predicted = outputs[0]
    if reference.shape != predicted.shape:
        reference = reference.reshape(predicted.shape)
    difference = np.abs(predicted.astype(np.float64) - reference.astype(np.float64))
    scale = np.maximum(np.abs(reference.astype(np.float64)), 1e-6)
    return {
        "max_abs_diff": float(difference.max()),
        "max_rel_diff": float((difference / scale).max()),
        "mean_abs_diff": float(difference.mean()),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help=".rknn 文件")
    parser.add_argument("--input", default=None, help="输入 npy（形状含 batch 维）")
    parser.add_argument("--reference", default=None,
                        help="PyTorch 的参考输出 npy，用于算精度偏差")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--core-mask", type=int, default=0,
                        help="0=自动 1/2/4=单核 7=三核")
    parser.add_argument("--sweep-cores", action="store_true",
                        help="把 0/1/2/4/7 都测一遍")
    parser.add_argument("--check-only", action="store_true",
                        help="只检查环境，不跑推理")
    parser.add_argument("--json", default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 78)
    print("RK3588 板端推理基准")
    print("=" * 78)

    environment = check_environment()
    for key, value in environment.items():
        print(f"  {key:<18} {value}")

    if args.check_only:
        if args.json:
            Path(args.json).write_text(
                json.dumps({"environment": environment}, indent=2, ensure_ascii=False),
                encoding="utf-8")
        return 0

    if not args.model or not args.input:
        print("\n需要 --model 与 --input", file=sys.stderr)
        return 2

    inputs = [np.load(args.input)]
    print(f"\n模型：{args.model}")
    print(f"输入：{args.input}  形状 {inputs[0].shape}")

    masks = [0, 1, 2, 4, 7] if args.sweep_cores else [args.core_mask]
    results = {}
    last_outputs = None
    for mask in masks:
        print(f"\n--- core_mask = {mask} ---")
        try:
            stats, last_outputs = benchmark(
                args.model, inputs, args.iterations, args.warmup, mask)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            results[str(mask)] = {"error": str(exc)}
            continue
        print(f"  中位数 {stats['median_ms']} ms   均值 {stats['mean_ms']} ms   "
              f"p95 {stats['p95_ms']} ms   → {stats['fps_median']} FPS")
        results[str(mask)] = stats

    payload = {
        "environment": environment,
        "model": args.model,
        "input_shape": list(inputs[0].shape),
        "results": results,
        "note": "板端实测。功耗需要 USB-C 功率计，本脚本测不到。",
    }
    if args.reference and last_outputs is not None:
        payload["accuracy"] = compare(args.reference, last_outputs)
        print(f"\n精度偏差：max_abs={payload['accuracy']['max_abs_diff']:.3e} "
              f"max_rel={payload['accuracy']['max_rel_diff']:.3e}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
