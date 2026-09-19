"""在 x86 上用 rknn-toolkit2 把 ONNX 转成 RK3588 的 .rknn。

**为什么在服务器上就能做**：rknn-toolkit2 是**编译器**，官方支持 x86_64 Ubuntu。
真正需要板子的只有**推理**那一步（板子上装 rknn-toolkit-lite2 + librknnrt）。
所以「能不能转、哪些算子掉 CPU、INT8 掉多少精度」这三件事现在就能全部拿到答案。

用法（必须用装了 rknn-toolkit2 的环境）：
    # 1. 逐个探测改写后的模块，先确认支点算子
    python -m tools.rknn_toolkit_probe --probe-directory /tmp/rknn_probe

    # 2. 转整张部署图
    python -m tools.rknn_toolkit_probe --onnx deploy/point_encoder_rknn_deploy.onnx \
        --output deploy/point_encoder_rk3588.rknn

    # 3. 带 INT8 量化（需要标定集，见 tools/build_rknn_calibration_set.py）
    python -m tools.rknn_toolkit_probe --onnx ... --output ... \
        --calibration-dataset outputs/rknn_calibration/dataset.txt

每个模型都会报告：转换是否成功、哪些算子被丢到 CPU、以及（可选）模拟器上的精度偏差。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# rknn-toolkit2 会往 stdout 打大量日志，这些正则用来从中捞出关键结论
CPU_FALLBACK_PATTERNS = (
    re.compile(r"not supported.*?(?:CPU|on CPU)", re.IGNORECASE),
    re.compile(r"will be executed on CPU", re.IGNORECASE),
    re.compile(r"Warning:.*?unsupported", re.IGNORECASE),
)


def _capture(function, *args, **kwargs):
    """跑一段会狂打日志的代码，返回 (结果, 日志文本)。"""

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


def _extract_cpu_ops(log: str):
    lines = []
    for line in log.splitlines():
        if any(pattern.search(line) for pattern in CPU_FALLBACK_PATTERNS):
            cleaned = line.strip()
            if cleaned and cleaned not in lines:
                lines.append(cleaned)
    return lines


def convert_one(onnx_path: str, output_path: str, target: str = "rk3588",
                calibration: str = None, do_quantization: bool = False,
                verbose: bool = False):
    """转一个 ONNX，返回结果字典。"""

    from rknn.api import RKNN

    result = {
        "onnx": onnx_path,
        "rknn": output_path,
        "target": target,
        "quantized": bool(do_quantization and calibration),
        "ok": False,
        "error": None,
        "cpu_ops": [],
        "log_excerpt": "",
    }

    rknn = RKNN(verbose=verbose)
    try:
        code, log = _capture(
            rknn.config,
            target_platform=target,
            # 输入已经是归一化后的特征（xyz + normal + rgb），不需要 toolkit 再做 mean/std
            mean_values=None,
            std_values=None,
            optimization_level=3,
        )
        result["log_excerpt"] += log
        if code != 0:
            raise RuntimeError(f"rknn.config 返回 {code}")

        code, log = _capture(rknn.load_onnx, model=onnx_path)
        result["log_excerpt"] += log
        if code != 0:
            raise RuntimeError(f"rknn.load_onnx 返回 {code}")

        build_kwargs = {"do_quantization": result["quantized"]}
        if result["quantized"]:
            build_kwargs["dataset"] = calibration
        code, log = _capture(rknn.build, **build_kwargs)
        result["log_excerpt"] += log
        if code != 0:
            raise RuntimeError(f"rknn.build 返回 {code}")

        code, log = _capture(rknn.export_rknn, export_path=output_path)
        result["log_excerpt"] += log
        if code != 0:
            raise RuntimeError(f"rknn.export_rknn 返回 {code}")

        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            rknn.release()

    result["cpu_ops"] = _extract_cpu_ops(result["log_excerpt"])
    if result["ok"] and os.path.exists(output_path):
        result["size_kb"] = round(os.path.getsize(output_path) / 1024, 1)
    return result


def check_accuracy(onnx_path: str, reference_npy: str, rknn_path: str,
                   input_npy: str = None):
    """用 rknn-toolkit2 的**模拟器**跑一遍，和 PyTorch 参考输出比对。

    ⚠️ 模拟器不等于真机：它只在 x86 上模拟 NPU 的算子行为，
    INT8 的量化误差趋势可参考，但**延迟和真实精度必须在板子上测**。
    """

    from rknn.api import RKNN

    data = np.load(reference_npy)
    if input_npy and os.path.exists(input_npy):
        inputs = [np.load(input_npy)]
    else:
        return {"ok": False, "error": "缺少输入 npy，无法做模拟器比对"}

    rknn = RKNN(verbose=False)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rknn.config(target_platform="rk3588", optimization_level=3)
            if rknn.load_rknn(rknn_path) != 0:
                return {"ok": False, "error": "load_rknn 失败"}
            if rknn.init_runtime() != 0:
                return {"ok": False, "error": "init_runtime 失败（模拟器不可用）"}
        outputs = rknn.inference(inputs=inputs)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            rknn.release()

    if not outputs:
        return {"ok": False, "error": "模拟器没有返回输出"}

    predicted = outputs[0]
    reference = data if data.shape == predicted.shape else data.reshape(predicted.shape)
    difference = np.abs(predicted.astype(np.float64) - reference.astype(np.float64))
    scale = np.maximum(np.abs(reference.astype(np.float64)), 1e-6)
    return {
        "ok": True,
        "max_abs_diff": float(difference.max()),
        "max_rel_diff": float((difference / scale).max()),
        "note": "模拟器结果，不是真机实测",
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe-directory", default=None,
                        help="跑 rknn_probe_models 产出的目录（逐个模块探测）")
    parser.add_argument("--onnx", default=None, help="转单个 ONNX")
    parser.add_argument("--output", default=None, help=".rknn 输出路径")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--calibration-dataset", default=None,
                        help="INT8 标定集清单（dataset.txt）；给了就开量化")
    parser.add_argument("--check-accuracy", action="store_true",
                        help="用模拟器比对精度（需要同目录下有 .npy 参考输出）")
    parser.add_argument("--json", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.probe_directory and not args.onnx:
        print("要么给 --probe-directory，要么给 --onnx", file=sys.stderr)
        return 2

    jobs = []
    if args.probe_directory:
        probe_dir = Path(args.probe_directory)
        for onnx_path in sorted(probe_dir.glob("*.onnx")):
            jobs.append((str(onnx_path), str(onnx_path.with_suffix(".rknn"))))
    if args.onnx:
        out = args.output or str(Path(args.onnx).with_suffix(".rknn"))
        jobs.append((args.onnx, out))

    print("=" * 78)
    print("RKNN 转换探测 —— target_platform = %s" % args.target)
    print("=" * 78)
    if args.calibration_dataset:
        print("INT8 量化：开（标定集 %s）" % args.calibration_dataset)
    else:
        print("INT8 量化：关（只测算子支持性）")

    results = []
    for onnx_path, rknn_path in jobs:
        name = Path(onnx_path).stem
        print(f"\n--- {name} ---")
        result = convert_one(
            onnx_path, rknn_path, target=args.target,
            calibration=args.calibration_dataset,
            do_quantization=bool(args.calibration_dataset),
            verbose=args.verbose,
        )
        result["name"] = name
        if result["ok"]:
            print(f"  [ OK ] 转换成功  {result.get('size_kb', '?')} KB  -> {rknn_path}")
        else:
            print(f"  [FAIL] {result['error']}")
        if result["cpu_ops"]:
            print(f"  掉到 CPU 的算子（{len(result['cpu_ops'])} 条警告）：")
            for line in result["cpu_ops"][:5]:
                print(f"    {line[:150]}")
        else:
            print("  没有检测到 CPU 回退警告")

        if args.check_accuracy and result["ok"]:
            probe_dir = Path(onnx_path).parent
            ref = probe_dir / f"{name}.npy"
            input_npy = probe_dir / f"{name}_input.npy"
            if ref.exists():
                accuracy = check_accuracy(onnx_path, str(ref), rknn_path,
                                          input_npy=str(input_npy))
                result["simulator_accuracy"] = accuracy
                if accuracy["ok"]:
                    print(f"  模拟器偏差：max_abs={accuracy['max_abs_diff']:.3e} "
                          f"max_rel={accuracy['max_rel_diff']:.3e}（不是真机）")
                else:
                    print(f"  模拟器比对跳过：{accuracy['error']}")

        results.append(result)

    succeeded = sum(1 for r in results if r["ok"])
    print(f"\n{'=' * 78}")
    print(f"汇总：{succeeded} / {len(results)} 个模型转换成功")
    print("=" * 78)
    for r in results:
        mark = "OK  " if r["ok"] else "FAIL"
        extra = "" if r["ok"] else f"  {r['error']}"
        print(f"  [{mark}] {r['name']:<20}{extra}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"target": args.target, "results": results},
                      handle, indent=2, ensure_ascii=False)
        print(f"\n已写入 {args.json}")

    print("\n提醒：这只是 x86 上的**编译**结果。延迟与真实精度必须在 RK3588 上测 ——")
    print("      板子只需要装 rknn-toolkit-lite2 + librknnrt，不需要 toolkit2。")
    return 0 if succeeded == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
