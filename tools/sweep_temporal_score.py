"""离线测试「时间一致性」是否能提升实例评分的排序质量。

直觉：Grounding DINO 的单帧误检（对着墙面/背景给出一个 0.4 分的 "door"）
往往只出现一两帧；而真正的物体会在连续多帧被重复观测到。把"这条轨迹被
观测了多少次"并入置信度，是视频/多帧 3D 流水线相对于单帧检测器独有的
信息优势。

本脚本不重跑流水线，仅用现有 tracking.json + segmentation 重算 AP。

用法：
python -m tools.sweep_temporal_score \
    --tracking-json <run>/association/tracking.json \
    --segmentation-root <run>/segmentation \
    --gt-root outputs/gt/office0 \
    --gt-classes bin,chair,door,sofa,table,tv-screen \
    --max-area-fraction 0.4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate_against_gt import average_precision
from tools.sweep_score_formula import load_instances


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-json", required=True)
    parser.add_argument("--segmentation-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--gt-classes", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-gt-area", type=int, default=100)
    parser.add_argument("--max-area-fraction", type=float, default=0.4)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--image-width", type=int, default=1200)
    parser.add_argument("--image-height", type=int, default=680)
    args = parser.parse_args()

    image_area = args.image_width * args.image_height
    gt_root = Path(args.gt_root)
    manifest = json.load((gt_root / "gt_manifest.json").open(encoding="utf-8"))
    object_to_class = {int(k): v for k, v in manifest["object_id_to_class"].items()}
    class_filter = (
        {item.strip() for item in args.gt_classes.split(",") if item.strip()}
        if args.gt_classes
        else None
    )

    tracking_raw = json.load(open(args.tracking_json, encoding="utf-8"))
    # global_id → 被观测次数
    counts = {}
    for frame in tracking_raw["frames"]:
        for item in frame["associations"]:
            gid = item.get("global_id")
            if gid is not None:
                counts[gid] = counts.get(gid, 0) + 1
    tracking = {
        frame["frame_index"]: {
            item["local_instance_id"]: item.get("global_id")
            for item in frame["associations"]
        }
        for frame in tracking_raw["frames"]
    }
    frames = sorted(tracking)

    cache = []
    total_gt = 0
    for frame_index in frames:
        gt_image = cv2.imread(
            str(gt_root / "instance_masks" / f"instance{frame_index:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if gt_image is None:
            continue
        gt_ids = [
            int(v) for v in np.unique(gt_image)
            if v != 0
            and (gt_image == v).sum() >= args.min_gt_area
            and (class_filter is None or object_to_class.get(int(v)) in class_filter)
        ]
        total_gt += len(gt_ids)
        predictions = load_instances(
            frame_index, Path(args.segmentation_root), tracking[frame_index],
            args.min_score, args.max_area_fraction, image_area,
        )
        cache.append((frame_index, predictions, gt_image, gt_ids))

    # 预计算 best_iou 与基础分
    entries = []
    for _, predictions, gt_image, gt_ids in cache:
        gt_masks = {gid: gt_image == gid for gid in gt_ids}
        for gid, value in predictions.items():
            best_iou = 0.0
            for _, gt_mask in gt_masks.items():
                inter = int(np.logical_and(value["mask"], gt_mask).sum())
                union = value["area"] + int(gt_mask.sum()) - inter
                if union:
                    best_iou = max(best_iou, inter / union)
            entries.append({
                "dino": value["dino"],
                "sam": value["sam"],
                "obs": counts.get(gid, 1),
                "best": best_iou,
            })

    print(f"帧数 {len(cache)}  GT {total_gt}  预测 {len(entries)}")
    obs_array = np.array([e["obs"] for e in entries])
    print(f"观测次数分布: 1次={int((obs_array == 1).sum())}  "
          f"2次={int((obs_array == 2).sum())}  "
          f">=5次={int((obs_array >= 5).sum())}")
    faux = [e for e in entries if e["best"] < 0.01]
    real = [e for e in entries if e["best"] >= 0.25]
    print(f"纯误检(IoU<0.01) 平均观测次数={np.mean([e['obs'] for e in faux]):.2f}  "
          f"真阳性 平均观测次数={np.mean([e['obs'] for e in real]):.2f}\n")

    formulas = {
        "dino (基准)": lambda e: e["dino"],
        "sam": lambda e: e["sam"],
        "dino*log(1+obs)": lambda e: e["dino"] * np.log1p(e["obs"]),
        "dino*sqrt(obs)": lambda e: e["dino"] * np.sqrt(e["obs"]),
        "dino*obs/(obs+2)": lambda e: e["dino"] * e["obs"] / (e["obs"] + 2.0),
        "dino+0.02*(obs-1)": lambda e: e["dino"] + 0.02 * (e["obs"] - 1),
        "dino+0.05*(obs-1)": lambda e: e["dino"] + 0.05 * (e["obs"] - 1),
        "dino*sam*log(1+obs)": lambda e: e["dino"] * e["sam"] * np.log1p(e["obs"]),
        "sam*log(1+obs)": lambda e: e["sam"] * np.log1p(e["obs"]),
        "sam*obs/(obs+2)": lambda e: e["sam"] * e["obs"] / (e["obs"] + 2.0),
    }

    print(f"{'公式':<24}" + "".join(f"AP@{t:<8}" for t in args.iou_thresholds) + "之和")
    results = {}
    for name, formula in formulas.items():
        row = {}
        for threshold in args.iou_thresholds:
            records = [(float(formula(e)), e["best"]) for e in entries]
            row[f"ap_{threshold}"] = average_precision(records, total_gt, threshold)["ap"]
        results[name] = row
        total = sum(row.values())
        print(f"{name:<24}" + "".join(f"{row[f'ap_{t}']:<12.4f}" for t in args.iou_thresholds)
              + f"{total:.4f}")

    best = max(results, key=lambda k: sum(results[k].values()))
    print(f"\n最佳（AP 之和）: {best} -> {results[best]}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump({"total_gt": total_gt, "results": results, "best": best},
                      file, indent=2, ensure_ascii=False)
        print(f"报告: {args.output}")


if __name__ == "__main__":
    main()
