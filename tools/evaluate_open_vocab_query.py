"""用 GT 语义类别评测开放词汇文本查询的检索质量。

背景：query_instances.py 只能给出排序，无法回答"到底查得准不准"。
本脚本用 Replica 语义网格生成的 GT 实例掩码，为每个融合实例
反推一个**独立于检测器**的 GT 类别，再按类别算检索 AP。

GT 类别反推流程：
  对每一帧，把 pipeline 的检测掩码按 tracking.json 合并成预测实例，
  与该帧的 GT 标签图（像素值 = GT object_id）算 IoU；
  以 IoU × 交集面积 为权重投票，取累计权重最高的 object_id 作为该实例的 GT 物体，
  再经 gt_manifest.json 的 object_id_to_class 得到 GT 类别名。

指标（每个查询）：
  AP@k     —— VOC 式逐步插值，在完整排序上计算（不依赖人工阈值）
  P@1/P@3  —— 头部精度，直接反映"能不能用"
  recall   —— 该类别全部实例中被检索到的比例

用法（在项目根目录）：
python -m tools.evaluate_open_vocab_query \
    --query-results outputs/experiments/office0_fixed_detector_v3/embeddings/query_results.json \
    --tracking-json outputs/experiments/office0_fixed_detector_v3/tracking.json \
    --segmentation-root outputs/experiments/office0_fixed_detector_v3/segmentation \
    --gt-root outputs/gt/office0 \
    --output outputs/experiments/office0_fixed_detector_v3/open_vocab_evaluation.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# 查询词 → 可接受的 GT 类别。Replica 的类别名与检测器词表并不一致，
# 例如 detector 说 "desk"，Replica 标注是 "table"。
DEFAULT_TARGETS = {
    "chair": ["chair", "sofa"],
    "sofa": ["sofa", "chair"],
    "couch": ["sofa", "chair"],
    "trash can": ["bin", "tissue-paper"],
    "garbage bin": ["bin", "tissue-paper"],
    "computer monitor": ["tv-screen", "tablet"],
    "monitor": ["tv-screen", "tablet"],
    "desk": ["table", "desk-organizer", "panel"],
    "table": ["table", "desk-organizer", "panel"],
    "door": ["door"],
    "plant": ["indoor-plant", "plant-stand"],
    "rug": ["rug"],
}

# GT 类别 → 可接受的概念集合，用于算零样本概念指派精度。
# 用集合而非单一概念：Replica 同时有 chair 和 sofa 两个类别，而词表里
# 两者都在，把 sofa 判成 "sofa" 不应算错。集合定义与
# evaluate_against_gt.py 的 LABEL_SYNONYMS 保持一致的宽松度。
GT_CLASS_TO_CONCEPTS = {
    "chair": {"chair", "sofa"},
    "sofa": {"sofa", "chair"},
    "table": {"desk"},
    "desk-organizer": {"desk"},
    "panel": {"desk"},
    "bin": {"trash can"},
    "tissue-paper": {"trash can"},
    "tv-screen": {"computer monitor"},
    "tablet": {"computer monitor"},
    "door": {"door"},
}

# 单一规范概念，用于"严格"精度
GT_CLASS_TO_CONCEPT = {
    gt_class: sorted(concepts)[0] for gt_class, concepts in GT_CLASS_TO_CONCEPTS.items()
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query-results", required=True)
    parser.add_argument("--tracking-json", required=True)
    parser.add_argument("--segmentation-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-gt-area", type=int, default=100)
    parser.add_argument("--min-overlap-iou", type=float, default=0.10,
                        help="单帧投票的最低 IoU，低于此值视为误匹配不投票")
    parser.add_argument("--assign", action="append", default=[],
                        help='为查询显式指定 GT 类别，形如 "a place to sit=chair|sofa"')
    return parser.parse_args()


def load_tracking(path):
    data = json.load(open(path, encoding="utf-8"))
    return {
        frame["frame_index"]: {
            item["local_instance_id"]: item.get("global_id")
            for item in frame["associations"]
        }
        for frame in data["frames"]
    }


def build_gt_votes(tracking, segmentation_root, gt_root, min_gt_area, min_overlap_iou):
    """按帧做 IoU 投票，返回 {global_id: {gt_object_id: weight}}。"""

    votes = {}
    for frame_index in sorted(tracking):
        gt_path = Path(gt_root) / "instance_masks" / ("instance%06d.png" % frame_index)
        instances_path = Path(segmentation_root) / ("frame_%06d_instances.json" % frame_index)
        if not gt_path.exists() or not instances_path.exists():
            continue

        gt_image = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if gt_image is None:
            continue

        # 合并同 global_id 的多个检测掩码
        merged = {}
        for item in json.load(open(instances_path, encoding="utf-8")):
            global_id = tracking[frame_index].get(item["local_instance_id"])
            if global_id is None:
                continue
            mask = cv2.imread(item["mask_path"], cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            mask = mask > 0
            if not mask.any():
                continue
            merged[global_id] = (merged[global_id] | mask) if global_id in merged else mask

        for global_id, mask in merged.items():
            overlapping, counts = np.unique(gt_image[mask], return_counts=True)
            bucket = votes.setdefault(global_id, {})
            for object_id, count in zip(overlapping, counts):
                object_id = int(object_id)
                if object_id == 0:
                    continue
                gt_area = int((gt_image == object_id).sum())
                if gt_area < min_gt_area:
                    continue
                union = int(mask.sum()) + gt_area - int(count)
                iou = int(count) / union if union else 0.0
                if iou < min_overlap_iou:
                    continue
                bucket[object_id] = bucket.get(object_id, 0.0) + iou * int(count)

    return votes


def average_precision(relevance, scores):
    """VOC 式逐步插值 AP。relevance: {global_id: bool}，scores: {global_id: float}。"""

    ordered = sorted(scores, key=lambda g: -scores[g])
    total_relevant = sum(1 for value in relevance.values() if value)
    if total_relevant == 0:
        return 0.0, 0, 0
    hits = 0
    precision_sum = 0.0
    for rank, global_id in enumerate(ordered, start=1):
        if relevance.get(global_id, False):
            hits += 1
            precision_sum += hits / rank
    return precision_sum / total_relevant, hits, total_relevant


def main():
    args = parse_arguments()

    query_report = json.load(open(args.query_results, encoding="utf-8"))
    tracking = load_tracking(args.tracking_json)
    manifest = json.load(open(Path(args.gt_root) / "gt_manifest.json", encoding="utf-8"))
    object_to_class = {int(k): v for k, v in manifest["object_id_to_class"].items()}

    votes = build_gt_votes(tracking, args.segmentation_root, args.gt_root,
                           args.min_gt_area, args.min_overlap_iou)

    instance_gt = {}
    for global_id, bucket in sorted(votes.items()):
        best_object = max(bucket, key=lambda key: bucket[key])
        instance_gt[global_id] = {
            "gt_object_id": best_object,
            "gt_class": object_to_class.get(best_object, "unknown"),
            "vote_weight": round(bucket[best_object], 1),
            "runner_up_class": (
                object_to_class.get(sorted(bucket, key=lambda k: -bucket[k])[1], "unknown")
                if len(bucket) > 1 else None
            ),
        }

    # 融合标签 vs GT 类别：与文本查询无关的独立一致性检查
    label_agreement = {}
    fused_vs_gt = []
    for entry in query_report["instance_best_query"]:
        global_id = entry["global_id"]
        gt = instance_gt.get(global_id)
        fused_vs_gt.append({
            "global_id": global_id,
            "fused_label": entry["fused_label"],
            "assigned_concept": entry.get("assigned_concept"),
            "concept_margin": entry.get("concept_margin"),
            "gt_class": gt["gt_class"] if gt else None,
            "gt_object_id": gt["gt_object_id"] if gt else None,
        })
        if gt:
            key = (entry["fused_label"], gt["gt_class"])
            label_agreement[key] = label_agreement.get(key, 0) + 1

    # 零样本概念指派精度：把 GT 类别映射到概念集合后，看 CLIP 的 argmax 对不对。
    # 这是 Stage 4 语义模块的核心指标，与检索 AP 相互独立。
    zero_shot = {"correct": 0, "compatible_correct": 0, "total": 0,
                 "confusion": {}, "rows": []}
    for entry in fused_vs_gt:
        gt_class = entry["gt_class"]
        assigned = entry.get("assigned_concept")
        if gt_class not in GT_CLASS_TO_CONCEPTS or assigned is None:
            continue
        allowed = GT_CLASS_TO_CONCEPTS[gt_class]
        strict = GT_CLASS_TO_CONCEPT[gt_class] == assigned
        compatible = assigned in allowed
        zero_shot["total"] += 1
        zero_shot["correct"] += int(strict)
        zero_shot["compatible_correct"] += int(compatible)
        key = "%s -> %s" % ("|".join(sorted(allowed)), assigned)
        zero_shot["confusion"][key] = zero_shot["confusion"].get(key, 0) + 1
        zero_shot["rows"].append({
            "global_id": entry["global_id"],
            "gt_class": gt_class,
            "allowed_concepts": sorted(allowed),
            "assigned_concept": assigned,
            "concept_margin": entry.get("concept_margin"),
            "strict_correct": strict,
            "compatible": compatible,
        })
    zero_shot["accuracy"] = (
        round(zero_shot["correct"] / zero_shot["total"], 4) if zero_shot["total"] else None
    )
    zero_shot["compatible_accuracy"] = (
        round(zero_shot["compatible_correct"] / zero_shot["total"], 4)
        if zero_shot["total"] else None
    )

    # 查询词 → 目标 GT 类别
    targets = dict(DEFAULT_TARGETS)
    for item in args.assign:
        if "=" not in item:
            raise SystemExit("--assign 需要 query=class1|class2 形式：%s" % item)
        query, classes = item.split("=", 1)
        targets[query.strip().lower()] = [c.strip() for c in classes.split("|") if c.strip()]

    assignment = {entry["global_id"]: entry
                  for entry in query_report.get("instance_assignment", [])}

    query_metrics = []
    for entry in query_report["queries"]:
        query = entry["query"]
        target_classes = targets.get(query.strip().lower())
        if target_classes is None:
            print("跳过查询 \"%s\"：未指定 GT 目标类别（用 --assign 补充）" % query)
            continue

        scores = {int(k): float(v) for k, v in entry["scores"].items()}
        relevance = {
            global_id: (instance_gt.get(global_id, {}).get("gt_class") in target_classes)
            for global_id in scores
        }
        ap, hits, total_relevant = average_precision(relevance, scores)

        ordered = sorted(scores, key=lambda g: -scores[g])
        precision_at = {}
        for k in (1, 3, 5):
            head = ordered[:k]
            precision_at["p@%d" % k] = round(
                sum(1 for g in head if relevance[g]) / k, 4
            )

        # 该查询实际检索到的 GT 类别构成，便于定位"混淆成什么"
        retrieved_classes = {}
        for global_id in ordered[:5]:
            name = instance_gt.get(global_id, {}).get("gt_class", "unmatched")
            retrieved_classes[name] = retrieved_classes.get(name, 0) + 1

        # 接受阈值扫描：query_instances.py 的 z_score 判据到底该取多少。
        # z = (score - mean) / std，mean/std 已由查询工具记录。
        mean, std = entry["statistics"]["mean"], entry["statistics"]["std"]
        sweep = []
        for threshold in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            accepted = [g for g in ordered if (scores[g] - mean) / max(std, 1e-8) >= threshold]
            if accepted:
                precision = sum(1 for g in accepted if relevance[g]) / len(accepted)
            else:
                precision = None
            sweep.append({
                "z_threshold": threshold,
                "accepted": len(accepted),
                "precision": round(precision, 4) if precision is not None else None,
                "recall": round(
                    sum(1 for g in accepted if relevance[g]) / total_relevant, 4
                ) if total_relevant else 0.0,
            })

        # 概念指派 margin 扫描：vocabulary 模式下 margin 取多少合适。
        # 接受条件 = 该实例的 argmax 概念等于查询解析出的概念，且 margin 达标。
        resolved_concept = entry.get("resolved_concept")
        margin_sweep = []
        for margin in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
            accepted = [
                g for g in ordered
                if assignment.get(g, {}).get("assigned_concept") == resolved_concept
                and assignment.get(g, {}).get("concept_margin", 0.0) >= margin
            ]
            precision = (
                sum(1 for g in accepted if relevance[g]) / len(accepted) if accepted else None
            )
            margin_sweep.append({
                "margin": margin,
                "accepted": len(accepted),
                "precision": round(precision, 4) if precision is not None else None,
                "recall": round(sum(1 for g in accepted if relevance[g]) / total_relevant, 4)
                           if total_relevant else 0.0,
            })

        query_metrics.append({
            "query": query,
            "target_gt_classes": target_classes,
            "resolved_concept": resolved_concept,
            "ap": round(ap, 4),
            "hits": hits,
            "relevant_total": total_relevant,
            "recall": round(hits / total_relevant, 4) if total_relevant else 0.0,
            "precision_at": precision_at,
            "top5_gt_classes": retrieved_classes,
            "accepted_by_tool": entry["accepted"],
            "accepted_gt_classes": [
                instance_gt.get(g, {}).get("gt_class", "unmatched") for g in entry["accepted"]
            ],
            "acceptance_sweep": sweep,
            "vocabulary_margin_sweep": margin_sweep,
        })

    report = {
        "query_results": args.query_results,
        "gt_root": args.gt_root,
        "min_overlap_iou": args.min_overlap_iou,
        "instance_gt": {str(k): v for k, v in sorted(instance_gt.items())},
        "fused_vs_gt": fused_vs_gt,
        "zero_shot_assignment": zero_shot,
        "queries": query_metrics,
    }
    report["instances_without_gt"] = [
        entry["global_id"] for entry in fused_vs_gt if entry["gt_class"] is None
    ]
    if query_metrics:
        report["summary"] = {
            "mean_ap": round(float(np.mean([e["ap"] for e in query_metrics])), 4),
            "mean_p1": round(float(np.mean([e["precision_at"]["p@1"] for e in query_metrics])), 4),
            "query_count": len(query_metrics),
        }
        # 跨查询汇总的接受阈值扫描
        report["acceptance_sweep_summary"] = []
        for index, threshold in enumerate((0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
            rows = [e["acceptance_sweep"][index] for e in query_metrics]
            precisions = [r["precision"] for r in rows if r["precision"] is not None]
            report["acceptance_sweep_summary"].append({
                "z_threshold": threshold,
                "mean_accepted": round(float(np.mean([r["accepted"] for r in rows])), 3),
                "mean_precision": round(float(np.mean(precisions)), 4) if precisions else None,
                "mean_recall": round(float(np.mean([r["recall"] for r in rows])), 4),
            })

        report["vocabulary_margin_sweep_summary"] = []
        for index, margin in enumerate((0.0, 0.005, 0.01, 0.015, 0.02, 0.03)):
            rows = [e["vocabulary_margin_sweep"][index] for e in query_metrics]
            precisions = [r["precision"] for r in rows if r["precision"] is not None]
            report["vocabulary_margin_sweep_summary"].append({
                "margin": margin,
                "mean_accepted": round(float(np.mean([r["accepted"] for r in rows])), 3),
                "mean_precision": round(float(np.mean(precisions)), 4) if precisions else None,
                "mean_recall": round(float(np.mean([r["recall"] for r in rows])), 4),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("\n实例 GT 反推：%d / %d 个实例匹配到 GT 物体"
          % (len(instance_gt), len(fused_vs_gt)))
    print("\n%-6s %-14s %-20s %-18s %-6s %s"
          % ("G-ID", "融合标签", "GT 类别", "指派概念", "严格", "兼容"))
    for entry in fused_vs_gt:
        allowed = GT_CLASS_TO_CONCEPTS.get(entry["gt_class"])
        assigned = entry.get("assigned_concept") or "-"
        strict = "-" if allowed is None else ("OK" if GT_CLASS_TO_CONCEPT[entry["gt_class"]] == assigned else "X")
        compatible = "-" if allowed is None else ("OK" if assigned in allowed else "X")
        print("%-6s %-14s %-20s %-18s %-6s %s"
              % ("G%03d" % entry["global_id"], entry["fused_label"],
                 entry["gt_class"] or "-", assigned, strict, compatible))

    print("\n零样本概念指派：严格 %d / %d（%.4f）  兼容 %d / %d（%.4f）"
          % (zero_shot["correct"], zero_shot["total"], zero_shot["accuracy"] or 0.0,
             zero_shot["compatible_correct"], zero_shot["total"],
             zero_shot["compatible_accuracy"] or 0.0))
    for key, count in sorted(zero_shot["confusion"].items(), key=lambda kv: -kv[1]):
        print("    %-40s %d" % (key, count))

    print("\n开放词汇检索指标")
    print("%-22s %6s %6s %6s %6s %6s %6s" %
          ("query", "AP", "P@1", "P@3", "P@5", "hits", "total"))
    for entry in query_metrics:
        print("%-22s %6.3f %6.3f %6.3f %6.3f %6d %6d"
              % (entry["query"], entry["ap"],
                 entry["precision_at"]["p@1"], entry["precision_at"]["p@3"],
                 entry["precision_at"]["p@5"], entry["hits"], entry["relevant_total"]))

    if query_metrics:
        print("\n平均 AP = %.4f   平均 P@1 = %.4f"
              % (report["summary"]["mean_ap"], report["summary"]["mean_p1"]))
        print("\nz 阈值扫描（跨查询宏平均）")
        print("%8s %10s %10s %10s" % ("z", "accepted", "precision", "recall"))
        for row in report["acceptance_sweep_summary"]:
            print("%8.1f %10.1f %10s %10.3f"
                  % (row["z_threshold"], row["mean_accepted"],
                     ("%.3f" % row["mean_precision"]) if row["mean_precision"] is not None else "-",
                     row["mean_recall"]))

        print("\n概念 margin 扫描（vocabulary 模式，跨查询宏平均）")
        print("%8s %10s %10s %10s" % ("margin", "accepted", "precision", "recall"))
        for row in report["vocabulary_margin_sweep_summary"]:
            print("%8.3f %10.1f %10s %10.3f"
                  % (row["margin"], row["mean_accepted"],
                     ("%.3f" % row["mean_precision"]) if row["mean_precision"] is not None else "-",
                     row["mean_recall"]))
    print("\n输出：%s" % args.output)


if __name__ == "__main__":
    main()
