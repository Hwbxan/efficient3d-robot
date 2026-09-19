"""横向对比多个实验目录的 GT 指标与物体预算。

回答的是"改了 X 之后到底变好没有"，所以刻意做成**只读、可复现**：
不做任何聚合口径的二次加工，每个数字都直接来自各实验自己的
`gt_eval_*.json` / `object_budget.json`，缺哪个就跳过哪个并明确标注。

⚠️ 只看 AP 会误判。本项目里 `evaluate_against_gt.py` 的 AP 是**类别无关**的
（只用 IoU 匹配，完全不看标签），所以"把沙发标成椅子"这种错误**不会**反映在 AP 上，
只会反映在物体预算的按类别轨道数上。因此报告里 AP 与预算表并列，
**结论必须两个一起看**。

用法（在项目根目录）：

    python -m tools.compare_experiments \\
        --experiments outputs/experiments/office0_filtered_v4 \\
                      outputs/experiments/office0_filtered_v5_sofa \\
        --output outputs/experiments/comparison_v4_vs_v5.json
"""

import argparse
import json
from pathlib import Path

WINDOWS = ["all", "warmup", "heldout"]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", nargs="+", required=True,
                        help="实验目录列表，按对比顺序给出（第一个作为基准）")
    parser.add_argument("--output", default=None)
    parser.add_argument("--labels", nargs="+", default=None,
                        help="每个实验的显示名（默认用目录名）")
    parser.add_argument("--baseline", default=None,
                        help="作为基准的实验名；给出后额外打印相对差值")
    return parser.parse_args()


def load_json(path):
    if not Path(path).is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def summarise_eval(report):
    """把一份 gt_eval_*.json 压成一行可比指标。"""

    if report is None:
        return None
    detection = report.get("detection", {})
    association = report.get("association", {})
    half = detection.get("iou_0.5", {})
    quarter = detection.get("iou_0.25", {})
    label = report.get("label_consistency", {})
    # 注意：association.fragmented_gt_objects 是**字典**（类别 → {gt_id, track_ids}），
    # 不是计数。直接塞进表格会把整张表撑烂，所以这里只取条目数。
    fragmented = association.get("fragmented_gt_objects")
    if isinstance(fragmented, dict):
        fragmented_count = len(fragmented)
    elif isinstance(fragmented, list):
        fragmented_count = len(fragmented)
    else:
        fragmented_count = fragmented
    return {
        "frames": report.get("evaluated_frames"),
        "ap25": quarter.get("ap"),
        "precision25": quarter.get("precision"),
        "recall25": quarter.get("recall"),
        "tp25": quarter.get("tp"),
        "predictions": quarter.get("predictions"),
        "gt_instances": quarter.get("gt_instances"),
        "ap50": half.get("ap"),
        "recall50": half.get("recall"),
        "unassigned_observations": report.get("unassigned_observations"),
        "matched_gt_objects": association.get("matched_gt_objects"),
        "single_track_ratio": association.get("single_track_ratio"),
        "pure_track_ratio": association.get("pure_track_ratio"),
        "fragmentation_mean": association.get("fragmentation_mean"),
        "fragmented_gt_objects": fragmented_count,
        "label_consistency_ratio": label.get("ratio"),
    }


def collect(experiment, label):
    """读一个实验目录里所有可用的产物。"""

    root = Path(experiment)
    entry = {"label": label, "directory": str(root), "windows": {}, "budget": None}
    for window in WINDOWS:
        entry["windows"][window] = summarise_eval(
            load_json(root / f"gt_eval_{window}.json")
        )
    budget = load_json(root / "object_budget.json")
    if budget is not None:
        entry["budget"] = {
            "overall_ratio": budget.get("overall_ratio"),
            "gt_objects_total": budget.get("gt_objects_total"),
            "tracks_total": budget.get("tracks_total"),
            "undetectable_classes": budget.get("undetectable_classes", []),
            "new_track_summary": budget.get("new_track_summary", {}),
            "per_class": {
                name: {"gt_objects": item.get("gt_objects"),
                       "tracks": item.get("tracks"),
                       "ratio": item.get("ratio"),
                       "queryable": item.get("detector_queries_this_class")}
                for name, item in budget.get("per_class_budget", {}).items()
            },
        }
    return entry


def format_value(value, digits=4):
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (dict, list, tuple)):
        # 防御性：表格里绝不放复合结构，否则整张表会被撑烂
        return f"<{type(value).__name__} len={len(value)}>"
    return str(value)


METRIC_ROWS = [
    ("frames", "frames", 0),
    ("ap25", "AP@0.25", 4),
    ("precision25", "P@0.25", 4),
    ("recall25", "R@0.25", 4),
    ("tp25", "TP@0.25", 0),
    ("predictions", "predictions", 0),
    ("gt_instances", "GT instances", 0),
    ("ap50", "AP@0.50", 4),
    ("recall50", "R@0.50", 4),
    ("unassigned_observations", "unassigned obs", 0),
    ("matched_gt_objects", "matched GT obj", 0),
    ("single_track_ratio", "single-track ratio", 4),
    ("pure_track_ratio", "pure-track ratio", 4),
    ("fragmentation_mean", "fragmentation mean", 3),
    ("fragmented_gt_objects", "fragmented GT obj", 0),
    ("label_consistency_ratio", "label consistency", 4),
]


def print_metric_table(entries, window, baseline_label):
    print(f"\n=== GT 指标 · 窗口 {window} ===")
    header = "%-22s" % "metric" + "".join("%14s" % entry["label"][:13] for entry in entries)
    if baseline_label:
        header += "%14s" % "Δ vs 基准"
    print(header)
    print("-" * len(header))

    baseline = next((e for e in entries if e["label"] == baseline_label), None)
    for key, title, digits in METRIC_ROWS:
        values = [(e["windows"].get(window) or {}).get(key) for e in entries]
        if all(value is None for value in values):
            continue
        line = "%-22s" % title + "".join("%14s" % format_value(value, digits) for value in values)
        if baseline_label and baseline is not None:
            base = (baseline["windows"].get(window) or {}).get(key)
            current = values[-1]
            if isinstance(base, (int, float)) and isinstance(current, (int, float)):
                line += "%+14s" % format_value(current - base, digits)
            else:
                line += "%14s" % "--"
        print(line)


def print_budget_table(entries):
    if not any(entry["budget"] for entry in entries):
        print("\n（没有实验带 object_budget.json，跳过物体预算对比）")
        return

    print("\n=== 物体预算（过分割倍率 = 轨道数 / GT 物体数） ===")
    print("%-18s" % "实验" + "".join("%16s" % entry["label"][:15] for entry in entries))
    line = "%-18s" % "整体倍率"
    for entry in entries:
        budget = entry["budget"]
        line += "%16s" % (format_value(budget["overall_ratio"], 3) if budget else "--")
    print(line)

    # 并集类别，方便看出某个实验新引入了哪些类别（例如加了 sofa 词表）
    labels = []
    for entry in entries:
        if entry["budget"]:
            for name in entry["budget"]["per_class"]:
                if name not in labels:
                    labels.append(name)

    print("\n%-18s" % "按类别（轨道数）")
    print("%-18s" % "类别" + "".join("%16s" % entry["label"][:15] for entry in entries))
    for name in sorted(labels):
        line = "%-18s" % name
        for entry in entries:
            budget = entry["budget"]
            item = budget["per_class"].get(name) if budget else None
            if item is None:
                line += "%16s" % "--"
            else:
                mark = "" if item["queryable"] else "*"
                line += "%16s" % f"{item['tracks']}{mark}"
        print(line)
    print("（* = 该类别不在该实验的检测器词表里，结构性检不到）")

    print("\n%-18s" % "按类别（GT 物体数）")
    print("%-18s" % "类别" + "".join("%16s" % entry["label"][:15] for entry in entries))
    for name in sorted(labels):
        line = "%-18s" % name
        for entry in entries:
            budget = entry["budget"]
            item = budget["per_class"].get(name) if budget else None
            line += "%16s" % ("--" if item is None else item["gt_objects"])
        print(line)

    for entry in entries:
        budget = entry["budget"]
        if not budget:
            continue
        print(f"\n--- {entry['label']} 新建轨道归因 ---")
        summary = budget.get("new_track_summary", {})
        print(f"    总计 {summary.get('total')} 次"
              f"，同标签轨道在 2 个评测帧内还出现过 "
              f"{summary.get('same_label_track_within_2_eval_frames')} 次")
        for cause, count in (summary.get("attribution_counts") or {}).items():
            print(f"    {cause:<18} {count:>4}")
        if budget["undetectable_classes"]:
            print(f"    ⚠️ 词表外类别（结构性检不到）：{budget['undetectable_classes']}")


def main():
    args = parse_arguments()
    labels = args.labels or [Path(path).name for path in args.experiments]
    if len(labels) != len(args.experiments):
        raise SystemExit("--labels 数量必须与 --experiments 一致")

    entries = [collect(path, label) for path, label in zip(args.experiments, labels)]

    missing = [entry["label"] for entry in entries
               if not any(entry["windows"].values()) and not entry["budget"]]
    if missing:
        print(f"⚠️ 这些实验目录里什么都没找到（既无 gt_eval_*.json 也无 object_budget.json）：{missing}")

    for window in WINDOWS:
        if any(entry["windows"].get(window) for entry in entries):
            print_metric_table(entries, window, args.baseline)
        else:
            print(f"\n=== GT 指标 · 窗口 {window} ===\n（无数据）")

    print_budget_table(entries)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({
                "experiments": entries,
                "baseline": args.baseline,
                "note": (
                    "AP 是类别无关的（只看 IoU，不看标签），所以标签层面的收益"
                    "只会体现在物体预算的按类别轨道数上。两个表必须一起看。"
                ),
            }, handle, ensure_ascii=False, indent=2)
        print(f"\n报告：{output_path}")


if __name__ == "__main__":
    main()
