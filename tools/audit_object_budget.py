"""审计"物体预算"：GT 里有多少个物体 vs 流水线造了多少条轨道。

**不需要 2D 掩码**，只吃 `tracking.json` + `outputs/gt/<scene>/gt_manifest.json`，
所以能在任何机器上秒级跑完，用于回答：

1. **过分割有多严重**：`轨道数 / GT 物体数` 的倍率，按类别拆开；
2. **哪些类别根本检不到**：GT 里存在、但检测器的**查询词表**里没有的类别
   （例如 sofa —— 查询表只有 computer monitor/chair/desk/trash can/door，
   于是沙发永远检不出来，而 `evaluate_against_gt.py` 默认又把 sofa 计入 GT，
   系统性压低召回）；
3. **哪些轨道是可疑误检**：标签在窗口内没有任何对应 GT 物体的轨道
   （例如"地板被检成 desk"）。

⚠️ 与 `diagnose_oversegmentation.py` 的分工：那个工具用掩码做 IoU 匹配，
能回答"**哪一个**轨道对应**哪一个**物体、在哪一帧断开"；本工具只能比**数量**，
回答"总量上差多少、差在哪个类别"。数量审计先跑（便宜、无需掩码），
再决定值不值得上掩码诊断。

⚠️ 帧号语义（踩过坑）：评测帧的 `frame_index` 就是**真实帧号**（0,10,20,…,600），
相邻评测帧相差 10。因此 `index` 之差 = **真实帧间隔**；要得到"隔了几个评测帧"
必须用帧在窗口里的**位置**之差。两者差 10 倍，混用会把 10 帧的断档报成 100 帧，
把"普通关联失败"误读成"另有隐情"。报告里两个量都显式给出：`gap_frames` / `gap_steps`。

用法（在项目根目录）：

    python -m tools.audit_object_budget \
        --tracking-json outputs/experiments/office0_fixed_detector_v3/tracking.json \
        --gt-root outputs/gt/office0 \
        --output outputs/experiments/office0_fixed_detector_v3/object_budget.json \
        --start-frame 0 --end-frame 300
"""

import argparse
import collections
import json
from pathlib import Path

# GT 类别 → 流水线标签的 **1:1** 映射。
#
# 注意这与 evaluate_against_gt.py 里的 LABEL_SYNONYMS 用途不同：那里是"同义词集合"
# （判断标签是否可接受，所以 chair 同时接受 chair 和 sofa），这里是"一个 GT 物体
# 归到唯一一个标签下"，否则沙发会被同时算进 chair 和 sofa 两个预算里。
GT_CLASS_TO_LABEL = {
    "chair": "chair",
    "sofa": "sofa",
    "table": "desk",
    "tv-screen": "computer monitor",
    "bin": "trash can",
    "door": "door",
}

# 检测器默认查询的类别（run_instance_sequence.py 的 --classes 默认值）。
DEFAULT_DETECTOR_CLASSES = ["computer monitor", "chair", "desk", "trash can", "door"]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-json", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--min-visible-frames", type=int, default=1,
                        help="GT 物体至少可见多少帧才算一个目标物体（过滤只露一像素的碎屑）。"
                             "同时作用于预算表、逐帧计数、新建轨道归因三处，保证口径一致")
    parser.add_argument("--detector-classes", nargs="+", default=DEFAULT_DETECTOR_CLASSES,
                        help="检测器的查询词表，用于识别'GT 有但根本没法检'的类别")
    return parser.parse_args()


def load_tracking(path):
    data = json.load(open(path, encoding="utf-8"))
    frames = {}
    for frame in data["frames"]:
        index = int(frame["frame_index"])
        frames[index] = [
            {
                "local_instance_id": int(item["local_instance_id"]),
                "global_id": item.get("global_id"),
                "label": item.get("raw_label"),
                "decision": item.get("decision"),
            }
            for item in frame["associations"]
        ]
    tracks = [
        {"global_id": int(t["global_id"]), "label": t.get("label"),
         "status": t.get("status"), "observations": t.get("observation_count")}
        for t in data.get("tracks", [])
    ]
    return frames, tracks


def collect_gt_budget(gt_manifest, start_frame, end_frame):
    """统计窗口内每个 GT 物体的可见帧集合。"""

    id_to_class = {int(k): v for k, v in gt_manifest["object_id_to_class"].items()}
    visible_frames = collections.defaultdict(set)
    for key, stats in gt_manifest["frames"].items():
        index = int(key)
        if start_frame is not None and index < start_frame:
            continue
        if end_frame is not None and index > end_frame:
            continue
        for object_id in stats["visible_objects"]:
            visible_frames[int(object_id)].add(index)
    return id_to_class, visible_frames


def main():
    args = parse_arguments()

    frames, tracks = load_tracking(args.tracking_json)
    gt_manifest = json.load(open(Path(args.gt_root) / "gt_manifest.json", encoding="utf-8"))
    id_to_class, visible_frames = collect_gt_budget(
        gt_manifest, args.start_frame, args.end_frame
    )

    window = sorted(frames)
    if not window:
        raise SystemExit("tracking.json 里没有帧")

    # GT 预算：按 1:1 映射归类，并过滤可见帧过少的碎屑。
    #
    # `qualified_objects` 是"什么算一个目标物体"的**唯一定义**，下面三处（预算表、
    # 逐帧计数、新建轨道归因）全部复用它。早期版本只有预算表应用了
    # min_visible_frames，逐帧计数和归因用的是"全部有标签的 GT 物体"，
    # 于是同一个报告里"GT 有几个物体"会出现三个不同答案。
    gt_objects = collections.defaultdict(dict)     # label -> {object_id: 可见帧数}
    qualified_objects = set()
    excluded_classes = collections.Counter()
    for object_id, seen in visible_frames.items():
        class_name = id_to_class.get(object_id, "?")
        label = GT_CLASS_TO_LABEL.get(class_name)
        if label is None:
            excluded_classes[class_name] += 1
            continue
        if len(seen) < args.min_visible_frames:
            continue
        qualified_objects.add(object_id)
        gt_objects[label][object_id] = len(seen)

    tracks_by_label = collections.Counter(t["label"] for t in tracks)

    # 过分割倍率
    budget = {}
    for label in sorted(set(gt_objects) | set(tracks_by_label)):
        objects = gt_objects.get(label, {})
        track_count = tracks_by_label.get(label, 0)
        budget[label] = {
            "gt_objects": len(objects),
            "tracks": track_count,
            "ratio": (round(track_count / len(objects), 3) if objects else None),
            "gt_object_visibility_frames": dict(sorted(objects.items())),
            "detector_queries_this_class": label in set(args.detector_classes),
        }

    # 可疑误检：标签在窗口内没有任何对应 GT 物体的轨道
    orphan_tracks = [
        {"global_id": t["global_id"], "label": t["label"],
         "observations": t["observations"], "status": t["status"]}
        for t in tracks
        if t["label"] not in gt_objects
    ]

    # GT 有、但检测器查询表里没有的类别 → 结构性检不到。
    # 只在**我们真正想检测的那 6 个目标类别**里找，否则 wall/floor/ceiling
    # 这些本来就不该检的结构性类别会把警告淹没。
    target_labels_present = {
        GT_CLASS_TO_LABEL[id_to_class[object_id]] for object_id in qualified_objects
    }
    undetectable = sorted(target_labels_present - set(args.detector_classes))

    # 逐帧：观测数 vs GT 可见数（用同一个 qualified_objects，delta 才有意义）
    #
    # ⚠️ delta 必须扣掉**结构性检不到的类别**（如 sofa —— GT 有、但检测器词表里没有）。
    # 否则每一帧都被这类物体固定压低若干，把"系统性过度检测"误读成"漏检"：
    # 实测 office0 窗口 0–300 里，不扣 sofa 时均值是 -0.77（看着像漏检），
    # 扣掉 2 个沙发后均值转正，真实情况是**几乎每帧都在过度检测**。
    # 所以这里同时给出 `gt_visible_objects`（全部目标物体）和
    # `gt_visible_detectable`（其中理论上检得到的），delta 用后者算。
    detectable_labels = set(args.detector_classes)
    per_frame = {}
    for index in window:
        visible = [object_id for object_id in qualified_objects if index in visible_frames[object_id]]
        detectable = [
            object_id for object_id in visible
            if GT_CLASS_TO_LABEL[id_to_class[object_id]] in detectable_labels
        ]
        observations = len(frames[index])
        per_frame[str(index)] = {
            "observations": observations,
            "gt_visible_objects": len(visible),
            "gt_visible_detectable": len(detectable),
            "delta": observations - len(detectable),
        }

    # 新建轨道的归因（无需掩码）。两路证据合起来判断：
    #   A. 同标签的上一条轨道最近一次被观测是什么时候（关联失败线索）
    #   B. **GT 里该类物体在本帧 / 上一评测帧是否可见**（物体到底在不在）
    #
    # B 是关键：如果该类 GT 物体连续两帧都在视野里，流水线却新建了一条轨道，
    # 那就既不是"新物体进场"也不是"遮挡断档"，而是**过分割**——
    # 要么 2D 漏检后重新捕获，要么 3D 关联没接住。二者都说明问题不在"看不见"。
    gt_class_frames = collections.defaultdict(set)
    for object_id in qualified_objects:
        label = GT_CLASS_TO_LABEL[id_to_class[object_id]]
        gt_class_frames[label] |= visible_frames[object_id]

    # ⚠️ 评测帧的 frame_index 就是**真实帧号**（0,10,20,…,600），不是"第几个评测帧"。
    # 所以 index 之差已经是真实帧间隔；要拿"隔了几个评测帧"必须用 window 里的**位置**之差。
    # 早期版本把 index 之差当成评测帧数再乘 10，把 10 真实帧的断档报成 100 帧，
    # 结论会从"普通关联失败"被误读成"另有隐情"。这里两个量都显式给出。
    previous_of = {}
    position_of = {}
    for position, index in enumerate(window):
        previous_of[index] = window[position - 1] if position > 0 else None
        position_of[index] = position

    last_seen_by_label = {}
    new_track_events = []
    for index in window:
        snapshot = dict(last_seen_by_label)          # 先取本帧之前的快照
        for item in frames[index]:
            if item["decision"] != "new_tentative":
                continue
            label = item["label"]
            previous = snapshot.get(label)           # (frame_index, position, global_id)
            gap_frames = (index - previous[0]) if previous else None
            gap_steps = (position_of[index] - previous[1]) if previous else None

            gt_now = index in gt_class_frames.get(label, set())
            previous_frame = previous_of.get(index)
            gt_before = previous_frame is not None and previous_frame in gt_class_frames.get(label, set())

            if gt_now and gt_before:
                attribution = "over_segmentation"
            elif gt_now:
                attribution = "appearance"
            elif gt_before:
                attribution = "gt_left_view"
            else:
                attribution = "no_gt_object"

            new_track_events.append({
                "frame": index,
                "global_id": item["global_id"],
                "label": label,
                "previous_same_label_track": previous[2] if previous else None,
                "previous_same_label_frame": previous[0] if previous else None,
                "gap_frames": gap_frames,            # 真实帧间隔
                "gap_steps": gap_steps,              # 评测帧间隔（相邻评测帧 = 1）
                "gt_class_visible_this_frame": gt_now,
                "gt_class_visible_previous_frame": gt_before,
                "attribution": attribution,
            })
        for item in frames[index]:
            last_seen_by_label[item["label"]] = (index, position_of[index], item["global_id"])

    attribution_counts = collections.Counter(event["attribution"] for event in new_track_events)
    # "关联没接住"的判据：同标签的上一条轨道就在**上一个评测帧**（或再上一个）还出现过，
    # 却仍然新建了轨道。这里必须用评测帧间隔（gap_steps），不能用真实帧间隔。
    association_failures = sum(
        1 for event in new_track_events
        if event["gap_steps"] is not None and event["gap_steps"] <= 2
    )

    total_gt = sum(item["gt_objects"] for item in budget.values())
    total_tracks = sum(item["tracks"] for item in budget.values())
    observation_counts = collections.Counter()
    per_frame_by_label = {}
    for index in window:
        counts = collections.Counter(item["label"] for item in frames[index])
        observation_counts.update(counts)
        per_frame_by_label[str(index)] = dict(counts.most_common())

    # 最"超配"的几帧：观测数 - 可检 GT 物体数 最大的帧，并列出这些帧里各类别的观测数。
    # 这是定位"到底是哪个类别在超发"的最快路径 —— 只看总量会以为是关联问题，
    # 按类别拆开后通常会发现是某一类在**首帧就超发**（实测 office0 是 chair）。
    worst_frames = sorted(
        per_frame.items(), key=lambda kv: -kv[1]["delta"]
    )[:5]

    report = {
        "tracking_json": args.tracking_json,
        "gt_root": args.gt_root,
        "frame_window": [window[0], window[-1]],
        "frames_evaluated": len(window),
        "min_visible_frames": args.min_visible_frames,
        "qualified_gt_objects": len(qualified_objects),
        "detector_classes": list(args.detector_classes),
        "gt_objects_total": total_gt,
        "tracks_total": total_tracks,
        "overall_ratio": round(total_tracks / total_gt, 3) if total_gt else None,
        "per_class_budget": budget,
        "observations_per_label": dict(observation_counts.most_common()),
        "observations_per_label_per_frame": per_frame_by_label,
        "orphan_tracks": orphan_tracks,
        "undetectable_classes": undetectable,
        "excluded_gt_classes": dict(excluded_classes.most_common()),
        "per_frame": per_frame,
        "new_track_events": new_track_events,
        "new_track_summary": {
            "total": len(new_track_events),
            "attribution_counts": dict(attribution_counts.most_common()),
            "same_label_track_within_2_eval_frames": association_failures,
        },
        "attribution_legend": {
            "over_segmentation": "该类 GT 物体本帧与上一评测帧都可见，流水线仍新建轨道 → 过分割（漏检重捕 或 关联未接住）",
            "appearance": "该类 GT 物体本帧才出现 → 新物体进场，新建合理",
            "gt_left_view": "该类 GT 物体上一帧可见、本帧不可见 → 物体已离开视野",
            "no_gt_object": "本帧该类没有任何 GT 物体 → 可疑误检",
        },
        "note": (
            "本工具只比较**数量**（GT 物体数 vs 轨道数），不做空间匹配，"
            "所以无法指出哪个轨道对应哪个物体；需要那一层信息请用 "
            "tools/diagnose_oversegmentation.py（需要 2D 掩码）。"
            "ratio 是上界：一条轨道也可能横跨两个 GT 物体（误合并）。"
            "归因是同标签启发式，用于判断方向，不是精确匹配。"
            "⚠️ 帧号语义：评测帧的 frame_index 就是真实帧号（0,10,…,600），"
            "所以 gap_frames 是真实帧间隔、gap_steps 是评测帧间隔（相邻评测帧 = 1），"
            "两者相差 10 倍，别混用。"
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"窗口 {window[0]}–{window[-1]}（{len(window)} 帧）")
    print(f"GT 物体 {total_gt} 个 vs 流水线轨道 {total_tracks} 条 "
          f"→ 过分割倍率 {report['overall_ratio']}")
    print()
    print("%-18s %8s %8s %8s  %s" % ("label", "GT物体", "轨道", "倍率", "可检?"))
    for label, item in sorted(budget.items(), key=lambda kv: -(kv[1]["ratio"] or 0)):
        ratio = "%.2fx" % item["ratio"] if item["ratio"] is not None else "--"
        queryable = "是" if item["detector_queries_this_class"] else "**否**"
        print("%-18s %8d %8d %8s  %s"
              % (label, item["gt_objects"], item["tracks"], ratio, queryable))

    if undetectable:
        print(f"\n⚠️ GT 里有、但检测器查询表里没有的类别（结构性检不到）：{undetectable}")
        print(f"   检测器查询表：{list(args.detector_classes)}")

    if orphan_tracks:
        print(f"\n可疑误检（标签在窗口内没有任何对应 GT 物体）：")
        for item in orphan_tracks:
            print("  G%-4s %-18s 观测 %s 次  状态 %s"
                  % (item["global_id"], item["label"], item["observations"], item["status"]))

    deltas = [entry["delta"] for entry in per_frame.values()]
    over = sum(1 for value in deltas if value > 0)
    under = sum(1 for value in deltas if value < 0)
    exact = sum(1 for value in deltas if value == 0)
    print(f"\n逐帧 观测数 - GT可检物体数：均值 {sum(deltas) / len(deltas):+.2f}"
          f"  范围 {min(deltas):+d} .. {max(deltas):+d}")
    print(f"   过度检测 {over} 帧 / 漏检 {under} 帧 / 恰好 {exact} 帧"
          f"（共 {len(deltas)} 帧；GT 计数已扣除结构性检不到的类别）")

    if worst_frames and worst_frames[0][1]["delta"] > 0:
        print("\n超配最严重的几帧（按类别拆开，看是谁在超发）：")
        for frame, entry in worst_frames:
            if entry["delta"] <= 0:
                continue
            breakdown = per_frame_by_label.get(frame, {})
            detail = "  ".join("%s x%d" % (label, count) for label, count in breakdown.items())
            print("    frame %-4s 观测 %d / 可检GT %d  (Δ%+d)   %s"
                  % (frame, entry["observations"], entry["gt_visible_detectable"],
                     entry["delta"], detail))

    summary = report["new_track_summary"]
    print(f"\n新建轨道 {summary['total']} 次，归因：")
    for name, count in summary["attribution_counts"].items():
        print("  %-18s %3d   %s" % (name, count, report["attribution_legend"][name]))
    print(f"  其中同标签轨道在 2 个评测帧内还出现过："
          f"{summary['same_label_track_within_2_eval_frames']}")

    def describe_gap(event):
        """把断档写成 '隔了 N 个评测帧（M 真实帧）'，两个量都来自实测，不再靠乘系数猜。"""
        if event["gap_frames"] is None:
            return "此前同标签无轨道"
        return ("间隔 %d 个评测帧（%d 真实帧）"
                % (event["gap_steps"], event["gap_frames"]))

    over_segmented = [e for e in new_track_events if e["attribution"] == "over_segmentation"]
    if over_segmented:
        print("\n过分割明细（该类 GT 物体连续可见却仍新建轨道）：")
        for event in over_segmented:
            print("    frame %-4d G%-4s %-18s  %s"
                  % (event["frame"], event["global_id"], event["label"], describe_gap(event)))

    false_positives = [e for e in new_track_events if e["attribution"] == "no_gt_object"]
    if false_positives:
        print("\n可疑误检（该类 GT 在本帧完全不存在）：")
        for event in false_positives:
            print("    frame %-4d G%-4s %s" % (event["frame"], event["global_id"], event["label"]))

    print(f"报告：{output_path}")


if __name__ == "__main__":
    main()
