"""只统计固定评测区间，并选择待人工核验的表面匹配案例。"""

from collections import Counter, defaultdict


def associations(tracking, start, end):
    return [
        item for frame in tracking["frames"] if start <= frame["frame_index"] <= end
        for item in frame["associations"]
    ]


def decision_counts(items):
    return dict(Counter(item["decision"] for item in items))


def make_heldout_report(baseline, updated, comparison, start=310, end=600):
    frames = list(range(start, end + 1, 10))
    for tracking in [baseline, updated]:
        actual = [frame["frame_index"] for frame in tracking["frames"] if start <= frame["frame_index"] <= end]
        if actual != frames:
            raise ValueError("评测帧列表不完整或顺序不正确")
    old_items = associations(baseline, start, end)
    new_items = associations(updated, start, end)
    old_keys = [(item["frame_index"], item["local_instance_id"]) for item in old_items]
    new_keys = [(item["frame_index"], item["local_instance_id"]) for item in new_items]
    if old_keys != new_keys or len(set(old_keys)) != len(old_keys):
        raise ValueError("评测的观测集合、顺序不同，或存在重复记录")

    mapping = defaultdict(Counter)
    for before, after in zip(old_items, new_items):
        old_id, new_id = before["global_id"], after["global_id"]
        mapping[str(old_id)][str(new_id)] += 1
    rejected_reasons = Counter(
        candidate["reason"] for item in new_items
        for candidate in item.get("surface_candidates", []) if not candidate["surface_passed"]
    )
    warmup_old = associations(baseline, 0, start - 1)
    warmup_new = associations(updated, 0, start - 1)
    report = {
        "protocol": "同一场景的后续时间段验证；从第 0 帧因果回放，不是跨场景测试",
        "evaluation_frames": frames,
        "evaluation_frame_count": len(frames),
        "evaluation_observation_count": len(new_items),
        "baseline_decisions": decision_counts(old_items),
        "v2_decisions": decision_counts(new_items),
        "baseline_new_ids_in_evaluation": sum(item["decision"] == "new_tentative" for item in old_items),
        "v2_new_ids_in_evaluation": sum(item["decision"] == "new_tentative" for item in new_items),
        "baseline_ids_observed_in_evaluation": len({item["global_id"] for item in old_items if item["global_id"] is not None}),
        "v2_ids_observed_in_evaluation": len({item["global_id"] for item in new_items if item["global_id"] is not None}),
        "warmup_baseline_created_ids": sum(item["decision"] == "new_tentative" for item in warmup_old),
        "warmup_v2_created_ids": sum(item["decision"] == "new_tentative" for item in warmup_new),
        "v2_association_sources": dict(Counter(item["association_source"] for item in new_items)),
        "surface_rejection_counts_per_candidate_not_per_observation": dict(rejected_reasons),
        "surface_attempt_observation_count": sum(bool(item.get("surface_candidates")) for item in new_items),
        "surface_parameters": comparison["surface_parameters"],
        "changed_assignment_events": [
            event for event in comparison["changed_assignment_events"] if start <= event["frame_index"] <= end
        ],
        "baseline_to_v2_observation_mapping": dict((key, dict(value)) for key, value in mapping.items()),
        "limitations": [
            "统计不使用前 0～300 帧的成功案例作为评测结果；历史状态允许保留。",
            "已匹配、新建 ID 及覆盖率都不是真值精度，不自动判定重复建档或误合并。",
            "在看到本区间结果后调参，该区间就不能再作为未参与调参的验证集。",
        ],
    }
    return report


def make_manual_review(baseline, updated, start=310, end=600, samples_per_group=6):
    old_lookup = {
        (item["frame_index"], item["local_instance_id"]): item
        for item in associations(baseline, start, end)
    }
    # 只建立引用索引，选择候选时仍使用各帧日志中当时的历史状态。
    observed_ids = {
        (item["frame_index"], item["global_id"]): item["local_instance_id"]
        for frame in updated["frames"] for item in frame["associations"]
        if item["global_id"] is not None
    }
    successful, rejected = [], []
    for item in associations(updated, start, end):
        candidates = item.get("surface_candidates", [])
        if not candidates:
            continue
        is_surface_match = item["association_source"] == "surface"
        if is_surface_match:
            candidate = next(candidate for candidate in candidates if candidate["global_id"] == item["global_id"])
        else:
            measured = [candidate for candidate in candidates if "coverage" in candidate]
            same_label = [candidate for candidate in candidates if candidate["reason"] != "requires_exact_known_label"]
            if measured:
                candidate = max(measured, key=lambda candidate: candidate["coverage"])
            elif same_label:
                candidate = min(same_label, key=lambda candidate: candidate["center_distance_m"])
            else:
                candidate = min(candidates, key=lambda candidate: candidate["center_distance_m"])

        before = old_lookup[(item["frame_index"], item["local_instance_id"])]
        reference_frame = candidate["track_last_seen_frame"]
        if reference_frame >= item["frame_index"]:
            raise ValueError("人工核验引用到了非过去帧")
        case = {
            "group": "surface_matched" if is_surface_match else "not_surface_matched",
            "current_frame": item["frame_index"],
            "current_local_id": item["local_instance_id"],
            "raw_label": item["raw_label"],
            "baseline_global_id": before["global_id"],
            "baseline_decision": before["decision"],
            "v2_global_id": item["global_id"],
            "v2_decision": item["decision"],
            "association_source": item["association_source"],
            "candidate_global_id": candidate["global_id"],
            "reference_frame": reference_frame,
            "reference_local_id": observed_ids[(reference_frame, candidate["global_id"])],
            "candidate_evidence": candidate,
            "human_judgement": None,
            "human_notes": "",
        }
        (successful if is_surface_match else rejected).append(case)

    # 优先检查成功中的边界案例，以及拒绝中覆盖较高的可疑漏匹配。
    successful.sort(key=lambda case: (case["candidate_evidence"]["coverage"], case["current_frame"]))
    rejected.sort(key=lambda case: (-case["candidate_evidence"].get("coverage", -1), case["current_frame"]))
    selected = successful[:samples_per_group] + rejected[:samples_per_group]
    for index, case in enumerate(selected, start=1):
        case["case_id"] = f"C{index:03d}"
    display_frames = sorted({frame for case in selected for frame in [case["current_frame"], case["reference_frame"]]})
    if not display_frames:
        display_frames = list(range(start, end + 1, 50))
        if end not in display_frames:
            display_frames.append(end)
    return {
        "instructions": [
            "核对当前局部掩码与参考帧中候选实例，必要时查看更早观测或单独的 3D 点云。",
            "参考图只展示最近一次观测；真实匹配参考表面累计了更早历史，不限于这一张图。",
            "human_judgement 可填 correct_match、wrong_merge、missed_match、correct_rejection 或 uncertain。",
            "成功和拒绝分层选择了边界案例，不是随机抽样，不能直接代表整体准确率。",
            "没有人工核验前保留 null；不能用算法覆盖率替代人工判断。",
        ],
        "available_success_cases": len(successful),
        "available_rejected_cases": len(rejected),
        "selected_case_count": len(selected),
        "display_frames": display_frames,
        "cases": selected,
    }
