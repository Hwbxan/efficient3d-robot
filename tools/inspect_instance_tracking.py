import json
from argparse import ArgumentParser
from pathlib import Path

from src.mapping.geometric_instance_tracker import (
    GeometricInstanceTracker,
    set_known_labels,
)
from tools.check_instance_pairs import load_observations


# PATCH_KNOWN_LABELS_V1
def collect_observed_labels(instances_root, frames):
    """兜底：没有显式 --classes 时，直接从实例 JSON 里收集出现过的标签。

    检测器输出的 label 就是提示词原文，所以当上层忘记透传 --classes 时，
    用它自己产生的标签当白名单是最贴近事实的选择——总好过退回硬编码的
    8 个默认类别，把新类别全部丢成 unknown。
    """

    labels = set()

    for frame_index in frames:
        metadata_path = (
            instances_root / f"frame_{frame_index:06d}" / "instances_3d.json"
        )
        if not metadata_path.is_file():
            continue
        with metadata_path.open("r", encoding="utf-8") as file:
            for item in json.load(file):
                text = str(item.get("label", "")).strip().lower()
                if text:
                    labels.add(text)

    return sorted(labels)


def main():
    parser = ArgumentParser()

    # PATCH_LABEL_GATE_V1
    parser.add_argument("--label-gate", default="off",
                        choices=["off", "strict", "support"],
                        help="标签否决门控档位：off=不否决，"
                             "strict=已知标签不同即否决，"
                             "support=只在家具与非家具之间否决")
    parser.add_argument(
        "--instances-root",
        type=Path,
        default=Path("outputs/instances_3d"),
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0, 10],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/association/two_frame_tracking.json"),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="当前提示词列表；缺省时自动从实例 JSON 收集。",
    )

    arguments = parser.parse_args()

    # PATCH_KNOWN_LABELS_ARGS_V1
    # 标签白名单必须在建轨之前注入，否则不在默认白名单里的类别拿不到投票，
    # 融合结果会整体退化成 unknown。
    known_labels = arguments.classes
    if not known_labels:
        known_labels = collect_observed_labels(
            arguments.instances_root,
            arguments.frames,
        )
    if known_labels:
        set_known_labels(known_labels)
        print(
            f"标签白名单（{len(known_labels)} 类）："
            f"{', '.join(known_labels)}",
            flush=True,
        )

    if any(
        current <= previous
        for previous, current in zip(
            arguments.frames,
            arguments.frames[1:],
        )
    ):
        raise ValueError("--frames 必须严格递增")

    tracker = GeometricInstanceTracker(
        label_gate=(arguments.label_gate != "off"),
        label_gate_mode=arguments.label_gate)  # SUPPORT_GATE_V1
    frame_results = []

    for frame_index in arguments.frames:
        metadata_path = (
            arguments.instances_root
            / f"frame_{frame_index:06d}"
            / "instances_3d.json"
        )

        observations = load_observations(metadata_path)

        # 保证首次分配 ID 的顺序可复现。
        observations.sort(
            key=lambda observation: observation["local_instance_id"]
        )

        results = tracker.update(
            observations,
            frame_index,
        )

        frame_results.append(
            {
                "frame_index": frame_index,
                "associations": results,
            }
        )

        print(f"\nFrame {frame_index:06d}")

        for result in results:
            global_id = result["global_id"]
            cost = result["association_cost"]

            global_text = (
                f"G{global_id:03d}"
                if global_id is not None
                else "pending"
            )
            cost_text = (
                f"{cost:.3f}"
                if cost is not None
                else "-"
            )

            print(
                f"  Local {result['local_instance_id']:02d} "
                f"{result['raw_label']:18s} "
                f"→ {global_text:8s} | "
                f"{result['decision']:20s} | "
                f"Cost={cost_text}"
            )

    tracks = tracker.export_tracks()

    print("\n最终实例记录")

    for track in tracks:
        print(
            f"  G{track['global_id']:03d} | "
            f"{track['label']:18s} | "
            f"{track['status']:10s} | "
            f"观测次数={track['observation_count']} | "
            f"最后出现帧={track['last_seen_frame']}"
        )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with arguments.output.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "frames": frame_results,
                "tracks": tracks,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n关联记录：{arguments.output.resolve()}")


if __name__ == "__main__":
    main()