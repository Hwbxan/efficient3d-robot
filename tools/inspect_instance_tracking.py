import json
from argparse import ArgumentParser
from pathlib import Path

from src.mapping.geometric_instance_tracker import (
    GeometricInstanceTracker,
)
from tools.check_instance_pairs import load_observations


def main():
    parser = ArgumentParser()

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

    arguments = parser.parse_args()

    if any(
        current <= previous
        for previous, current in zip(
            arguments.frames,
            arguments.frames[1:],
        )
    ):
        raise ValueError("--frames 必须严格递增")

    tracker = GeometricInstanceTracker()
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