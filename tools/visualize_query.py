"""把文本查询结果渲染成 3D 可视化（PLY + PNG）。

对每个查询，把场景中全部实例点云画成浅灰背景，命中的实例按
相似度上色（turbo 色图，越红越匹配），另存一份只含命中实例的 PLY，
便于在 Open3D / MeshLab 中查看。

用法（在项目根目录）：
python -m tools.visualize_query \
    --query-results outputs/experiments/office0_fixed_detector_v3/embeddings/query_results.json \
    --individual-directory outputs/experiments/office0_fixed_detector_v3/fusion_attempt_01/individual \
    --output-directory outputs/experiments/office0_fixed_detector_v3/embeddings/query_visuals
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query-results", required=True)
    parser.add_argument("--individual-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--max-points-per-instance", type=int, default=6000)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def read_points(path):
    archive = np.load(path)
    return archive["points_world"].astype(np.float64)


def write_ply(path, points, colors):
    """写出二进制 little-endian PLY（XYZ + RGB），与项目既有产物一致。"""

    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by tools/visualize_query.py\n"
        "element vertex %d\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n" % len(points)
    ).encode("ascii")

    records = np.empty(len(points), dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    records["x"], records["y"], records["z"] = points[:, 0], points[:, 1], points[:, 2]
    records["red"], records["green"], records["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(records.tobytes())


def main():
    args = parse_arguments()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    individual = Path(args.individual_directory)

    report = json.load(open(args.query_results, encoding="utf-8"))

    # 预读所有实例点云，避免每个查询重复 IO。
    clouds = {}
    for path in sorted(individual.glob("G*.npz")):
        global_id = int(path.stem[1:])
        clouds[global_id] = read_points(path)
    print("载入 %d 个实例点云" % len(clouds))

    all_points = np.concatenate([clouds[g] for g in sorted(clouds)], axis=0)
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    extent = np.maximum(upper - lower, 0.01)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    summary = []
    for entry in report["queries"]:
        query = entry["query"]
        slug = "".join(ch if ch.isalnum() else "_" for ch in query).strip("_") or "query"

        scores = {row["global_id"]: row["score"] for row in entry["ranking"]}
        accepted = set(entry["accepted"])

        # 高精度默认阈值（margin>=0.010）下有些查询一个实例都不接受。
        # 这时退化为只高亮排序第一的实例，并在标题里标明"未达阈值"，
        # 既不掩盖精度问题，也保证图始终有可看的内容。
        below_margin = not accepted
        if below_margin and entry["ranking"]:
            top = entry["ranking"][0]
            accepted = {top["global_id"]}
            scores.setdefault(top["global_id"], top["score"])

        # 被接受但不在 top-k 排行里的实例也要有分数，回退到 0。
        for global_id in accepted:
            scores.setdefault(global_id, 0.0)

        if accepted:
            low = min(scores[g] for g in accepted)
            high = max(scores[g] for g in accepted)
        else:
            low, high = 0.0, 1.0
        span = max(high - low, 1e-6)

        figure = plt.figure(figsize=(18, 8))
        highlighted_points, highlighted_colors = [], []

        # 标题用英文：服务器 matplotlib 无 CJK 字体，中文会渲染成方框。
        for view, (elevation, azimuth, title) in enumerate([
            (18, -60, "View A (azimuth -60 deg)"),
            (70, -60, "View B (top-down, elev 70 deg)"),
        ]):
            axes = figure.add_subplot(1, 2, view + 1, projection="3d")

            # 背景：全部实例统一浅灰，提供空间参照。
            for global_id in sorted(clouds):
                points = clouds[global_id]
                rng = np.random.default_rng(global_id)
                take = min(len(points), args.max_points_per_instance)
                sample = points[rng.choice(len(points), take, replace=False)]
                axes.scatter(sample[:, 0], sample[:, 1], sample[:, 2],
                             color="#c8c8c8", s=0.6, alpha=0.30, depthshade=False)

            # 前景：命中的实例按分数上色。
            for global_id in sorted(accepted):
                points = clouds[global_id]
                rng = np.random.default_rng(global_id)
                take = min(len(points), args.max_points_per_instance * 3)
                sample = points[rng.choice(len(points), take, replace=False)]
                ratio = (scores.get(global_id, 0.0) - low) / span
                color = cm.turbo(0.15 + 0.85 * ratio)
                axes.scatter(sample[:, 0], sample[:, 1], sample[:, 2],
                             color=color, s=1.4, alpha=0.95, depthshade=False)
                if view == 0:
                    highlighted_points.append(points)
                    highlighted_colors.append(
                        np.tile((np.array(color[:3]) * 255).astype(np.uint8), (len(points), 1))
                    )

            axes.set_xlim(lower[0] - 0.01, upper[0] + 0.01)
            axes.set_ylim(lower[1] - 0.01, upper[1] + 0.01)
            axes.set_zlim(lower[2] - 0.01, upper[2] + 0.01)
            axes.set_box_aspect(extent)
            axes.view_init(elev=elevation, azim=azimuth)
            axes.set_xlabel("World X (m)")
            axes.set_ylabel("World Y (m)")
            axes.set_zlabel("World Z (m)")
            axes.set_title(title, fontsize=10)

        labels = ", ".join("G%03d(%.3f)" % (g, scores[g]) for g in sorted(accepted)) or "none"
        suffix = "  [top-1 only: below acceptance margin]" if below_margin else ""
        figure.suptitle(
            "Query \"%s\" -> concept \"%s\" -- %d instance(s)%s\n%s"
            % (query, entry.get("resolved_concept", "?"), len(accepted), suffix, labels),
            fontsize=12,
        )
        png_path = output_directory / ("query_%s.png" % slug)
        figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(figure)

        ply_path = output_directory / ("query_%s.ply" % slug)
        if highlighted_points:
            write_ply(ply_path,
                      np.concatenate(highlighted_points, axis=0),
                      np.concatenate(highlighted_colors, axis=0))

        summary.append({
            "query": query,
            "accepted": sorted(accepted),
            "png": str(png_path),
            "ply": str(ply_path) if highlighted_points else None,
        })
        print("查询 \"%s\"：命中 %d 个 -> %s" % (query, len(accepted), png_path.name))

    with open(output_directory / "visualization_index.json", "w", encoding="utf-8") as handle:
        json.dump({"queries": summary}, handle, ensure_ascii=False, indent=2)
    print("输出目录：%s" % output_directory)


if __name__ == "__main__":
    main()
