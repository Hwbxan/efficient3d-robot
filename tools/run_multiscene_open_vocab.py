"""在已跑完的多场景评测产物上，跑开放词汇文本检索评测并汇总。

前置：tools.run_multiscene_eval 已产出每场景的 association/tracking.json、
segmentation/、fusion_attempt_01/instance_map.json，以及 outputs/gt/<scene>。
本脚本只做「检索」这一段：CLIP 实例嵌入 → 文本查询 → 对 GT 算检索 AP。

每个场景三步：
  1. extract_instance_embeddings  掩码裁剪 → CLIP 图像嵌入（面积加权）
  2. query_instances              CLIP 文本嵌入 → 余弦排序（提示词集成 + z-score）
  3. evaluate_open_vocab_query    用 GT 语义类别反推每个实例的真实类别，算 AP/P@k

用法（在项目根目录）：
    python3 -m tools.run_multiscene_open_vocab \
        --scenes office_0 office_1 ... office_2 room_0 room_1 room_2 \
        --eval-root outputs/multiscene_eval_v2 \
        --real-root datasets/processed/Replica \
        --gt-root outputs/gt \
        --output outputs/multiscene_eval_v2/open_vocab_summary.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_QUERIES = [
    "chair", "desk", "door", "computer monitor", "trash can", "sofa",
]


def replica_sequence_name(scene: str) -> str:
    """官方场景名 → Nice-SLAM RGB-D 序列目录名（office_1 → office1）。"""
    return scene.replace("_", "")


def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [sys.executable, "-u", "-m", cmd[0], *cmd[1:]],
            stdout=handle, stderr=subprocess.STDOUT,
        )
    return result.returncode == 0


def evaluate_scene(scene, eval_root, real_root, gt_root, queries, device):
    scene_dir = eval_root / scene
    sequence_dir = real_root / replica_sequence_name(scene)
    gt_scene = gt_root / scene
    embeddings = scene_dir / "embeddings"

    if not (scene_dir / "association" / "tracking.json").exists():
        print(f"  ✗ {scene}: 缺 tracking.json，跳过")
        return None
    if not gt_scene.exists():
        print(f"  ✗ {scene}: 缺 GT（{gt_scene}），跳过")
        return None

    log_dir = eval_root / "open_vocab_logs"
    # 1. 嵌入
    ok = run([
        "tools.extract_instance_embeddings",
        "--run-directory", str(scene_dir),
        "--scene-directory", str(sequence_dir),
        "--output-directory", str(embeddings),
        "--device", device or "cuda",
    ], log_dir / f"{scene}_embed.log")
    if not ok:
        print(f"  ✗ {scene}: 嵌入提取失败，看 {log_dir}/{scene}_embed.log")
        return None

    # 2. 查询
    query_results = embeddings / "query_results.json"
    ok = run([
        "tools.query_instances",
        "--embedding-directory", str(embeddings),
        "--queries", *queries,
        "--top-k", "10",
        "--output", str(query_results),
    ], log_dir / f"{scene}_query.log")
    if not ok:
        print(f"  ✗ {scene}: 查询失败")
        return None

    # 3. 评测
    evaluation = embeddings / "open_vocab_evaluation.json"
    ok = run([
        "tools.evaluate_open_vocab_query",
        "--query-results", str(query_results),
        "--tracking-json", str(scene_dir / "association" / "tracking.json"),
        "--segmentation-root", str(scene_dir / "segmentation"),
        "--gt-root", str(gt_scene),
        "--output", str(evaluation),
    ], log_dir / f"{scene}_eval.log")
    if not ok:
        print(f"  ✗ {scene}: 评测失败")
        return None

    report = json.loads(evaluation.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    print(f"  ✓ {scene}: 平均 AP={summary.get('mean_ap')}  "
          f"P@1={summary.get('mean_p1')}  查询数={summary.get('query_count')}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--eval-root", type=Path,
                        default=Path("outputs/multiscene_eval_v2"))
    parser.add_argument("--real-root", type=Path,
                        default=Path("datasets/processed/Replica"))
    parser.add_argument("--gt-root", type=Path, default=Path("outputs/gt"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/multiscene_eval_v2/"
                                     "open_vocab_summary.json"))
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    per_scene = {}
    for scene in args.scenes:
        print(f"场景 {scene}")
        report = evaluate_scene(scene, args.eval_root, args.real_root,
                                args.gt_root, args.queries, args.device)
        if report is not None:
            per_scene[scene] = report

    # 汇总：跨场景宏平均（只统计两场景都出现的查询，避免场景间查询集不齐）
    query_ap = {}
    for scene, report in per_scene.items():
        for entry in report.get("queries", []):
            query_ap.setdefault(entry["query"], {})[scene] = entry["ap"]

    macro = {}
    for query, scene_ap in query_ap.items():
        if len(scene_ap) < 1:
            continue
        macro[query] = {
            "scenes": len(scene_ap),
            "ap": round(sum(scene_ap.values()) / len(scene_ap), 4),
        }

    all_ap = [entry["ap"] for q in query_ap.values() for entry in
              [{"ap": v} for v in q.values()]]
    summary = {
        "scenes": list(per_scene.keys()),
        "queries": args.queries,
        "per_query_macro_ap": macro,
        "mean_ap_across_all": round(
            sum(all_ap) / len(all_ap), 4) if all_ap else None,
        "scene_summaries": {
            scene: per_scene[scene].get("summary", {}) for scene in per_scene
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n已完成 {len(per_scene)} 个场景；宏平均 AP：")
    for query, info in sorted(macro.items(), key=lambda kv: -kv[1]["ap"]):
        print(f"  {query:<18} AP={info['ap']:.3f}  ({info['scenes']} 场景)")
    print(f"  整体平均 AP = {summary['mean_ap_across_all']}")
    print(f"输出：{args.output}")


if __name__ == "__main__":
    main()
