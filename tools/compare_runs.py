"""并排比较多次运行的检测 / 关联指标。

调参时会留下一堆 run 目录，每次只看一行汇总很难判断「AP 掉了是精度掉了还是
召回掉了」。这里把若干 run 的精确率、召回率、单轨率等拉平到一张表里。

用法（项目根目录）：
    python -m tools.compare_runs \
        --runs 基线=outputs/multiscene_eval 实验=/tmp/eval_win
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def metrics_of(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    detection = data["detection"]["iou_0.25"]
    association = data["association"]
    return {
        "ap": detection["ap"],
        "precision": detection["precision"],
        "recall": detection["recall"],
        "single": association["single_track_ratio"],
        "pure": association["pure_track_ratio"],
        "frag": association["fragmentation_mean"],
        "tracks": association["matched_tracks"],
        "gt": association["matched_gt_objects"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="room_2")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="形如 名称=路径 的若干次运行根目录")
    args = parser.parse_args()

    rows = []
    for item in args.runs:
        name, _, root = item.partition("=")
        path = Path(root) / args.scene / "gt_eval.json"
        if not path.is_file():
            print(f"跳过 {name}：{path} 不存在")
            continue
        rows.append((name, metrics_of(path)))

    header = (f"{'运行':<26}{'AP@.25':>9}{'精确率':>9}{'召回率':>9}"
              f"{'单轨率':>8}{'纯轨率':>8}{'碎片':>8}{'轨道/GT':>10}")
    print(header)
    print("-" * len(header))
    for name, metric in rows:
        print(f"{name:<26}{metric['ap']:>9.4f}{metric['precision']:>9.4f}"
              f"{metric['recall']:>9.4f}{metric['single']:>8.3f}"
              f"{metric['pure']:>8.3f}{metric['frag']:>8.3f}"
              f"{str(metric['tracks']) + '/' + str(metric['gt']):>10}")


if __name__ == "__main__":
    main()
