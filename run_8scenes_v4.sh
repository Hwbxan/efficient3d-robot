#!/bin/bash
# Demo A 最终跑 v4：8 场景 × 2000 连续帧（含 3D 语义地图落盘 + 全局融合）
#
# 与 v2/v3 的差异：提示词从 6 个扩到覆盖全部 15 类 GT 白名单，
# 并开启单帧内合并 / 跨标签合并（防止 desk 与 table 把同一物体检出两遍）。
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
SCENES="${1:-office0 office1 office2 office3 office4 room0 room1 room2}"
FRAMES=$(seq 0 1999 | tr '\n' ' ')
OUTROOT=outputs/rt8_v4

CLASSES=("tv screen" "monitor" "tablet" "chair" "stool" "sofa"
         "desk" "table" "desk organizer" "trash can" "basket"
         "tissue box" "door")
MERGE_COVERAGE="${MERGE_COVERAGE:-0.5}"
CROSS="${CROSS:---cross-label-merge}"

mkdir -p $OUTROOT
for s in $SCENES; do
  echo "############ $(date +%H:%M:%S) START $s ############"
  /usr/bin/time -f "$s wall=%es" $PY -u -m tools.run_sequence_efficient \
    --scene-directory datasets/processed/Replica/$s \
    --run-directory $OUTROOT/$s \
    --frames $FRAMES \
    --dino-model checkpoints/grounding-dino-base \
    --classes "${CLASSES[@]}" \
    --box-threshold 0.30 \
    --detect-interval 10 \
    --propagate-stride 2 \
    --propagate-min-area 150 \
    --pixel-stride 2 \
    --merge-coverage $MERGE_COVERAGE \
    $CROSS \
    --no-preview \
    > $OUTROOT/$s.log 2>&1
  echo "############ $(date +%H:%M:%S) END $s rc=$? ############"
  grep -E "wall=" $OUTROOT/$s.log | tail -2
done
echo ALLDONE8
