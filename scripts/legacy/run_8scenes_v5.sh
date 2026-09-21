#!/bin/bash
# Demo A v5：8 场景 × 2000 帧，词汇表从 13 扩到 Replica 全 39 个非结构性类别。
#
# 动机：网格协议评测显示，8 场景共 39 类 / 240 个非结构性 GT 物体，
# v4 的 13 个提示词只覆盖 98 个（41%）。全类别 AP25 只有 40.0，而
# 已覆盖类别（seen）上已达 82.3 —— 短板是词汇，不是几何。
# 用数据集 GT 类别名作文本查询是开放词汇评测的标准设定（OpenMask3D /
# Open3DIS / OVI-MAP 都这么做），不构成"偷看标签"。
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
SCENES="${1:-office0 office1 office2 office3 office4 room0 room1 room2}"
FRAMES=$(seq 0 1999 | tr '\n' ' ')
OUTROOT=outputs/rt8_v5

# v4 的 13 个 + 补齐 Replica 其余非结构性类别（共 39 类）
CLASSES=(
  # --- v4 已有 ---
  "tv screen" "monitor" "tablet" "chair" "stool" "sofa"
  "desk" "table" "desk organizer" "trash can" "basket"
  "tissue box" "door"
  # --- 新增：Replica 高频未覆盖类别 ---
  "lamp" "cushion" "book" "pillow" "blanket" "vase"
  "bottle" "plate" "clock" "camera" "bench"
  "potted plant" "plant stand" "cabinet" "picture frame" "nightstand"
  "sculpture" "cloth" "tv stand" "candle" "pot"
  "comforter" "bed" "bowl" "shelf"
)
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
