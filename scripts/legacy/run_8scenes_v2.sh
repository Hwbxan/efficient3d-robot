#!/bin/bash
# Demo A 最终跑 v2：8 场景 × 2000 连续帧
#
# 重要：--classes 必须用 bash 数组传，多词提示词（"tv screen" / "trash can"）
# 不能被空格拆开。之前写成字符串 $CLASSES 时被拆成 tv/screen/trash/can
# 共 8 个提示词，与 A/B 实验里用的 6 个完全不同 —— 这是宏平均 AP 从 0.506
# 掉到 0.438 的真正原因。
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
SCENES="${1:-office0 office1 office2 office3 office4 room0 room1 room2}"
FRAMES=$(seq 0 1999 | tr '\n' ' ')
CLASSES=("tv screen" "chair" "desk" "trash can" "door" "sofa")
OUTROOT=outputs/rt8_v2

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
    --skip-ply \
    --no-preview \
    > $OUTROOT/$s.log 2>&1
  echo "############ $(date +%H:%M:%S) END $s rc=$? ############"
  grep -E "wall=" $OUTROOT/$s.log | tail -2
done
echo ALLDONE8
