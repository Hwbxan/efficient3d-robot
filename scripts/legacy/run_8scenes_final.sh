#!/bin/bash
# Demo A 最终跑：8 场景 × 2000 连续帧
# 配置（由跨场景 A/B 选定）：DINO-Base + P1 提示词 + box 0.30 + N=10 几何传播
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
SCENES="${1:-office0 office1 office2 office3 office4 room0 room1 room2}"
FRAMES=$(seq 0 1999 | tr '\n' ' ')
CLASSES="tv screen chair desk trash can door sofa"
OUTROOT=outputs/rt8_final

mkdir -p $OUTROOT
for s in $SCENES; do
  echo "############ $(date +%H:%M:%S) START $s ############"
  /usr/bin/time -f "$s wall=%es" $PY -u -m tools.run_sequence_efficient \
    --scene-directory datasets/processed/Replica/$s \
    --run-directory $OUTROOT/$s \
    --frames $FRAMES \
    --dino-model checkpoints/grounding-dino-base \
    --classes $CLASSES \
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
