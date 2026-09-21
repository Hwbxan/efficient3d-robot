#!/bin/bash
# 8 场景全序列（2000 帧，stride=1）实时感知：N=10 检测抽稀 + 几何传播
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
SCENES="${1:-office0 office1 office2 office3 office4 room0 room1 room2}"
FRAMES=$(seq 0 1999 | tr '\n' ' ')
CLASSES="computer monitor chair desk trash can door sofa"
OUTROOT=outputs/rt8_n10

mkdir -p $OUTROOT
for s in $SCENES; do
  echo "############ $(date +%H:%M:%S) START $s ############"
  /usr/bin/time -f "$s wall=%es" $PY -u -m tools.run_sequence_efficient \
    --scene-directory datasets/processed/Replica/$s \
    --run-directory $OUTROOT/$s \
    --frames $FRAMES \
    --classes $CLASSES \
    --detect-interval 10 \
    --propagate-stride 2 \
    --propagate-min-area 150 \
    --pixel-stride 2 \
    --skip-ply \
    --no-preview \
    > $OUTROOT/$s.log 2>&1
  echo "############ $(date +%H:%M:%S) END $s rc=$? ############"
  grep -E "wall=|FPS|平均|mean" $OUTROOT/$s.log | tail -5
done
echo ALLDONE8
