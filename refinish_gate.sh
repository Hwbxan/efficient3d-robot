#!/bin/bash
# 只重跑「关联 + 融合」（纯 CPU），复用已有 instances_3d，只改标签门控档位。
# 结果写进 fusion_attempt_02，不动 fusion_attempt_01。
# 用法：refinish_gate.sh <run-root 名> <场景目录名> <off|strict|support> <后缀>
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
RUNROOT=$1
S=$2
MODE=$3
TAG=$4
RUN=outputs/$RUNROOT/$S
FRAMES=$(seq 0 1999 | tr '\n' ' ')
CLASSES=$(cat canonical_classes.json | tr -d '[]",' | tr '\n' ' ')

rm -rf $RUN/fusion_attempt_02
$PY -m tools.inspect_instance_tracking \
  --instances-root $RUN/instances_3d --frames $FRAMES \
  --output $RUN/association/tracking_$TAG.json \
  --classes $CLASSES --label-gate $MODE 2>&1 | tail -2
$PY -m tools.replay_instance_fusion \
  --tracking-json $RUN/association/tracking_$TAG.json \
  --instances-root $RUN/instances_3d \
  --voxel-size 0.02 --output-directory $RUN/fusion_attempt_02 2>&1 | tail -2
echo "REFINISH_DONE_${RUNROOT}_${S}_${TAG}"
