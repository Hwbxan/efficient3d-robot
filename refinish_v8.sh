#!/bin/bash
# 只重跑「关联 + 融合」（纯 CPU），复用已有 instances_3d。
# 用法：refinish_v8.sh <run-root 下的场景目录名> <标签门控 0|1>
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
S=$1
GATE=$2
RUN=outputs/rt8_v8/$S
FRAMES=$(seq 0 1999 | tr '\n' ' ')
CLASSES=$(cat canonical_classes.json | tr -d '[]",' | tr '\n' ' ')

# 旧结果改名留档，融合脚本拒绝覆盖同名目录
rm -rf ${RUN}/fusion_attempt_01_nogate
if [ "$GATE" = "1" ]; then mv $RUN/fusion_attempt_01 ${RUN}/fusion_attempt_01_nogate; fi
rm -f $RUN/progress/tracking.json $RUN/progress/fusion.json $RUN/progress/preview.json

GATEARG=""
if [ "$GATE" = "1" ]; then GATEARG="--label-gate"; fi

echo "### $S label_gate=$GATE 关联中……"
$PY -m tools.inspect_instance_tracking \
  --instances-root $RUN/instances_3d --frames $FRAMES \
  --output $RUN/association/tracking.json \
  --classes $CLASSES $GATEARG 2>&1 | tail -4
echo "### 融合中……"
$PY -m tools.replay_instance_fusion \
  --tracking-json $RUN/association/tracking.json \
  --instances-root $RUN/instances_3d \
  --voxel-size 0.02 --output-directory $RUN/fusion_attempt_01 2>&1 | tail -3
echo "REFINISH_DONE_${S}_gate${GATE}"
