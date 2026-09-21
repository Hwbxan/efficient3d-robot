#!/bin/bash
# 只重跑「关联 + 融合」，复用已有的 instances_3d（不重跑 DINO/SAM2）
# 用法：refinish_v7.sh <场景目录名> <标签门控:0|1>
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
S=$1
GATE=$2
RUN=outputs/rt8_v7/$S
# 旧融合结果先让位（数值已记录），否则融合脚本拒绝覆盖
rm -rf $RUN/fusion_attempt_01
FRAMES=$(seq 0 1999 | tr '\n' ' ')
CLASSES=$(cat canonical_classes.json | tr -d '[]",' | tr '\n' ' ')

# 清掉旧的关联/融合进度，强制重算；fusion 用新 attempt 目录避免覆盖
rm -f $RUN/progress/tracking.json $RUN/progress/fusion.json $RUN/progress/preview.json

GATEARG=""
if [ "$GATE" = "1" ]; then GATEARG="--label-gate"; fi

echo "### $S label_gate=$GATE 关联中……"
$PY -m tools.inspect_instance_tracking \
  --instances-root $RUN/instances_3d --frames $FRAMES \
  --output $RUN/association/tracking.json \
  --classes $CLASSES $GATEARG 2>&1 | tail -5
echo "### 融合中……"
$PY -m tools.replay_instance_fusion \
  --tracking-json $RUN/association/tracking.json \
  --instances-root $RUN/instances_3d \
  --voxel-size 0.02 --output-directory $RUN/fusion_attempt_01 2>&1 | tail -4
echo "REFINISH_DONE_${S}_gate${GATE}"
