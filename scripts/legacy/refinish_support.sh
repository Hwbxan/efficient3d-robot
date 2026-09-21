#!/bin/bash
# 只重跑「关联 + 融合」（纯 CPU），复用已有 instances_3d，并且只改标签门控档位。
# 用法：refinish_support.sh <场景目录名> <off|strict|support>
# 结果写到 fusion_attempt_02，不动 fusion_attempt_01；评测时用
#   FUSION_DIR=fusion_attempt_02 python eval_mesh_protocol.py ...
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
S=$1
MODE=$2
RUN=outputs/rt8_v8/$S
FRAMES=$(seq 0 1999 | tr '\n' ' ')
CLASSES=$(cat canonical_classes.json | tr -d '[]",' | tr '\n' ' ')

rm -rf $RUN/fusion_attempt_02
echo "### $S label_gate=$MODE 关联中……"
$PY -m tools.inspect_instance_tracking \
  --instances-root $RUN/instances_3d --frames $FRAMES \
  --output $RUN/association/tracking_$MODE.json \
  --classes $CLASSES --label-gate $MODE 2>&1 | tail -4
echo "### 融合中……"
$PY -m tools.replay_instance_fusion \
  --tracking-json $RUN/association/tracking_$MODE.json \
  --instances-root $RUN/instances_3d \
  --voxel-size 0.02 --output-directory $RUN/fusion_attempt_02 2>&1 | tail -3
echo "REFINISH_DONE_${S}_${MODE}"
