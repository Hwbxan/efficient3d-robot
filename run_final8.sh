#!/bin/bash
# 最终 8 场景（Demo A 的正式成绩）：v8 的分组/别名/跨组 NMS + 选定的检测间隔
# + 选定的标签门控档位。参数由命令行传入，方便在不改脚本的前提下对照。
#
# 用法：run_final8.sh <输出目录名> <detect-interval> <label-gate 档位>
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
OUT=outputs/$1
INTERVAL=$2
GATE=$3
FRAMES=$(seq 0 1999 | tr '\n' ' ')
SAM=checkpoints/sam2.1-hiera-base-plus
mkdir -p $OUT
GATEARG=""
if [ "$GATE" != "off" ]; then GATEARG="--label-gate $GATE"; fi
echo "### OUT=$OUT interval=$INTERVAL gate=$GATE"
for s in office0 office1 office2 office3 office4 room0 room1 room2; do
  echo "############ $(date +%H:%M:%S) START $s ############"
  /usr/bin/time -f "$s wall=%es" $PY -u -m tools.run_sequence_efficient \
    --scene-directory datasets/processed/Replica/$s \
    --run-directory $OUT/$s --frames $FRAMES \
    --dino-model checkpoints/grounding-dino-base --sam-model $SAM \
    --class-groups-file class_groups_v8.json \
    --label-alias-file label_alias_v7.json \
    --groups-per-frame all --group-nms-iou 0.60 \
    --box-threshold 0.30 \
    --detect-interval $INTERVAL --propagate-stride 2 --propagate-min-area 150 \
    --pixel-stride 2 --merge-coverage 0.5 --cross-label-merge \
    --no-preview $GATEARG \
    > $OUT/$s.log 2>&1
  echo "############ $(date +%H:%M:%S) END $s rc=$? ############"
done
echo "ALLDONE_FINAL8_$1"
