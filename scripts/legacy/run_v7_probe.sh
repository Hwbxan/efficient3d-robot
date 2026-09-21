#!/bin/bash
# v7 探针：扩词汇 + 分组轮转 + 别名归一，先在两个场景上验证
# office_2（小物体多、最难）、room_0（GT 物体最多）
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
FRAMES=$(seq 0 1999 | tr '\n' ' ')
OUTROOT=outputs/rt8_v7
SAM=checkpoints/sam2.1-hiera-base-plus
mkdir -p $OUTROOT
for s in office2 room0; do
  echo "############ $(date +%H:%M:%S) START $s ############"
  /usr/bin/time -f "$s wall=%es" $PY -u -m tools.run_sequence_efficient \
    --scene-directory datasets/processed/Replica/$s \
    --run-directory $OUTROOT/$s --frames $FRAMES \
    --dino-model checkpoints/grounding-dino-base --sam-model $SAM \
    --class-groups-file class_groups_v7.json \
    --label-alias-file config/label_alias_v7.json \
    --box-threshold 0.30 \
    --detect-interval 10 --propagate-stride 2 --propagate-min-area 150 \
    --pixel-stride 2 --merge-coverage 0.5 --cross-label-merge --no-preview \
    > $OUTROOT/$s.log 2>&1
  echo "############ $(date +%H:%M:%S) END $s rc=$? ############"
done
echo "ALLDONE_V7_PROBE"
