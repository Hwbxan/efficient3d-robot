#!/bin/bash
# v9 = v8 的分组/别名/NMS 不变，只把检测间隔 20 调回 10。
# 目的：判定 v8 在 room_0 上 AP25 34.9->28.2 的下跌，是不是"检测帧砍半、视角多样性不足"造成的。
# 代价：detection 算力翻倍（约 25 FPS），这里只关心精度上界。
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
FRAMES=$(seq 0 1999 | tr '\n' ' ')
OUTROOT=outputs/rt8_v9
SAM=checkpoints/sam2.1-hiera-base-plus
mkdir -p $OUTROOT
for s in office2 room0; do
  echo "############ $(date +%H:%M:%S) START $s ############"
  /usr/bin/time -f "$s wall=%es" $PY -u -m tools.run_sequence_efficient \
    --scene-directory datasets/processed/Replica/$s \
    --run-directory $OUTROOT/$s --frames $FRAMES \
    --dino-model checkpoints/grounding-dino-base --sam-model $SAM \
    --class-groups-file config/class_groups_v8.json \
    --label-alias-file config/label_alias_v7.json \
    --groups-per-frame all --group-nms-iou 0.60 \
    --box-threshold 0.30 \
    --detect-interval 10 --propagate-stride 2 --propagate-min-area 150 \
    --pixel-stride 2 --merge-coverage 0.5 --cross-label-merge \
    --no-preview \
    > $OUTROOT/$s.log 2>&1
  echo "############ $(date +%H:%M:%S) END $s rc=$? ############"
done
echo "ALLDONE_V9_PROBE"
