#!/bin/bash
# v8：每个检测帧跑 2 遍 DINO（大件/软装 一组、小件/电子/灯具 一组），跨组 NMS 去重，
# 检测间隔 10 -> 20 抵消算力，保持 ~30 FPS，同时每个物体每次都被查询到。
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
FRAMES=$(seq 0 1999 | tr '\n' ' ')
OUTROOT=outputs/rt8_v8
SAM=checkpoints/sam2.1-hiera-base-plus
mkdir -p $OUTROOT
for s in office2 room0; do
  echo "############ $(date +%H:%M:%S) START $s ############"
  /usr/bin/time -f "$s wall=%es" $PY -u -m tools.run_sequence_efficient \
    --scene-directory datasets/processed/Replica/$s \
    --run-directory $OUTROOT/$s --frames $FRAMES \
    --dino-model checkpoints/grounding-dino-base --sam-model $SAM \
    --class-groups-file class_groups_v8.json \
    --label-alias-file label_alias_v7.json \
    --groups-per-frame all --group-nms-iou 0.60 \
    --box-threshold 0.30 \
    --detect-interval 20 --propagate-stride 2 --propagate-min-area 150 \
    --pixel-stride 2 --merge-coverage 0.5 --cross-label-merge --no-preview \
    > $OUTROOT/$s.log 2>&1
  echo "############ $(date +%H:%M:%S) END $s rc=$? ############"
done
echo "ALLDONE_V8_PROBE"
