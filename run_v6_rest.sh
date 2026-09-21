#!/bin/bash
# 补齐 v6（SAM2.1-hiera-base-plus）剩余 3 个场景：office1 office3 room2。
# office0/office2/office4/room0/room1 已在同目录、同参数下跑完，会被缓存跳过。
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
FRAMES=$(seq 0 1999 | tr '\n' ' ')
OUTROOT=outputs/rt8_sam_sam2.1-hiera-base-plus
SAM=checkpoints/sam2.1-hiera-base-plus
CLASSES=(
  "tv screen" "monitor" "tablet" "chair" "stool" "sofa"
  "desk" "table" "desk organizer" "trash can" "basket"
  "tissue box" "door"
  "lamp" "cushion" "book" "pillow" "blanket" "vase"
  "bottle" "plate" "clock" "camera" "bench"
  "potted plant" "plant stand" "cabinet" "picture frame" "nightstand"
  "sculpture" "cloth" "tv stand" "candle" "pot"
  "comforter" "bed" "bowl" "shelf"
)
for s in office1 office3 room2; do
  echo "############ $(date +%H:%M:%S) START $s ############"
  /usr/bin/time -f "$s wall=%es" $PY -u -m tools.run_sequence_efficient \
    --scene-directory datasets/processed/Replica/$s \
    --run-directory $OUTROOT/$s --frames $FRAMES \
    --dino-model checkpoints/grounding-dino-base --sam-model $SAM \
    --classes "${CLASSES[@]}" --box-threshold 0.30 \
    --detect-interval 10 --propagate-stride 2 --propagate-min-area 150 \
    --pixel-stride 2 --merge-coverage 0.5 --cross-label-merge --no-preview \
    > $OUTROOT/$s.log 2>&1
  echo "############ $(date +%H:%M:%S) END $s rc=$? ############"
  grep -E "wall=" $OUTROOT/$s.log | tail -1
  df -h /data | tail -1
done
echo ALLDONE_V6
