#!/bin/bash
# 对照探针：只把 SAM2 骨干从 hiera-tiny 换成 hiera-base-plus，其余与 v5 完全一致。
#
# 动机：OVI-MAP Table 5 显示 SAM2 的类无关实例 AP50 只有 27.8，换成
# CropFormer 是 50.8 —— 掩码质量是 AP50/AP75 的最大单一杠杆。我们用的是
# 更小的 hiera-tiny，探针用来量化"换骨干"到底值多少 AP、要付多少帧率。
#
# 只跑 2 个场景（office0 / room1），与 v5 的同两场景直接对比。
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
SCENES="${2:-office0 room1}"
SAM="${1:-checkpoints/sam2.1-hiera-base-plus}"
TAG=$(basename "$SAM")
FRAMES=$(seq 0 1999 | tr '\n' ' ')
OUTROOT=outputs/rt8_sam_$TAG

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

while ! grep -q ALLDONE_V5 outputs/finish_v5.out 2>/dev/null; do
  sleep 60
done
echo "############ $(date +%H:%M:%S) v5 收尾完成，开始 SAM 骨干探针：$TAG ############"

mkdir -p $OUTROOT
for s in $SCENES; do
  echo "############ $(date +%H:%M:%S) START $s ############"
  /usr/bin/time -f "$s wall=%es" $PY -u -m tools.run_sequence_efficient \
    --scene-directory datasets/processed/Replica/$s \
    --run-directory $OUTROOT/$s \
    --frames $FRAMES \
    --dino-model checkpoints/grounding-dino-base \
    --sam-model $SAM \
    --classes "${CLASSES[@]}" \
    --box-threshold 0.30 \
    --detect-interval 10 \
    --propagate-stride 2 \
    --propagate-min-area 150 \
    --pixel-stride 2 \
    --merge-coverage 0.5 \
    --cross-label-merge \
    --no-preview \
    > $OUTROOT/$s.log 2>&1
  echo "############ $(date +%H:%M:%S) END $s rc=$? ############"
  grep -E "wall=" $OUTROOT/$s.log | tail -1
  grep -oE "FPS[^,}]*|[0-9]+\.[0-9]+ *fps" $OUTROOT/$s.log | tail -3
  df -h /data | tail -1
done
echo PROBE_DONE_$TAG
