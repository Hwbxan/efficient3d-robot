#!/bin/bash
# v5 标签修复后重跑：关联 → 预览 → 融合。
#
# 背景：tools/inspect_instance_tracking.py 之前拿不到 --classes，
# KNOWN_LABELS 停在 8 个硬编码类别，导致 v5 新增的 25 个类别全部退化成
# "unknown"（room_1 28 个实例里 22 个）。补丁已修（含自动兜底）。
# 逐帧的检测/分割/3D 提升都有 progress 缓存，重跑只会跳过它们，
# 只重建关联、预览与融合，几何结果基本不变（实测 10963 条关联只变 10 条）。
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
SCENES="${1:-office0 office1 office2 office3 office4 room0 room1 room2}"
FRAMES=$(seq 0 1999 | tr '\n' ' ')
OUTROOT=outputs/rt8_v5

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

# 等 6 个剩余场景跑完
while ! grep -q ALLDONE8 outputs/rt8_v5_rest.out 2>/dev/null; do
  sleep 60
done
echo "############ $(date +%H:%M:%S) 前置 6 场景已完成，开始重跑关联/融合 ############"

for s in $SCENES; do
  d=$OUTROOT/$s
  echo "############ $(date +%H:%M:%S) RELABEL $s ############"
  rm -f $d/progress/tracking.json $d/progress/preview.json $d/progress/fusion.json
  rm -rf $d/fusion_attempt_*
  /usr/bin/time -f "$s wall=%es" $PY -u -m tools.run_sequence_efficient \
    --scene-directory datasets/processed/Replica/$s \
    --run-directory $d \
    --frames $FRAMES \
    --dino-model checkpoints/grounding-dino-base \
    --classes "${CLASSES[@]}" \
    --box-threshold 0.30 \
    --detect-interval 10 \
    --propagate-stride 2 \
    --propagate-min-area 150 \
    --pixel-stride 2 \
    --merge-coverage 0.5 \
    --cross-label-merge \
    --no-preview \
    > $d.relabel.log 2>&1
  echo "############ $(date +%H:%M:%S) DONE $s rc=$? ############"
  $PY labchk.py $d/fusion_attempt_01/instance_map.json
done
echo RELABELDONE8
