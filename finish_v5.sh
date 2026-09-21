#!/bin/bash
# 磁盘写满后的 v5 收尾：
#   1) 重跑 5 个缺失/损坏场景（office2 office3 office4 room0 room2）
#   2) 只对 3 个"标签退化"场景（office0 office1 room1）重建关联/预览/融合
#      —— 它们的逐帧检测/分割/3D 提升有 progress 缓存，会直接跳过
# 注意：5 个新场景是用打过补丁的代码跑的，标签已经正确，无需再重跑关联。
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
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

run_scene() {
  local s=$1
  echo "############ $(date +%H:%M:%S) START $s ############"
  /usr/bin/time -f "$s wall=%es" $PY -u -m tools.run_sequence_efficient \
    --scene-directory datasets/processed/Replica/$s \
    --run-directory $OUTROOT/$s \
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
    > $OUTROOT/$s.log 2>&1
  echo "############ $(date +%H:%M:%S) END $s rc=$? ############"
  grep -E "wall=" $OUTROOT/$s.log | tail -1
  df -h /data | tail -1
  $PY labchk.py $OUTROOT/$s/fusion_attempt_01/instance_map.json 2>/dev/null || echo "$s 融合产物缺失"
}

# ---- 阶段 1：缺失/损坏场景 ----
for s in office2 office3 office4 room0 room2; do
  rm -rf $OUTROOT/$s
  run_scene $s
done
echo PHASE1DONE

# ---- 阶段 2：只重建 3 个标签退化场景的关联/预览/融合 ----
for s in office0 office1 room1; do
  echo "############ $(date +%H:%M:%S) RELABEL $s ############"
  rm -f $OUTROOT/$s/progress/tracking.json \
        $OUTROOT/$s/progress/preview.json \
        $OUTROOT/$s/progress/fusion.json
  rm -rf $OUTROOT/$s/fusion_attempt_*
  run_scene $s
done
echo ALLDONE_V5
