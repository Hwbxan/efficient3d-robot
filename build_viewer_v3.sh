#!/bin/bash
# 从 v3（含 3D 地图落盘）重建 8 场景语义地图查看器数据
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
SCENES="office0 office1 office2 office3 office4 room0 room1 room2"
for s in $SCENES; do
  echo "#### $(date +%H:%M:%S) EMB $s ####"
  $PY -u -m tools.extract_instance_embeddings \
    --run-directory outputs/rt8_v3/$s \
    --scene-directory datasets/processed/Replica/$s \
    --output-directory outputs/rt8_v3/$s/embeddings \
    > outputs/rt8_v3/$s/emb.log 2>&1
  echo "#### $(date +%H:%M:%S) EMB $s rc=$? ####"
  tail -2 outputs/rt8_v3/$s/emb.log
done
echo "#### $(date +%H:%M:%S) BUILD VIEWER DATA ####"
$PY -u -m tools.build_viewer_data \
  --eval-root outputs/rt8_v3 \
  --scenes office0,office1,office2,office3,office4,room0,room1,room2 \
  --all \
  --output outputs/viewer_data_v3
echo VIEWDONE
