#!/bin/bash
# 为 8 场景生成 stride=8 的密集 GT（每场景 250 帧，覆盖整条 2000 帧轨迹）。
# 关键：N=10 时检测帧是 0,10,20,...；GT 帧是 0,8,16,...，
# 只有 ≡0 (mod 40) 的帧重合 —— 97.5% 的评测帧是几何传播帧，不会被"检测帧刷分"。
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
for s in office0 office1 office2 office3 office4 room0 room1 room2; do
  rs=$(echo $s | sed -E 's/([a-z]+)([0-9]+)/\1_\2/')
  echo "######## $(date +%H:%M:%S) GT $s -> $rs ########"
  $PY -u -m tools.generate_gt_instance_masks \
    --scene-directory datasets/processed/Replica/$s \
    --replica-scene $rs \
    --replica-root datasets/raw/replica_v1 \
    --output-directory outputs/gt8/$rs \
    --start-frame 0 --end-frame 1999 --frame-stride 8 --preview-count 0 \
    > outputs/gt8_$s.log 2>&1
  echo "######## $(date +%H:%M:%S) GT $s done rc=$? ########"
done
echo GTDONE8
