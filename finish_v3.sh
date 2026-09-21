#!/bin/bash
# 补齐 v3 的 room0/room1/room2（前 5 个场景已完成），然后重建语义地图查看器。
#
# 事故说明：本地 /workspace/efficient3d_robot 是旧镜像，缺 --no-preview /
# --no-fusion / set_known_labels 三处只存在于远程的改动。整文件上传时把它们
# 冲掉了，room0~room2 因 "unrecognized arguments: --no-preview" 5 秒即失败。
# 已用 patch_restore.py 增量恢复。以后改远程一律用增量补丁，不整文件覆盖。
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
OUTROOT=outputs/rt8_v3
FRAMES=$(seq 0 1999 | tr '\n' ' ')
CLASSES=("tv screen" "chair" "desk" "trash can" "door" "sofa")
mkdir -p $OUTROOT

for s in room0 room1 room2; do
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
    --no-preview \
    > $OUTROOT/$s.log 2>&1
  echo "############ $(date +%H:%M:%S) END $s ############"
  grep -E "跟踪器标签集合" $OUTROOT/$s.log | head -1
  grep -E "wall=" $OUTROOT/$s.log | tail -1
done
echo ALLDONE8 > outputs/rt8_v3_marker.txt

echo "############ $(date +%H:%M:%S) 语义地图统计 ############"
$PY collect_map_v3.py > outputs/map_v3.out 2>&1
tail -12 outputs/map_v3.out
echo MAPDONE >> outputs/map_v3.out

echo "############ $(date +%H:%M:%S) 查看器数据 ############"
bash build_viewer_v3.sh > outputs/viewer_v3.out 2>&1
tail -6 outputs/viewer_v3.out
echo VIEWERDONE >> outputs/viewer_v3.out
echo FINALDONE > outputs/final_marker.txt
