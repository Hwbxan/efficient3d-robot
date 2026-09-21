#!/bin/bash
# 用 v5（标签修复后）重建 Demo A 的开放词汇语义地图查看器数据。
#
# 为什么必须重建：修复前 room_1 有 22/28 实例标签是 unknown，旧 viewer 里
# 的标签全是错的。语义地图是 Demo A 的核心交付物，不能带着错标签上线。
#
# 流程：① 逐场景提 CLIP 实例嵌入 → ② 烘焙查看器 JSON → 本地打包自包含 HTML
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
SCENES="office0 office1 office2 office3 office4 room0 room1 room2"
OUTROOT=outputs/rt8_v5
OUTDATA=outputs/demo_a_data_v5

mkdir -p $OUTDATA

for s in $SCENES; do
  echo "############ $(date +%H:%M:%S) EMB $s ############"
  $PY -u -m tools.extract_instance_embeddings \
    --run-directory $OUTROOT/$s \
    --scene-directory datasets/processed/Replica/$s \
    --output-directory $OUTROOT/$s/embeddings \
    > $OUTROOT/$s.emb.log 2>&1
  echo "############ rc=$? ############"
  grep -E "完成|跳过小掩码" $OUTROOT/$s.emb.log | tail -2
  df -h /data | tail -1
done
echo EMBDONE

echo "############ $(date +%H:%M:%S) 烘焙查看器数据 ############"
$PY -u -m tools.build_viewer_data \
  --eval-root $OUTROOT \
  --all \
  --output $OUTDATA \
  > outputs/build_viewer_data_v5.log 2>&1
echo "############ 查看器数据 rc=$? ############"
tail -12 outputs/build_viewer_data_v5.log
ls -la $OUTDATA
echo VIEWERDONE
