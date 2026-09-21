#!/bin/bash
# 最终版 Demo A 交付物：实例 CLIP 嵌入 → 查看器数据 → 第一人称高亮视频
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
RUNROOT=outputs/rt8_v10
OUTDATA=outputs/demo_a_data_v10
SCENES="office0 office1 office2 office3 office4 room0 room1 room2"

mkdir -p $OUTDATA

echo "############ $(date +%H:%M:%S) 1/3 实例嵌入 ############"
for s in $SCENES; do
  $PY -u -m tools.extract_instance_embeddings \
    --run-directory $RUNROOT/$s \
    --scene-directory datasets/processed/Replica/$s \
    --output-directory $RUNROOT/$s/embeddings \
    > $RUNROOT/$s.emb.log 2>&1
  echo "  $s rc=$? $(tail -1 $RUNROOT/$s.emb.log | cut -c1-90)"
done
echo EMBDONE

echo "############ $(date +%H:%M:%S) 2/3 查看器数据 ############"
$PY -u -m tools.build_viewer_data \
  --eval-root $RUNROOT --all --output $OUTDATA \
  > outputs/build_viewer_data_v10.log 2>&1
echo "  rc=$? $(tail -3 outputs/build_viewer_data_v10.log)"
ls -la $OUTDATA
echo VIEWERDONE

echo "############ $(date +%H:%M:%S) 3/3 第一人称视频 ############"
OUT=outputs/demo_video_v10
FLAT=outputs/demo_video_v10_flat
mkdir -p $OUT $FLAT
one () {
  s=$1
  $PY -u -m tools.make_egocentric_video \
    --scene $s \
    --rgb-root datasets/processed/Replica \
    --eval-root $RUNROOT \
    --max-frames 600 --fps 30 \
    --style highlight --fill-alpha 0.38 --label-mode text \
    --new-frames 20 --bitrate 2.5M \
    --out $OUT/$s > $OUT/$s.log 2>&1
  if [ -f $OUT/$s/${s}_ego.mp4 ]; then
    cp $OUT/$s/${s}_ego.mp4 $FLAT/$s.mp4
    rm -rf $OUT/$s/${s}_ego_frames
    echo "  OK $s $(du -h $FLAT/$s.mp4 | cut -f1)"
  else
    echo "  FAIL $s"; tail -5 $OUT/$s.log
  fi
}
export -f one
export OUT FLAT PY RUNROOT
printf '%s\n' $SCENES | xargs -P 3 -I{} bash -c 'one {}'
echo VIDEODONE
echo "############ $(date +%H:%M:%S) ALLDONE_FINALIZE_V10 ############"
