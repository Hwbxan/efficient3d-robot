#!/bin/bash
# 为 8 个场景生成"相机视角 + 逐步高亮 + 物体名称"的实时感知视频
# 用法：bash make_videos8.sh [场景列表] [最大帧数] [fps] [并行度]
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
SCENES="${1:-office0 office1 office2 office3 office4 room0 room1 room2}"
MAXF="${2:-600}"
FPS="${3:-30}"
JOBS="${4:-4}"
OUT=outputs/demo_video_rt8
FLAT=outputs/demo_video_rt8_flat
mkdir -p $OUT $FLAT

one () {
  s=$1
  $PY -u -m tools.make_egocentric_video \
    --scene $s \
    --rgb-root datasets/processed/Replica \
    --eval-root outputs/rt8_final \
    --max-frames $MAXF --fps $FPS \
    --style highlight --fill-alpha 0.38 --label-mode text \
    --new-frames 20 --bitrate 2.5M \
    --out $OUT/$s > $OUT/$s.log 2>&1
  if [ -f $OUT/$s/${s}_ego.mp4 ]; then
    cp $OUT/$s/${s}_ego.mp4 $FLAT/$s.mp4
    rm -rf $OUT/$s/${s}_ego_frames   # 中间 PNG 帧太大，合成后删掉
    echo "OK $s $(du -h $FLAT/$s.mp4 | cut -f1)"
  else
    echo "FAIL $s"; tail -5 $OUT/$s.log
  fi
}
export -f one
export OUT FLAT MAXF FPS PY
printf '%s\n' $SCENES | xargs -P $JOBS -I{} bash -c 'one {}'
echo VIDEODONE8
