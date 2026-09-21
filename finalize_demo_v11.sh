#!/bin/bash
# Demo A v11 定稿流程（在 v10 基础上插入第 0 步：实例质量打分器）
#
# v10 的问题是打分器不在流程里 —— 它是手动跑的后处理，重跑一遍管线
# 拿不到 45.2/21.6/7.0 这个数字。定稿必须让流程自洽。
#
# 步骤：0 打分器 -> 1 实例嵌入 -> 2 查看器数据 -> 3 第一人称高亮视频
set -u
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
RUNROOT=outputs/rt8_v10
OUTDATA=outputs/demo_a_data_v11
SCENES="office0 office1 office2 office3 office4 room0 room1 room2"

echo "############ $(date +%H:%M:%S) 0/4 实例质量打分器 ############"
# 留一场景交叉验证：预测某场景时只许用其余 7 个场景训练。
# 注意这一步需要 Replica 标注来算回归目标 IoU（离线标定，不在推理路径上）。
FUSION_DIR=fusion_attempt_01 $PY -u apply_quality_score.py \
  --run-root $RUNROOT --train-mode labeled --alpha 3.0 \
  --out outputs/quality_scores_v11.json \
  > outputs/apply_quality_score_v11.log 2>&1
echo "  rc=$? $(tail -2 outputs/apply_quality_score_v11.log | head -1)"

echo "############ $(date +%H:%M:%S) 1/4 实例嵌入 ############"
mkdir -p $OUTDATA
for s in $SCENES; do
  $PY -u -m tools.extract_instance_embeddings \
    --run-directory $RUNROOT/$s \
    --scene-directory datasets/processed/Replica/$s \
    --output-directory $RUNROOT/$s/embeddings \
    > $RUNROOT/$s.emb.log 2>&1
  echo "  $s rc=$? $(tail -1 $RUNROOT/$s.emb.log | cut -c1-90)"
done
echo EMBDONE

echo "############ $(date +%H:%M:%S) 2/4 查看器数据 ############"
$PY -u -m tools.build_viewer_data \
  --eval-root $RUNROOT --scenes office0,office1,office2,office3,office4,room0,room1,room2 \
  --output $OUTDATA \
  > outputs/build_viewer_data_v11.log 2>&1
echo "  rc=$? $(tail -3 outputs/build_viewer_data_v11.log)"
ls -la $OUTDATA
echo VIEWERDONE

echo "############ $(date +%H:%M:%S) 3/4 第一人称视频 ############"
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

echo "############ $(date +%H:%M:%S) 4/4 复测 5 口径 ############"
export FUSION_DIR=fusion_attempt_01
for ms in "labeled all" "agnostic all" "labeled seen" "agnostic seen"; do
  set -- $ms
  $PY eval_mesh_protocol.py --run-root $RUNROOT \
      --mode "$1" --subset "$2" --dump "/tmp/v11_$1_$2.json" 2>&1 \
      | grep -E "^宏平均" | sed "s/^/  $1\/$2 /"
done
$PY eval_mesh_protocol.py --run-root $RUNROOT \
    --mode labeled --subset all --strict --dump /tmp/v11_strict.json 2>&1 \
    | grep -E "^宏平均" | sed "s/^/  strict /"
echo "############ $(date +%H:%M:%S) ALLDONE_FINALIZE_V11 ############"
