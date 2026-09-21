#!/bin/bash
# 打分器部署后重测全口径（5 组）
cd /data/efficient3d_robot
export FUSION_DIR=fusion_attempt_01
PY=/miniconda3/bin/python3
for ms in "labeled all" "agnostic all" "labeled seen" "agnostic seen"; do
  set -- $ms
  echo "=== $1 / $2 ==="
  $PY eval_mesh_protocol.py --run-root outputs/rt8_v10 \
      --mode "$1" --subset "$2" --dump "/tmp/q_$1_$2.json" 2>&1 | grep -E "宏平均|^office|^room"
done
echo "=== labeled / all / strict ==="
$PY eval_mesh_protocol.py --run-root outputs/rt8_v10 \
    --mode labeled --subset all --strict --dump /tmp/q_strict.json 2>&1 | grep -E "宏平均|^office|^room"
echo DONE
