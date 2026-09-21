#!/bin/bash
# v5 / v6 四口径对照：agnostic|labeled × all|seen
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
OUT=/data/efficient3d_robot/cmp_v5_v6.txt
: > $OUT
for RUN in rt8_v5 rt8_sam_sam2.1-hiera-base-plus; do
  for MODE in agnostic labeled; do
    for SUB in all seen; do
      echo "### $RUN $MODE $SUB" >> $OUT
      $PY eval_mesh_protocol.py --run-root outputs/$RUN --dist 0.05 \
          --mode $MODE --subset $SUB 2>&1 | sed -n '4,16p' >> $OUT
      echo "" >> $OUT
    done
  done
done
echo "ALLDONE_CMP" >> $OUT
