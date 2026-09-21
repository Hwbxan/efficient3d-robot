#!/bin/bash
# 同义词修好后重跑：v5 / v6 × agnostic|labeled(syn)|labeled(strict) × all|seen
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
OUT=/data/efficient3d_robot/cmp_v5_v6b.txt
: > $OUT
for RUN in rt8_v5 rt8_sam_sam2.1-hiera-base-plus; do
  $PY eval_mesh_protocol.py --run-root outputs/$RUN --dist 0.05 --mode agnostic --subset all 2>&1 | sed -n '4,17p' >> $OUT
  $PY eval_mesh_protocol.py --run-root outputs/$RUN --dist 0.05 --mode labeled  --subset all 2>&1 | sed -n '4,17p' >> $OUT
  $PY eval_mesh_protocol.py --run-root outputs/$RUN --dist 0.05 --mode labeled  --subset all --strict 2>&1 | sed -n '4,17p' >> $OUT
  $PY eval_mesh_protocol.py --run-root outputs/$RUN --dist 0.05 --mode labeled  --subset seen 2>&1 | sed -n '4,17p' >> $OUT
done
echo "ALLDONE_CMPB" >> $OUT
