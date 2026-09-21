#!/bin/bash
set -x
cd /data/efficient3d_robot
GT=outputs/gt_dense/office_0
CLASSES=bin,basket,tissue-paper,chair,sofa,stool,armchair,couch,door,table,desk,desk-organizer,tv-screen,tablet,monitor

run () {
  NAME=$1; ROOT=$2
  /miniconda3/bin/python3 -m tools.evaluate_against_gt \
    --tracking-json $ROOT/association/tracking.json \
    --segmentation-root $ROOT/segmentation \
    --gt-root $GT \
    --output $ROOT/gt_eval_dense.json \
    --gt-classes $CLASSES \
    --max-area-fraction 0.4 > $ROOT/gt_eval_dense.log 2>&1
  echo "=== $NAME done rc=$? ==="
}

run baseline outputs/dense_office0/office_0
run n5 outputs/rt_n5
run n10 outputs/rt_n10
echo ALLDONE
