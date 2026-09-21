#!/bin/bash
# v7 冒烟：只跑几帧，验证分组轮转 + 别名归一在运行时可用
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3
$PY -u -m tools.run_sequence_efficient \
  --scene-directory datasets/processed/Replica/office2 \
  --run-directory outputs/v7_smoke_office2 --frames 0 1 2 3 4 5 6 7 8 9 10 11 \
  --dino-model checkpoints/grounding-dino-base \
  --sam-model checkpoints/sam2.1-hiera-base-plus \
  --class-groups-file class_groups_v7.json \
  --label-alias-file config/label_alias_v7.json \
  --box-threshold 0.30 --detect-interval 10 --propagate-stride 2 \
  --propagate-min-area 150 --pixel-stride 2 --merge-coverage 0.5 \
  --cross-label-merge --no-preview 2>&1 | head -40
echo "SMOKE_RC=$?"
