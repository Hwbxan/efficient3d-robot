#!/bin/bash
# v4 跑完后自动试跑 v5（39 类全词汇）的两个代表性场景：
#   office0 = v4 里最差的（AP50 6.8），room1 = 物体最少最有挑战的（v3 只有 7 个实例）
# 目的：验证"扩词汇"到底是涨精度还是引入误检，再决定要不要跑满 8 个场景。
cd /data/efficient3d_robot
while ! grep -q ALLDONE8 outputs/rt8_v4.out 2>/dev/null; do sleep 30; done
echo "=== v4 完成 $(date +%H:%M:%S) ==="
setsid nohup bash run_8scenes_v5.sh "office0 room1" \
  > outputs/rt8_v5_probe.out 2>&1 < /dev/null &
sleep 5
tail -3 outputs/rt8_v5_probe.out
