#!/bin/bash
# v5 probe（office0 + room1，38 类全词汇）跑完后自动评测，与 v4 同场景对比。
cd /data/efficient3d_robot
while ! grep -q ALLDONE8 outputs/rt8_v5_probe.out 2>/dev/null; do sleep 30; done
echo "=== v5 probe 完成 $(date +%H:%M:%S) ==="
PY=/miniconda3/bin/python3
S=office_0,room_1
echo
echo "########## v5 全类别 ##########"
$PY eval_mesh_protocol.py --run-root outputs/rt8_v5 --scenes $S --mode agnostic --subset all 2>&1 | sed -n '4,14p'
echo
echo "########## v5 seen 子集 ##########"
$PY eval_mesh_protocol.py --run-root outputs/rt8_v5 --scenes $S --mode agnostic --subset seen 2>&1 | sed -n '4,14p'
echo
echo "########## v4 同场景基线（对照）##########"
$PY eval_mesh_protocol.py --run-root outputs/rt8_v4 --scenes $S --mode agnostic --subset all 2>&1 | sed -n '4,14p'
$PY eval_mesh_protocol.py --run-root outputs/rt8_v4 --scenes $S --mode agnostic --subset seen 2>&1 | sed -n '4,14p'
echo
echo "########## v5 帧率 ##########"
for s in office0 room1; do grep -E '可持续帧率' outputs/rt8_v5/$s.log | tail -1; done
echo V5EVALDONE
