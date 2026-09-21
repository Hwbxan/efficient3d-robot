#!/bin/bash
# 定稿：labeled IoU 训练打分器 -> 重测 5 口径 -> 导出每场景明细
cd /data/efficient3d_robot
export FUSION_DIR=fusion_attempt_01
PY=/miniconda3/bin/python3

$PY scripts/quality/apply_quality_score.py --run-root outputs/rt8_v10 \
    --train-mode labeled --out qs_labeled.json 2>&1 | tail -2

$PY eval_mesh_protocol.py --run-root outputs/rt8_v10 \
    --mode labeled --subset all --dump /tmp/q_labeled_all.json > /dev/null 2>&1
$PY eval_mesh_protocol.py --run-root outputs/rt8_v10 \
    --mode agnostic --subset all --dump /tmp/q_agnostic_all.json > /dev/null 2>&1
$PY eval_mesh_protocol.py --run-root outputs/rt8_v10 \
    --mode labeled --subset seen --dump /tmp/q_labeled_seen.json > /dev/null 2>&1
$PY eval_mesh_protocol.py --run-root outputs/rt8_v10 \
    --mode agnostic --subset seen --dump /tmp/q_agnostic_seen.json > /dev/null 2>&1
$PY eval_mesh_protocol.py --run-root outputs/rt8_v10 \
    --mode labeled --subset all --strict --dump /tmp/q_strict.json > /dev/null 2>&1

$PY - <<'EOF'
import json, numpy as np
from pathlib import Path
def load(p):
    d = json.loads(Path(p).read_text())
    return d
res = {}
for name, p in [("labeled_all", "/tmp/q_labeled_all.json"),
                ("agnostic_all", "/tmp/q_agnostic_all.json"),
                ("labeled_seen", "/tmp/q_labeled_seen.json"),
                ("agnostic_seen", "/tmp/q_agnostic_seen.json"),
                ("strict", "/tmp/q_strict.json")]:
    d = load(p)
    res[name] = {"macro": [float(np.mean([r[k] for r in d]))
                           for k in ("ap25", "ap50", "ap75")],
                 "per_scene": {r["scene"]: [r["ap25"], r["ap50"], r["ap75"]]
                               for r in d}}
Path("/data/efficient3d_robot/q_summary.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1))
for k, v in res.items():
    print(f"{k:<14} AP25 {v['macro'][0]*100:5.1f}  AP50 {v['macro'][1]*100:5.1f}"
          f"  AP75 {v['macro'][2]*100:5.1f}")
EOF
echo DONE
