#!/bin/bash
# 目录整理后同步 README 的"目录结构"章节
set -eu
cd /data/efficient3d_robot
PY=/miniconda3/bin/python3

$PY - <<'PYEOF'
from pathlib import Path
import re

P = Path("/data/efficient3d_robot/README.md")
s = P.read_text()

NEW_STRUCT = """## 目录结构

```
.
├── src/                       核心库（几何 / 映射 / 关联 / 融合）
├── tools/                     命令行工具（推理、嵌入、查看器、视频）
├── config/                    提示词分组、标签别名、类别表
├── eval_mesh_protocol.py      Replica 网格协议评测器（多个脚本依赖，故留根目录）
├── scripts/
│   ├── run_final8.sh          主入口：跑 8 场景
│   ├── finalize_demo_v11.sh   定稿流程：打分器→嵌入→查看器→视频→复测
│   ├── eval/                  评测与归因（AP 上界、排序敏感性、逐物体归因）
│   ├── quality/               实例质量打分器（特征抽取、建模、写回）
│   ├── diag/                  早期诊断脚本
│   ├── analysis/              分析工具（CLIP、融合分布、分阶段存活率）
│   ├── ablation/              消融实验与探针
│   ├── collect/               统计收集与结果快照
│   ├── site/                  展示页生成
│   └── legacy/                历史脚本与实验记录
└── docs/
    ├── DEMO_A_V11.md          Demo A 阶段汇总（指标 / 归因 / 复现）
    └── RESULTS.md             全部实验记录
```

数据、模型权重、运行输出都不进仓库（`datasets/`、`checkpoints/`、
`outputs/` 已在 `.gitignore`），需要按 `docs/` 说明另外准备。"""

# 替换已有的"目录结构"章节
m = re.search(r"^## 目录结构\s*$", s, re.M)
if m:
    # 找到该章节到下一个 ## 之间
    start = m.start()
    nxt = re.search(r"^## ", s[m.end():], re.M)
    end = m.end() + nxt.start() if nxt else len(s)
    s = s[:start] + NEW_STRUCT + "\n\n---\n\n" + s[end:]
    P.write_text(s)
    print("README 目录结构章节已替换")
else:
    print("未找到『## 目录结构』章节，追加到文件末尾")
    s = s.rstrip() + "\n\n---\n\n" + NEW_STRUCT + "\n"
    P.write_text(s)

# 复现命令里的路径也要跟着改
s = P.read_text()
s = s.replace("bash run_final8.sh rt8_v10 20 support",
              "bash scripts/run_final8.sh rt8_v10 20 support")
s = s.replace("bash finalize_demo_v11.sh",
              "bash scripts/finalize_demo_v11.sh")
s = s.replace("    --run-root outputs/rt8_v10 --mode labeled --subset all",
              "    --run-root outputs/rt8_v10 --mode labeled --subset all")
s = s.replace("python3 diag_gap.py --run-root outputs/rt8_v10",
              "python3 scripts/eval/diag_gap.py --run-root outputs/rt8_v10")
P.write_text(s)
print("README 复现命令路径已同步")
PYEOF

echo "=== 确认 ==="
grep -n "scripts/run_final8\|scripts/finalize_demo_v11\|scripts/eval/diag_gap" README.md | head -5
