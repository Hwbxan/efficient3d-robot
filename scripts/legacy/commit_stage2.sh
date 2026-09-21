#!/bin/bash
# 第二批：核心 src/tools 改动 + 分析工具 + 结果快照
set -u
cd /data/efficient3d_robot

echo "=== 7/9 核心推理与融合改动 ==="
git add src/ tools/run_sequence_efficient.py tools/run_instance_sequence.py \
        tools/replay_instance_fusion.py tools/extract_instance_embeddings.py \
        tools/make_egocentric_video.py tools/build_viewer_data.py 2>/dev/null
git commit -q -m "feat: 分组提示词 + 跨组 NMS + 检测抽稀 + 分组阈值

- run_sequence_efficient: --class-groups-file 多组提示词逐组检测，
  组间 NMS 去重；--group-box-thresholds 支持逐组框阈值
  （Grounding DINO 对所有短语做 softmax，大物体压制小物体分数，
  分组 + 逐组阈值是缓解手段）
- --detect-interval 检测抽稀，中间帧走 mask_propagation 几何传播
- --label-gate {off,strict,support} 三档标签门控
- 检测抽稀的意外收益：interval 10->20 反而让两场景均值 AP@.25
  从 37.1 涨到 40.0，逐帧独立检测的抖动框被时间维度去噪了" && echo "  OK"

echo "=== 8/9 分析与评测工具 ==="
git add ap_ceiling.py ap_rank.py ap_score.py diag_assoc.py diag_assoc2.py \
        diag_label.py diag_mesh.py fusion_stat.py stage_loss.py \
        extract_track_emb.py insp_emb.py rt_compare_dense.py 2>/dev/null
git commit -q -m "feat(tools): AP 上界/排序敏感性/打分函数/分阶段存活率分析

ap_ceiling.py   GT 物体 i 的 IoU 上界 = |S_i|/|V_i|（可见顶点占比），
                测出天花板 95.6/89.4/69.2，用来判断差距是否可达
ap_rank.py      同一预测集换排序能涨多少：oracle AP@.50 28.6 vs 当前 15.4
                -> 瓶颈不是"检不出"而是"不会打分"
ap_score.py     打分函数对比：oracle(IoU) 45.3/23.5/10.9，
                det x obs x vdr 39.5/10.6/3.0，det_score 36.2
fusion_stat.py  融合标签/体素分布；发现 office_2 的 box 每帧出现 40 次
                却只融合出 0 个实例
stage_loss.py   小物体在各阶段的存活率" && echo "  OK"

echo "=== 9/9 结果快照 ==="
git add results_E.md results_add_C.md results_add_D.md results_append.md \
        demo_a_stats_v10.json demo_a_map_v10.json demo_a_stats.json \
        demo_a_map_v3.json 2>/dev/null
git commit -q -m "docs(results): v11 定稿结果快照与 RESULTS 增补

results_E.md 对应 RESULTS.md 1.5.23：评测口径 bug 修复 + 实例质量
打分器，含留一场景 CV 的完整对比表和 239 个 GT 物体的归因。

demo_a_stats_v10.json / demo_a_map_v10.json 是 v11 定稿的原始数据
（FPS、耗时、5 口径 AP、逐场景实例与类别分布），展示页由它们生成。" && echo "  OK"

# 把剩余的实验脚本一次性收进一个 commit，避免散落
git add -A 2>/dev/null
if [ -n "$(git diff --cached --name-only)" ]; then
  git commit -q -m "chore: 收敛本轮实验脚本与配置

一次性探索脚本（消融、探针、诊断、补丁）与中间产物快照，
保留供追溯，不影响主链路。" && echo "  剩余实验脚本 OK"
fi

echo
echo "=== 最终状态 ==="
git status --porcelain | wc -l
echo "个文件待处理"
echo
git log --oneline | head -12
echo
echo "=== 文件总数 ==="
git ls-files | wc -l
echo DONE2
