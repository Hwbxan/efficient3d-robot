#!/bin/bash
# Demo A v11 阶段成果分类提交（不 push，push 在沙箱侧做）
set -u
cd /data/efficient3d_robot

echo "=== 1/6 评测协议：同义词 bug 修复 + FUSION_DIR 支持 ==="
git add eval_mesh_protocol.py
git commit -q -m "fix(eval): 同义词表的连字符是死代码 + 支持 FUSION_DIR 覆盖

label_match 比较的是 norm() 之后的类名（'-' 已转空格），而 SYNONYM 的
值集合里写的是 indoor-plant / tv-screen 这类连字符形式，永远匹配不上。
最明显的受害者是盆栽：Replica 用 indoor-plant，我们输出 potted plant，
5 个盆栽全部被判 0 分。修复后 AP@.25 42.6 -> 44.2。

只补真同义，不做近义放宽：实测打通 cabinet<->shelf、vase<->pot 这类
近义映射后 room_1 的 AP@.50 从 5.7 掉到 3.1，错误标签抢占 GT 占用名额，
净收益为负。

另：load_prediction 的 fusion 目录改为可被 FUSION_DIR 环境变量覆盖，
便于对比不同融合产物。" && echo "  OK"

echo "=== 2/6 实例质量打分器 ==="
git add dump_rank_feat.py apply_quality_score.py patch_eval_score.py \
        fit_rank.py fit_rank2.py rank_feat_v10.json 2>/dev/null
git commit -q -m "feat(rank): 实例质量打分器，AP@.50 15.4 -> 21.6

AP 是 PR 曲线下面积，预测集合固定时完全由排序决定。原先用
log(1+观测帧数) 当置信度，与真实 IoU 的 Spearman 只有 +0.502：
沙发在 359 帧里出现过但每帧只看到一小撮体素，分数虚高；被 30 帧稳定
确认的小台灯分数反而低。真正有信号的是体素级多帧确认度 fo_mean
（每个体素平均被多少帧看到），+0.562。

用 Ridge 把 10 维无 GT 特征（log_obs/log_fo/log_vox/fill/ext/det_max/
area/fo_f5/mfvr/input_pts + 类别先验）映射到真实 IoU，留一场景交叉验证
（预测某场景时只许用其余 7 场景训练），Spearman 提到 +0.626。

  log(1+obs)  44.2/15.4/5.1  (+0.502)
  fo_mean     45.0/18.2/7.1  (+0.562)
  Ridge CV    45.2/21.6/7.0  (+0.626)
  oracle      51.7/28.6/13.5

8 个场景的 AP@.50 全部提升（office4 16.6->35.7）。
代价：strict 口径 AP@.25 从 28.1 掉到 23.3 —— 打分器按含同义词的 IoU
训练，在只认同名的 strict 下排序不再最优。已在文档中如实记录。

打分器写回 instance_map.json 的 quality_score，评测优先读它。" && echo "  OK"

echo "=== 3/6 差距归因工具 ==="
git add diag_gap.py gap_rows_v10.json 2>/dev/null
git commit -q -m "feat(diag): 逐 GT 物体归因，区分几何能力与命名能力

对 239 个 GT 物体各算两个 IoU：IoU_any（最佳预测实例，不看类别）与
IoU_lbl（类别必须匹配）。差值即纯标签损失。

  IoU>=0.25 已命中        123
  被覆盖但 IoU<0.25        71  <- 过分割 / 被大物体吞并，可改进
  完全无预测点覆盖         37  <- 受轨迹限制
  纯命名错配               16

结论：纯标签损失只有 16 个，主要矛盾不是命名而是分割质量。

lamp 是死账：36 个物体（占 15%）平均顶点覆盖度只有 0.08 ——
Nice-SLAM 轨迹是平视的，天花板灯基本不在视野内，是输入决定的不可达。

被吞并的典型（room_2 一个 shelf 吃掉架上所有东西）：plate 覆盖 0.56
IoU 0.03，box 0.87/0.03，bowl 0.98/0.04，sculpture 0.52/0.04。" && echo "  OK"

echo "=== 4/6 标签门控 support 档 ==="
git add src/mapping/geometric_instance_tracker.py \
        tools/inspect_instance_tracking.py 2>/dev/null
git commit -q -m "feat(track): 支撑面标签门控 support 档

新增第三档门控：只否决家具<->非家具的合并，家具类之间仍允许合并。

对比三档（office_2 - room_0 两场景均值）：
  off      AP@.25 40.0
  strict   AP@.25 37.1  <- desk/table、stool/chair 被拆碎
  support  AP@.25 45.6  <- 采用

strict 会把同类物体的不同实例拆散（office_2 的 AP@.50 从 25.7 掉到
21.4），而完全关闭门控又会让沙发吞掉坐垫。" && echo "  OK"

echo "=== 5/6 定稿流程与统计 ==="
git add finalize_demo_v11.sh finalize_demo_v10.sh run_final8.sh \
        run_eval_q.sh finish_q.sh collect_stats_v10.py collect_map_v10.py \
        run_v9_probe.sh run_v8_probe.sh run_v7_probe.sh \
        collect_stats.py collect_stats_v2.py collect_map_v3.py \
        build_viewer_v3.sh 2>/dev/null
git commit -q -m "feat(pipeline): Demo A v11 定稿流程，打分器接进主链路

run_final8.sh <outdir> <interval> <gate> 跑 8 场景；
finalize_demo_v11.sh 一键出全部交付物：
  0 实例质量打分器（留一场景 CV 标定，离线，不在推理路径）
  1 实例 CLIP 嵌入
  2 查看器数据
  3 第一人称高亮视频
  4 复测 5 口径

v10 的问题正是打分器不在流程里 —— 它是手动后处理，重跑管线拿不到
45.2/21.6/7.0。定稿必须让流程自洽。

最终配置：v8 分组/别名/跨组 NMS + detect-interval 20 + label-gate support
+ 质量打分器。" && echo "  OK"

echo "=== 6/6 文档 ==="
git add docs/ README.md canonical_classes.json class_groups_v7.json \
        class_groups_v8.json label_alias_v7.json 2>/dev/null
git commit -q -m "docs: Demo A v11 阶段汇总 + README 换用网格协议口径

新增 docs/DEMO_A_V11.md：定稿指标、流水线、打分器、评测协议、
239 个 GT 物体的归因、已知局限、复现命令。

README 门面此前是早期口径（200 帧步长 10、逐帧密集 GT、
0.608/0.519/6.15 FPS），与 Replica 网格协议不可比，会误导人。
换成 45.2/21.6/7.0 @ 34.0 FPS，并挂上阶段汇总链接。

同时记录 v11 全口径（8 场景宏平均）：
  labeled/all   45.2 / 21.6 / 7.0
  agnostic/all  46.9 / 22.7 / 7.0
  labeled/seen  68.1 / 33.5 / 11.3
  strict        23.3 /  7.7 / 2.1
对标：OVO-SLAM 32.8/23.6/11.1，OVI-MAP 76.7/50.8/22.0（离线）。
AP@.25 领先 OVO-SLAM 12.4，AP@.50/.75 落后 2.0/4.1。" && echo "  OK"

echo "=== 剩余未提交 ==="
git status --porcelain | head -10
echo "(剩余 $(git status --porcelain | wc -l) 个，应为实验脚本/配置等)"
echo DONE
