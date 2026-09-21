#!/bin/bash
# 整理仓库目录：把平铺在根目录的 102 个文件按职责归类
#
# 硬约束：16 个脚本 import eval_mesh_protocol，且都用
# sys.path.insert(0, "/data/efficient3d_robot") —— 只要 eval_mesh_protocol.py
# 留在仓库根，其余脚本移到任何子目录都还能 import 到它。据此设计。
set -eu
cd /data/efficient3d_robot

echo "=== 0. 建目录 ==="
mkdir -p config scripts/{eval,quality,diag,analysis,ablation,collect,legacy}

echo "=== 1. 配置 ==="
git mv -f class_groups_v7.json class_groups_v8.json label_alias_v7.json \
        canonical_classes.json config/ 2>/dev/null || true

echo "=== 2. 评测 ==="
git mv -f ap_ceiling.py ap_rank.py ap_score.py diag_gap.py relabel_eval.py \
        rt_compare_dense.py vis_ceiling.py scripts/eval/ 2>/dev/null || true
git mv -f run_eval_q.sh finish_q.sh scripts/eval/ 2>/dev/null || true

echo "=== 3. 实例质量打分器 ==="
git mv -f dump_rank_feat.py apply_quality_score.py fit_rank.py fit_rank2.py \
        patch_eval_score.py patch_syn3.py scripts/quality/ 2>/dev/null || true

echo "=== 4. 诊断归因 ==="
git mv -f diag8.py diag9.py diag_assoc.py diag_assoc2.py diag_label.py \
        diag_mesh.py diagnose_dense.py scripts/diag/ 2>/dev/null || true

echo "=== 5. 分析 ==="
git mv -f clip_gap.py clip_probe.py fusion_stat.py stage_loss.py \
        inst_stat.py track_stat.py labchk.py label_count.py label_freq.py \
        lamp_check.py gt_classes.py insp_emb.py extract_track_emb.py \
        merge_tracks.py sweep_cluster.py sweep_filter.py fps_all.py \
        fps_report.py scripts/analysis/ 2>/dev/null || true

echo "=== 6. 消融实验 ==="
git mv -f ab_detector.py ab_multiscene.py ab_prompt.py ab_vocab.py \
        cmp_sam.py cmp_v5_v6.sh cmp_v5_v6b.sh probe_recall.py \
        probe_recall2.py probe_recall3.py probe_sam2bp.sh \
        scripts/ablation/ 2>/dev/null || true

echo "=== 7. 统计收集与结果快照 ==="
git mv -f collect_stats.py collect_stats_v2.py collect_stats_v10.py \
        collect_map_v3.py collect_map_v10.py scripts/collect/ 2>/dev/null || true
git mv -f demo_a_stats.json demo_a_stats_v10.json demo_a_map_v3.json \
        demo_a_map_v10.json scripts/collect/ 2>/dev/null || true

echo "=== 8. 历史脚本 ==="
git mv -f run_8scenes_final.sh run_8scenes_rt.sh run_8scenes_v2.sh \
        run_8scenes_v3.sh run_8scenes_v4.sh run_8scenes_v5.sh \
        run_v6_rest.sh run_v7_probe.sh run_v8_probe.sh run_v9_probe.sh \
        run_dense_eval.sh finish_v3.sh finish_v5.sh build_viewer_v3.sh \
        make_videos8.sh make_videos_v2.sh rebuild_map_v5.sh \
        refinish_gate.sh refinish_support.sh refinish_v7.sh refinish_v8.sh \
        relabel_v5.sh gen_gt8.sh smoke_v7.sh wait_v4_then_v5.sh \
        wait_v5_then_eval.sh finalize_demo_v10.sh \
        scripts/legacy/ 2>/dev/null || true
git mv -f results_E.md results_add_C.md results_add_D.md results_append.md \
        scripts/legacy/ 2>/dev/null || true
git mv -f commit_stage.sh commit_stage2.sh gitignore_add.txt \
        scripts/legacy/ 2>/dev/null || true

echo "=== 9. 主入口 ==="
git mv -f run_final8.sh finalize_demo_v11.sh scripts/ 2>/dev/null || true

echo "=== 10. 删除根目录的 make_egocentric_video.py 副本（tools/ 里已有）==="
if git ls-files --error-unmatch make_egocentric_video.py >/dev/null 2>&1; then
  git rm -f -q make_egocentric_video.py
fi

echo
echo "=== 剩余根目录文件 ==="
git ls-files | grep -v '/' | sed 's/^/  /'
echo
echo "根目录文件数: $(git ls-files | grep -cv '/')"
