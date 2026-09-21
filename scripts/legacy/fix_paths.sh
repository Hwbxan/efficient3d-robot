#!/bin/bash
# 目录整理后同步修正引用路径（否则脚本跑不起来）
set -eu
cd /data/efficient3d_robot

echo "=== 1. 配置路径：class_groups_v8.json / label_alias_v7.json -> config/ ==="
for f in scripts/run_final8.sh \
         scripts/legacy/run_v7_probe.sh scripts/legacy/run_v8_probe.sh \
         scripts/legacy/run_v9_probe.sh scripts/legacy/smoke_v7.sh; do
  [ -f "$f" ] && sed -i \
    -e 's#--class-groups-file class_groups_v8\.json#--class-groups-file config/class_groups_v8.json#' \
    -e 's#--label-alias-file label_alias_v7\.json#--label-alias-file config/label_alias_v7.json#' \
    "$f" && echo "  $f"
done

echo "=== 2. 打分器路径：apply_quality_score.py -> scripts/quality/ ==="
[ -f scripts/finalize_demo_v11.sh ] && sed -i \
  's#\$PY -u apply_quality_score\.py#$PY -u scripts/quality/apply_quality_score.py#' \
  scripts/finalize_demo_v11.sh && echo "  scripts/finalize_demo_v11.sh"
[ -f scripts/eval/finish_q.sh ] && sed -i \
  's#\$PY apply_quality_score\.py#$PY scripts/quality/apply_quality_score.py#' \
  scripts/eval/finish_q.sh && echo "  scripts/eval/finish_q.sh"

echo "=== 3. apply_quality_score import dump_rank_feat（同目录，应已可用）==="
grep -n "from dump_rank_feat" scripts/quality/apply_quality_score.py || true

echo "=== 4. 校验：不应再有失效引用 ==="
BAD=$(grep -rn --include='*.sh' \
  -e '--class-groups-file class_groups_v8\.json' \
  -e '--label-alias-file label_alias_v7\.json' \
  -e 'apply_quality_score\.py' scripts/ 2>/dev/null \
  | grep -v 'scripts/quality/' | grep -v 'scripts/legacy/commit_stage' || true)
if [ -z "$BAD" ]; then echo "  ✅ 无失效引用"; else echo "  ⚠️ 仍有问题："; echo "$BAD"; fi

echo
echo "=== 5. 更新后的关键行 ==="
grep -n 'class-groups-file\|label-alias-file' scripts/run_final8.sh
grep -n 'apply_quality_score' scripts/finalize_demo_v11.sh
