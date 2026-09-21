"""把远端 demo_a_stats_v10.json 转成 build_demo_a_site.py 认识的旧字段名。

旧站点模板读 s["ap25"] / s["ap50"]（0~1），v10 的汇总脚本给的是
ap_labeled_all=[ap25, ap50, ap75] 等四口径三阈值。这里做一次投影，
同时把新口径挂上去，方便改模板。
"""
import json
import shutil
from pathlib import Path

SRC = Path("/workspace/demo_a_stats_v10.json")
DST = Path("/workspace/demo_a_stats.json")

d = json.loads(SRC.read_text())
for s in d["scenes"]:
    lab = s.get("ap_labeled_all", [0, 0, 0])
    agn = s.get("ap_agnostic_all", [0, 0, 0])
    st = s.get("ap_labeled_all_strict", [0, 0, 0])
    s["ap25"] = lab[0]
    s["ap50"] = lab[1]
    s["ap75"] = lab[2]
    s["ap25_agnostic"] = agn[0]
    s["ap50_agnostic"] = agn[1]
    s["ap25_strict"] = st[0]
# 站点模板读 stats["overall"]，v10 汇总脚本给的是 "summary"，两个名字都留一份
sm = d.get("summary", {})
lab = sm.get("ap_labeled_all", [0, 0, 0])
agn = sm.get("ap_agnostic_all", [0, 0, 0])
st = sm.get("ap_labeled_all_strict", [0, 0, 0])
sm["kpis"] = [
    [f"{sm.get('fps_mean', 0):.1f}", "平均吞吐 FPS（8 场景，单进程 RTX 3090）",
     {"good": True}],
    [f"{sm.get('fps_min', 0):.1f}", "最慢场景 FPS", {"good": True}],
    [f"{lab[0] * 100:.1f}", "AP@.25 · labeled / all（论文口径）", {"good": True}],
    [f"{lab[1] * 100:.1f}", "AP@.50 · labeled / all", {}],
    [f"{lab[2] * 100:.1f}", "AP@.75 · labeled / all", {}],
    [f"{agn[0] * 100:.1f}", "AP@.25 · class-agnostic / all", {}],
    [f"{st[0] * 100:.1f}", "AP@.25 · 严格同名口径（保守下界）", {}],
]
d["summary"] = sm
d["overall"] = sm
DST.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("写入", DST)
print(json.dumps(d["summary"], ensure_ascii=False, indent=2))
