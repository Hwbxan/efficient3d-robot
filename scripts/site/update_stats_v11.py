"""把打分器定稿后的 5 口径 AP 注入 demo_a_stats_v10.json。

输入：远端 q_summary.json（每场景 + 宏平均）
输出：/workspace/demo_a_stats_v10.json（就地更新）
"""
import json
from pathlib import Path

Q = Path("/workspace/q_summary.json")
S = Path("/workspace/demo_a_stats_v10.json")

q = json.loads(Q.read_text())
d = json.loads(S.read_text())


def norm_scene(name):
    """office_0 -> office0"""
    return name.replace("_", "")


# per_scene: {口径: {scene: [ap25, ap50, ap75]}}
per = {k: {norm_scene(s): v for s, v in q[k]["per_scene"].items()}
       for k in q}

n_upd = 0
for s in d["scenes"]:
    key = s["name"]
    for cal, field in (("labeled_all", "ap_labeled_all"),
                       ("agnostic_all", "ap_agnostic_all"),
                       ("labeled_seen", "ap_labeled_seen"),
                       ("agnostic_seen", "ap_agnostic_seen"),
                       ("strict", "ap_labeled_all_strict")):
        v = per[cal].get(key)
        if v is not None:
            s[field] = v
            n_upd += 1
    lab = s.get("ap_labeled_all", [0, 0, 0])
    s["ap25"], s["ap50"], s["ap75"] = lab[0], lab[1], lab[2]
    agn = s.get("ap_agnostic_all", [0, 0, 0])
    s["ap25_agnostic"], s["ap50_agnostic"] = agn[0], agn[1]
    st = s.get("ap_labeled_all_strict", [0, 0, 0])
    s["ap25_strict"] = st[0]

sm = d.get("summary", {})
for cal, field in (("labeled_all", "ap_labeled_all"),
                   ("agnostic_all", "ap_agnostic_all"),
                   ("labeled_seen", "ap_labeled_seen"),
                   ("agnostic_seen", "ap_agnostic_seen"),
                   ("strict", "ap_labeled_all_strict")):
    sm[field] = q[cal]["macro"]
d["summary"] = sm
d["overall"] = sm

S.write_text(json.dumps(d, ensure_ascii=False, indent=2))
print(f"更新 {n_upd} 个场景×口径字段")
print("\n宏平均：")
for k in ("labeled_all", "agnostic_all", "labeled_seen", "agnostic_seen", "strict"):
    m = q[k]["macro"]
    print(f"  {k:<14} {m[0]*100:5.1f} / {m[1]*100:5.1f} / {m[2]*100:5.1f}")
print("\n每场景 labeled/all AP25：")
for s in d["scenes"]:
    print(f"  {s['name']:<9} {s['ap_labeled_all'][0]*100:5.1f}"
          f"  (AP50 {s['ap_labeled_all'][1]*100:5.1f})")
