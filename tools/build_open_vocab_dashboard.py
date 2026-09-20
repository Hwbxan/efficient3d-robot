"""把多场景开放词汇检索评测渲染成可离线打开的对比看板。

读取 outputs/multiscene_eval_v2/open_vocab_summary.json 与每场景的
open_vocab_evaluation.json，输出自包含 HTML：
  - 头条：宏平均 AP（全场景 / 仅该物体存在的场景两种口径）
  - 逐查询 AP（两种口径对照，存在场景口径才是公平的"检索能力"度量）
  - 逐场景 × 逐查询 AP 热力表（单元格着色，缺失物体标灰）

用法：
    python3 -m tools.build_open_vocab_dashboard \
        --eval-root outputs/multiscene_eval_v2 \
        --output outputs/multiscene_eval_v2/open_vocab_dashboard.html
"""

import argparse
import json
from pathlib import Path


def load(eval_root, scenes):
    per_scene = {}
    for sc in scenes:
        path = eval_root / sc / "embeddings" / "open_vocab_evaluation.json"
        if path.exists():
            per_scene[sc] = {e["query"]: e for e in
                             json.load(open(path, encoding="utf-8"))["queries"]}
    summary = json.load(open(eval_root / "open_vocab_summary.json", encoding="utf-8"))
    return summary, per_scene


def color(ap):
    if ap is None:
        return "#30363d", "#8b98a5"
    if ap >= 0.75:
        return "#16331f", "#3fb950"
    if ap >= 0.5:
        return "#2a2a14", "#d29922"
    if ap >= 0.25:
        return "#33231a", "#e08a3c"
    return "#331a1a", "#f85149"


def build_html(eval_root: Path) -> str:
    summary = json.load(open(eval_root / "open_vocab_summary.json", encoding="utf-8"))
    scenes = summary["scenes"]
    queries = summary["queries"]
    per_scene = {}
    for sc in scenes:
        path = eval_root / sc / "embeddings" / "open_vocab_evaluation.json"
        if path.exists():
            per_scene[sc] = {e["query"]: e for e in
                             json.load(open(path, encoding="utf-8"))["queries"]}

    macro_all = summary.get("per_query_macro_ap_all_scenes", {})
    macro_present = summary.get("per_query_macro_ap_present_scenes", {})
    mean_all = summary.get("mean_ap_all_scenes")
    mean_present = summary.get("mean_ap_present_scenes")

    cards = f"""
    <div class="card">
      <div class="card-label">平均 AP（仅物体存在的场景）</div>
      <div class="card-value">{mean_present}</div>
      <div class="card-delta">公平口径：这些场景才该被检索到</div>
    </div>
    <div class="card">
      <div class="card-label">平均 AP（全部 8 场景）</div>
      <div class="card-value">{mean_all}</div>
      <div class="card-delta">含无该物体的场景（记 0）</div>
    </div>"""

    # 逐查询对照表
    qrows = []
    for q in queries:
        pa = macro_all.get(q)
        pp = macro_present.get(q)
        qrows.append(
            f"<tr><td class='scene'>{q}</td>"
            f"<td>{pa}</td><td class='hl'>{pp if pp is not None else '—'}</td></tr>")

    # 逐场景 × 逐查询热力表
    heat = []
    for sc in scenes:
        cells = [f"<td class='scene'>{sc}</td>"]
        for q in queries:
            e = per_scene.get(sc, {}).get(q)
            ap = e["ap"] if e else None
            bg, fg = color(ap)
            cells.append(
                f"<td style='background:{bg};color:{fg}'>"
                f"{'—' if ap is None else f'{ap:.2f}'}</td>")
        heat.append("<tr>" + "".join(cells) + "</tr>")

    queries_head = "".join(f"<th>{q}</th>" for q in queries)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>开放词汇检索评测看板 · Replica RGB-D</title>
<style>
  :root {{ --bg:#0f1419; --panel:#1a2129; --text:#e6edf3; --muted:#8b98a5;
           --line:#30363d; --accent:#58a6ff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
          font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
          line-height:1.5; padding:32px 24px 64px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); margin:0 0 24px; font-size:14px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
            gap:14px; margin-bottom:28px; }}
  .card {{ background:var(--panel); border:1px solid var(--line);
           border-radius:12px; padding:16px 18px; }}
  .card-label {{ color:var(--muted); font-size:13px; }}
  .card-value {{ font-size:28px; font-weight:700; margin:6px 0 2px; }}
  .card-delta {{ font-size:12px; color:var(--muted); }}
  .panel {{ background:var(--panel); border:1px solid var(--line);
            border-radius:12px; padding:18px 20px; margin-bottom:24px; }}
  .panel h2 {{ font-size:17px; margin:0 0 14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  th,td {{ padding:9px 10px; text-align:center; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; font-size:12.5px; }}
  td.scene {{ text-align:left; font-weight:600; }}
  td.hl {{ font-weight:700; color:var(--accent); }}
  .note {{ color:var(--muted); font-size:12.5px; margin-top:10px; }}
</style>
</head>
<body>
  <h1>开放词汇文本检索评测</h1>
  <p class="sub">数据集：Replica RGB-D（8 场景）· CLIP 图像嵌入 + 文本查询 + 余弦排序 · 对 GT 语义类别算 AP</p>

  <div class="cards">{cards}</div>

  <div class="panel">
    <h2>逐查询宏平均 AP</h2>
    <table>
      <thead><tr><th>查询词</th><th>全场景</th><th>仅物体存在的场景（公平口径）</th></tr></thead>
      <tbody>{''.join(qrows)}</tbody>
    </table>
    <p class="note">公平口径只在该物体确实出现在场景 GT 中的那些场景上求平均；全场景口径把"场景里没有该物体"也算作 AP=0，
    会低估检索能力（正确行为本就是返回空）。</p>
  </div>

  <div class="panel">
    <h2>逐场景 × 逐查询 AP 热力表</h2>
    <table>
      <thead><tr><th>场景</th>{queries_head}</tr></thead>
      <tbody>{''.join(heat)}</tbody>
    </table>
    <p class="note">绿≥0.75 · 黄≥0.5 · 橙≥0.25 · 红&lt;0.25 · 灰=该场景 GT 无此物体。
    明显弱项：computer monitor（小屏幕难检/易与桌面混淆）。</p>
  </div>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-root", type=Path,
                        default=Path("outputs/multiscene_eval_v2"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/multiscene_eval_v2/"
                                     "open_vocab_dashboard.html"))
    args = parser.parse_args()
    html = build_html(args.eval_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"已写入 {args.output} ({len(html)} 字节)")


if __name__ == "__main__":
    main()
