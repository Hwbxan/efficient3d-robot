"""把两次多场景评测（基线 + 新方案）的 summary.json 拼成可离线打开的对比看板。

读取两个 run 目录下的 summary.json，输出一份自包含 HTML：
  - 顶部配置说明
  - 头条均值卡片（排除 room_1 的 2 个 GT 物体异常场景），新方案 vs 基线带 Δ
  - 逐场景明细表，AP / 跟踪质量带相对基线的涨跌着色
  - 速度列来自每帧延迟（排除 warmup 帧 0），反映真实推理吞吐

用法：
    python3 -m tools.build_multiscene_dashboard \
        --baseline outputs/multiscene_eval \
        --new outputs/multiscene_eval_v2 \
        --output outputs/multiscene_eval_v2/dashboard.html
"""

import argparse
import json
import statistics
from pathlib import Path


def load_summary(run_directory: Path) -> dict:
    return json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))


def scene_metrics(run_directory: Path, scene: str) -> dict:
    """从 run 目录抽取单场景的关键指标。"""
    summary = load_summary(run_directory)
    scene_data = summary["scenes"][scene]

    detection = scene_data["detection"]
    association = scene_data["association"]

    # 速度：每帧延迟排除 warmup 帧 0
    latency_path = run_directory / scene / "latency.json"
    fps = None
    if latency_path.is_file():
        latency = json.loads(latency_path.read_text(encoding="utf-8"))
        frames = latency["frames"][1:]  # 跳过帧 0
        if frames:
            total_ms = statistics.mean(f["total_ms"] for f in frames)
            fps = 1000.0 / total_ms if total_ms > 0 else None

    return {
        "gt_instances": scene_data.get("gt_instances", 0),
        "ap25": detection["iou_0.25"]["ap"],
        "ap50": detection["iou_0.5"]["ap"],
        "precision": detection["iou_0.25"]["precision"],
        "recall": detection["iou_0.25"]["recall"],
        "single_track": association.get("single_track_ratio", float("nan")),
        "pure_track": association.get("pure_track_ratio", float("nan")),
        "fragmentation": association.get("fragmentation_mean", float("nan")),
        "label_consistency": scene_data.get("label_consistency", {}).get("ratio"),
        "fps": fps,
    }


def fmt(value, digits=3, suffix=""):
    if value is None:
        return "—"
    if isinstance(value, float) and (value != value):  # NaN
        return "—"
    return f"{value:.{digits}f}{suffix}"


def delta_cell(new, old, digits=3, good_is_up=True, suffix=""):
    """带涨跌着色的单元格。"""
    if old is None or new is None:
        return f"<td>{fmt(new, digits, suffix)}</td>"
    diff = new - old
    if abs(diff) < 1e-9:
        cls = "flat"
    elif (diff > 0) == good_is_up:
        cls = "up"
    else:
        cls = "down"
    sign = "+" if diff > 0 else ""
    return (
        f'<td class="{cls}">{fmt(new, digits, suffix)} '
        f'<span class="d">{sign}{fmt(diff, digits)}</span></td>'
    )


def headline_card(label, new, old, digits=3, good_is_up=True, suffix=""):
    if old is None or new is None:
        delta_text = ""
        cls = "flat"
    else:
        diff = new - old
        magnitude = fmt(abs(diff), digits)
        if abs(diff) < 1e-9:
            delta_text = "±0"
            cls = "flat"
        elif (diff > 0) == good_is_up:
            # diff 与「好方向」同向：改善
            delta_text = f"▲ +{magnitude}"
            cls = "up"
        else:
            # 与「好方向」反向：退步（diff 为负时 magnitude 已取绝对值）
            delta_text = f"▼ -{magnitude}"
            cls = "down"
    return f"""
    <div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value">{fmt(new, digits, suffix)}</div>
      <div class="card-delta {cls}">{delta_text} <span class="vs">vs {fmt(old, digits, suffix)}</span></div>
    </div>"""


EXCLUDE = {"room_1"}  # 仅 2 个 GT 物体，会污染头条均值


def mean_excluding(metrics_map, key):
    values = [m[key] for s, m in metrics_map.items()
              if s not in EXCLUDE and m[key] is not None
              and not (isinstance(m[key], float) and m[key] != m[key])]
    return statistics.mean(values) if values else None


def build_html(baseline_dir: Path, new_dir: Path) -> str:
    baseline_summary = load_summary(baseline_dir)
    new_summary = load_summary(new_dir)

    scenes = [s for s in new_summary["scenes"]
              if s in baseline_summary["scenes"]]
    scenes.sort()

    base_metrics = {s: scene_metrics(baseline_dir, s) for s in scenes}
    new_metrics = {s: scene_metrics(new_dir, s) for s in scenes}

    config = new_summary.get("config", {})

    # 头条均值
    base_mean = {k: mean_excluding(base_metrics, k)
                 for k in ("ap25", "ap50", "single_track", "pure_track",
                           "fragmentation", "fps")}
    new_mean = {k: mean_excluding(new_metrics, k)
                for k in ("ap25", "ap50", "single_track", "pure_track",
                          "fragmentation", "fps")}

    cards = "\n".join([
        headline_card("AP@0.25（均值）", new_mean["ap25"], base_mean["ap25"],
                      good_is_up=True),
        headline_card("AP@0.50（均值）", new_mean["ap50"], base_mean["ap50"],
                      good_is_up=True),
        headline_card("单轨率（均值）", new_mean["single_track"],
                      base_mean["single_track"], good_is_up=True),
        headline_card("纯轨率（均值）", new_mean["pure_track"],
                      base_mean["pure_track"], good_is_up=True),
        headline_card("碎片度（均值）", new_mean["fragmentation"],
                      base_mean["fragmentation"], good_is_up=False),
        headline_card("推理速度（均值 FPS）", new_mean["fps"], base_mean["fps"],
                      good_is_up=True),
    ])

    # 逐场景表
    rows = []
    for scene in scenes:
        bm = base_metrics[scene]
        nm = new_metrics[scene]
        rows.append(f"""      <tr>
        <td class="scene">{scene}</td>
        <td>{nm['gt_instances']}</td>
        {delta_cell(nm['ap25'], bm['ap25'])}
        {delta_cell(nm['ap50'], bm['ap50'])}
        <td>{fmt(nm['precision'])}</td>
        <td>{fmt(nm['recall'])}</td>
        {delta_cell(nm['single_track'], bm['single_track'])}
        {delta_cell(nm['pure_track'], bm['pure_track'])}
        {delta_cell(nm['fragmentation'], bm['fragmentation'], good_is_up=False)}
        {delta_cell(nm['label_consistency'], bm['label_consistency'])}
        <td>{fmt(nm['fps'], 2)}</td>
      </tr>""")

    mean_row = f"""      <tr class="mean">
        <td class="scene">均值（排除 room_1）</td>
        <td>—</td>
        {delta_cell(new_mean['ap25'], base_mean['ap25'])}
        {delta_cell(new_mean['ap50'], base_mean['ap50'])}
        <td>—</td>
        <td>—</td>
        {delta_cell(new_mean['single_track'], base_mean['single_track'])}
        {delta_cell(new_mean['pure_track'], base_mean['pure_track'])}
        {delta_cell(new_mean['fragmentation'], base_mean['fragmentation'], good_is_up=False)}
        <td>—</td>
        {delta_cell(new_mean['fps'], base_mean['fps'])}
      </tr>"""

    config_html = (
        f"<li><b>检测器</b>：{config.get('dino_model')} + {config.get('sam_model')}</li>"
        f"<li><b>开放词汇提示词</b>：{', '.join(config.get('classes', []))}</li>"
        f"<li><b>GT 类别白名单</b>：{len(config.get('gt_classes', []))} 类 "
        f"({', '.join(config.get('gt_classes', [])[:6])}…)</li>"
        f"<li><b>评测帧</b>：{config.get('render_frames')} 帧 / 步长 "
        f"{config.get('frame_stride')}（每场景 "
        f"{config.get('render_frames') // config.get('frame_stride')} 帧）</li>"
        f"<li><b>最大面积占比</b>：{config.get('max_area_fraction')} · "
        f"<b>box 阈值</b>：{config.get('box_threshold')}</li>"
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>多场景评测对比看板 · Replica RGB-D</title>
<style>
  :root {{
    --bg: #0f1419; --panel: #1a2129; --panel2: #222b35;
    --text: #e6edf3; --muted: #8b98a5; --line: #30363d;
    --up: #3fb950; --down: #f85149; --flat: #8b98a5;
    --accent: #58a6ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    line-height: 1.5; padding: 32px 24px 64px;
  }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }}
  .sub {{ color: var(--muted); margin: 0 0 24px; font-size: 14px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px; margin-bottom: 28px; }}
  .card {{ background: var(--panel); border: 1px solid var(--line);
           border-radius: 12px; padding: 16px 18px; }}
  .card-label {{ color: var(--muted); font-size: 13px; }}
  .card-value {{ font-size: 28px; font-weight: 700; margin: 6px 0 2px; }}
  .card-delta {{ font-size: 13px; }}
  .card-delta .vs {{ color: var(--muted); font-size: 12px; }}
  .up {{ color: var(--up); }}
  .down {{ color: var(--down); }}
  .flat {{ color: var(--flat); }}
  .panel {{ background: var(--panel); border: 1px solid var(--line);
            border-radius: 12px; padding: 18px 20px; margin-bottom: 24px; }}
  .panel h2 {{ font-size: 17px; margin: 0 0 14px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
  th, td {{ padding: 9px 10px; text-align: center; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12.5px; }}
  td.scene {{ text-align: left; font-weight: 600; }}
  tr.mean td {{ border-top: 2px solid var(--line); font-weight: 700; background: var(--panel2); }}
  td .d {{ font-size: 11px; opacity: 0.85; }}
  ul {{ margin: 0; padding-left: 20px; color: var(--muted); font-size: 13px; }}
  li {{ margin: 3px 0; }}
  .note {{ color: var(--muted); font-size: 12.5px; margin-top: 10px; }}
</style>
</head>
<body>
  <h1>多场景评测对比看板</h1>
  <p class="sub">数据集：Replica RGB-D（Nice-SLAM 真实扫描序列）· 8 场景 · 新方案 = 体素覆盖率关联</p>

  <div class="cards">
    {cards}
  </div>

  <div class="panel">
    <h2>逐场景明细（单元格内为相对基线的 Δ）</h2>
    <table>
      <thead>
        <tr>
          <th>场景</th><th>GT 物体</th>
          <th>AP@.25</th><th>AP@.50</th>
          <th>精确率</th><th>召回率</th>
          <th>单轨率</th><th>纯轨率</th>
          <th>碎片度</th><th>标签一致</th><th>FPS</th>
        </tr>
      </thead>
      <tbody>
{chr(10).join(rows)}
{mean_row}
      </tbody>
    </table>
    <p class="note">* 单轨率 = 一个 GT 物体只被一条 track 覆盖的比例；纯轨率 = 一条 track 只含单一 GT 类别的比例；
    碎片度 = 单个 GT 物体被切成的 track 数均值（越低越好）。标签一致 = 轨道预测标签与 GT 一致比例。</p>
    <p class="note">* room_1 仅含 basket×1 + door×1 两个 GT 物体，统计不稳定，已排除在头条均值之外，但保留在表中。</p>
  </div>

  <div class="panel">
    <h2>评测配置</h2>
    <ul>{config_html}</ul>
    <p class="note">评测协议：VOC 风格 AP（按 2D 检测 IoU 匹配 GT 掩码），GT 类别白名单 + 最大面积占比过滤；
    跟踪质量通过 Hungarian 匹配 GT 物体与预测 track 计算。速度来自每帧端到端延迟（排除 warmup 帧 0）。</p>
  </div>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    html = build_html(args.baseline, args.new)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"已写入 {args.output} ({len(html)} 字节)")


if __name__ == "__main__":
    main()
