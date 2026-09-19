"""把一次真实推理 run 的产物拼成可离线打开的交互式 HTML 看板。

读取 run_instance_sequence.py 的产物（association/tracking.json、
fusion_attempt_*/instance_map.json、tracking_preview 标注图、实例地图预览图），
生成一份自包含（图片 base64 内嵌）的 HTML，便于直接分享 / 在浏览器打开。

用法（在项目根目录）：
    /miniconda3/bin/python3 -m tools.build_demo_dashboard \
        --run-directory outputs/demo_run \
        --output outputs/demo_run/dashboard.html
"""

import argparse
import base64
import json
from pathlib import Path


def b64_image(path: Path) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def frame_gallery(run_directory: Path, tracking: dict) -> list:
    """每帧带全局 ID 的标注图，按帧顺序排列。"""

    items = []
    preview_dir = run_directory / "tracking_preview"
    for frame in tracking.get("frames", []):
        frame_index = frame["frame_index"]
        img = preview_dir / f"frame_{frame_index:06d}_global_ids.png"
        if img.is_file():
            items.append({"frame": frame_index, "src": b64_image(img)})
    return items


def instance_rows(instance_map: dict) -> list:
    rows = []
    for inst in instance_map.get("instances", []):
        bbox_min = inst.get("bbox_min_world") or [0, 0, 0]
        bbox_max = inst.get("bbox_max_world") or [0, 0, 0]
        size = [round(bbox_max[i] - bbox_min[i], 2) for i in range(3)]
        centroid = inst.get("visible_surface_median_world") or ["-", "-", "-"]
        rows.append({
            "global_id": inst.get("global_id"),
            "label": inst.get("label", "unknown"),
            "status": inst.get("tracking_status", "-"),
            "observations": inst.get("observation_count", "-"),
            "source_frames": inst.get("source_frames", []),
            "voxels": inst.get("voxel_count", "-"),
            "multi_frame_ratio": round(inst.get("multi_frame_voxel_ratio", 0.0) * 100, 1),
            "size": size,
            "centroid": [round(c, 2) for c in centroid],
        })
    return rows


def build_html(run_directory: Path, instance_map: dict, tracking: dict,
               contact_src: str, map_preview_src: str, gallery: list,
               config: dict) -> str:
    instances = instance_rows(instance_map)
    n_frames = len(tracking.get("frames", []))
    voxel_size = instance_map.get("voxel_size_m", "-")
    scene = config.get("scene_directory", "")

    gallery_html = ""
    for item in gallery:
        gallery_html += (
            f'<figure class="tile"><img loading="lazy" src="{item["src"]}" '
            f'alt="frame {item["frame"]}"/>'
            f'<figcaption>Frame {item["frame"]:06d}</figcaption></figure>'
        )

    rows_html = ""
    for r in instances:
        frames_txt = ", ".join(str(f) for f in r["source_frames"])
        rows_html += (
            "<tr>"
            f'<td class="gid">G{r["global_id"]:03d}</td>'
            f'<td><span class="tag">{r["label"]}</span></td>'
            f'<td>{r["status"]}</td>'
            f'<td>{r["observations"]}</td>'
            f'<td class="mono">{frames_txt}</td>'
            f'<td class="num">{r["voxels"]}</td>'
            f'<td class="num">{r["multi_frame_ratio"]}%</td>'
            f'<td class="mono">{r["size"]}</td>'
            f'<td class="mono">{r["centroid"]}</td>'
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Efficient3D-Robot · 在线实例地图 Demo 看板</title>
<style>
  :root {{
    --bg:#0f1115; --card:#181b22; --ink:#e8eaed; --muted:#9aa0aa;
    --accent:#4ea1ff; --line:#2a2f3a;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.6 -apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif; }}
  header {{ padding:32px 28px 20px; border-bottom:1px solid var(--line); }}
  h1 {{ margin:0 0 6px; font-size:24px; }}
  .sub {{ color:var(--muted); }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:24px 20px 60px; }}
  .cards {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0 8px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:16px 20px; min-width:150px; flex:1; }}
  .card .k {{ color:var(--muted); font-size:13px; }}
  .card .v {{ font-size:26px; font-weight:700; margin-top:4px; }}
  section {{ margin-top:34px; }}
  h2 {{ font-size:18px; border-left:4px solid var(--accent); padding-left:10px; }}
  .note {{ color:var(--muted); font-size:13px; margin:6px 0 14px; }}
  img.shot {{ width:100%; border:1px solid var(--line); border-radius:10px; display:block; }}
  .gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
    gap:12px; margin-top:14px; }}
  .tile {{ margin:0; background:var(--card); border:1px solid var(--line);
    border-radius:10px; overflow:hidden; }}
  .tile img {{ width:100%; display:block; }}
  .tile figcaption {{ padding:6px 10px; color:var(--muted); font-size:12px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:12px; font-size:14px; }}
  th,td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; }}
  td.gid {{ font-weight:700; color:var(--accent); }}
  td.num, td.mono {{ font-variant-numeric:tabular-nums; }}
  td.mono {{ color:#cdd3dc; }}
  .tag {{ background:#233044; color:#9ec5ff; border-radius:6px; padding:2px 8px; font-size:12px; }}
  code {{ background:#20242e; padding:2px 6px; border-radius:5px; color:#ffd9a0; }}
  pre {{ background:#20242e; padding:14px; border-radius:10px; overflow:auto;
    color:#d7e3ff; font-size:13px; }}
  footer {{ color:var(--muted); font-size:12px; padding:20px;
    border-top:1px solid var(--line); text-align:center; }}
</style>
</head>
<body>
<header>
  <h1>Efficient3D-Robot · 在线开放词汇 3D 实例地图</h1>
  <div class="sub">RGB-D 流 → 2D 开放词汇检测/分割 → 3D 提升 → 体素建图 → 实例关联 → 文本可查询实例地图。
  本看板由 <b>RTX 3090 真实推理</b> 产出（非回放）。</div>
</header>
<div class="wrap">

  <div class="cards">
    <div class="card"><div class="k">处理帧数</div><div class="v">{n_frames}</div></div>
    <div class="card"><div class="k">全局实例数</div><div class="v">{len(instances)}</div></div>
    <div class="card"><div class="k">体素大小 (m)</div><div class="v">{voxel_size}</div></div>
    <div class="card"><div class="k">场景</div><div class="v" style="font-size:18px">{Path(scene).name}</div></div>
  </div>

  <section>
    <h2>① 逐帧在线感知（带全局实例 ID 的标注）</h2>
    <div class="note">同一物体在不同帧被赋予相同全局 ID（G###），颜色一致；PENDING/SKIPPED 表示尚未关联。</div>
    <img class="shot" src="{contact_src}" alt="contact sheet"/>
  </section>

  <section>
    <h2>② 3D 实例地图预览</h2>
    <div class="note">融合后所有全局实例的世界坐标点云，按实例 ID 着色。</div>
    <img class="shot" src="{map_preview_src}" alt="3D instance map"/>
  </section>

  <section>
    <h2>③ 实例地图数据表</h2>
    <div class="note">多帧观测占比 = 被 ≥2 帧看到过的体素比例（越高越稳定）。size 为场景坐标系下的包围盒尺寸 (x,y,z) 米。</div>
    <table>
      <thead><tr><th>全局 ID</th><th>语义标签</th><th>状态</th><th>观测次数</th>
      <th>出现帧</th><th>体素数</th><th>多帧占比</th><th>包围盒 size (m)</th><th>质心 (x,y,z)</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </section>

  <section>
    <h2>④ 逐帧标注画廊</h2>
    <div class="note">拖动浏览每一帧的开放词汇检测结果与全局 ID 标注。</div>
    <div class="gallery">{gallery_html}</div>
  </section>

  <section>
    <h2>⑤ 文本查询（开放词汇）</h2>
    <div class="note">用 CLIP 把自然语言编码后与实例图像嵌入做余弦匹配，即可"问地图要某类物体"。</div>
    <pre>MINICONDA=/miniconda3/bin/python3
$MINICONDA -m tools.extract_instance_embeddings \\
    --run-directory {run_directory} --output outputs/demo_run/embeddings
$MINICONDA -m tools.query_instances \\
    --embedding-directory outputs/demo_run/embeddings \\
    --queries "chair" "a place to sit" "trash can" --top-k 5</pre>
    <p class="note">提示：自带 Calibrate-Before-Use 概念偏置校正，改写说法（如 "a place to sit"）也能命中 chair。</p>
  </section>

  <section>
    <h2>⑥ 复现这条 Demo 流水线</h2>
    <pre>HF_HUB_OFFLINE=1 /miniconda3/bin/python3 -m tools.run_instance_sequence \\
    --scene-directory {scene} \\
    --run-directory {run_directory} \\
    --frames {(' '.join(str(f) for f in (config.get('frames') or list(range(0,301,10)))))} \\
    --classes {(' '.join(config.get('classes', [])))}</pre>
  </section>

</div>
<footer>由 run_instance_sequence.py 真实推理产物生成 · 离线采样帧（非实时 FPS）· GPU: NVIDIA RTX 3090 24GB</footer>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_directory = Path(args.run_directory).resolve()

    fusion_done = json.loads((run_directory / "progress" / "fusion.json").read_text())
    fusion_directory = Path(fusion_done["path"])
    tracking = json.loads((run_directory / "association" / "tracking.json").read_text())
    instance_map = json.loads((fusion_directory / "instance_map.json").read_text())
    config = json.loads((run_directory / "run_config.json").read_text())

    contact_src = b64_image(run_directory / "tracking_preview" / "tracking_contact_sheet.png")
    map_preview_src = b64_image(fusion_directory / "instance_map_preview.png")
    gallery = frame_gallery(run_directory, tracking)

    html = build_html(
        run_directory=run_directory,
        instance_map=instance_map,
        tracking=tracking,
        contact_src=contact_src,
        map_preview_src=map_preview_src,
        gallery=gallery,
        config=config,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"看板已生成：{output.resolve()} （{output.stat().st_size / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
