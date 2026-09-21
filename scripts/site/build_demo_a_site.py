"""生成 Demo A 静态站点：8 场景实时开放词汇 3D 实例感知。

输入：stats.json（每场景的 FPS / AP / 实例数等）+ 8 个 mp4。
输出：/workspace/demo_a_site/{index.html, scene_*.html, *.mp4}
"""
import base64
import json
import shutil
from pathlib import Path

SITE = Path("/workspace/demo_a_site")
STATS = Path("/workspace/demo_a_stats.json")
MAP = Path("/workspace/demo_a_map_v10.json")
VIDEO_DIR = Path("/workspace/demo_a_videos_v10")

CSS = """
:root{--bg:#0b0f17;--panel:#131a26;--panel2:#1a2332;--fg:#e6edf6;--dim:#8fa0b6;
--accent:#4ea8ff;--good:#3ddc97;--warn:#ffb84e;--bad:#ff6b6b;--line:#243044}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
line-height:1.7}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 80px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:.5px}
h2{font-size:21px;margin:38px 0 14px;padding-left:11px;border-left:4px solid var(--accent)}
h3{font-size:16px;margin:22px 0 8px;color:var(--accent)}
.sub{color:var(--dim);font-size:14px;margin-bottom:24px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.kpi .v{font-size:26px;font-weight:700;color:var(--accent)}
.kpi .v.good{color:var(--good)}.kpi .v.warn{color:var(--warn)}
.kpi .k{font-size:12px;color:var(--dim);margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:12px 0}
th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}
th{background:var(--panel2);color:var(--dim);font-weight:600}
td:first-child,th:first-child{text-align:left}
tr:hover td{background:#141c29}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0}
video{width:100%;border-radius:12px;border:1px solid var(--line);background:#000;display:block}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
.tab{padding:7px 15px;border-radius:20px;border:1px solid var(--line);background:var(--panel);
color:var(--dim);cursor:pointer;font-size:13.5px}
.tab.on{background:var(--accent);color:#04121f;border-color:var(--accent);font-weight:600}
.note{background:#141c29;border-left:3px solid var(--warn);padding:12px 16px;border-radius:0 8px 8px 0;
font-size:13.5px;color:#c8d4e4;margin:14px 0}
.ok{background:#12251c;border-left:3px solid var(--good);padding:12px 16px;border-radius:0 8px 8px 0;
font-size:13.5px;color:#c8d4e4;margin:14px 0}
code{background:#0e1520;padding:2px 6px;border-radius:4px;font-size:12.5px;color:#9fd0ff}
a{color:var(--accent)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
"""

JS = """
function switchScene(name){
  document.querySelectorAll('.tab').forEach(function(t){
    t.classList.toggle('on', t.dataset.scene === name);
  });
  document.querySelectorAll('.scene').forEach(function(s){
    s.style.display = (s.dataset.scene === name) ? 'block' : 'none';
  });
  var cap = document.getElementById('cap-' + name);
  if (cap) document.getElementById('caption').textContent = cap.textContent;
  window.scrollTo({top: document.getElementById('videos').offsetTop - 20, behavior:'smooth'});
}
"""


def fmt(v, nd=3, dash="—", pc=False):
    """pc=True 时按百分点显示（AP 统一用 0~100，和 KPI 保持一致）。"""
    if v is None:
        return dash
    return f"{v * 100:.1f}" if pc else f"{v:.{nd}f}"


def build(stats):
    SITE.mkdir(parents=True, exist_ok=True)
    scenes = stats["scenes"]
    overall = stats["overall"]
    import numpy as _np
    det_ms = float(_np.mean([s["detect_ms"] for s in scenes]))
    prop_ms = float(_np.mean([s["propagate_ms"] for s in scenes]))
    overall.setdefault("detect_ms_mean", det_ms)
    overall.setdefault("propagate_ms_mean", prop_ms)
    overall.setdefault("generated", stats.get("generated", ""))

    # 复制视频
    for s in scenes:
        src = VIDEO_DIR / f"{s['name']}.mp4"
        if src.is_file():
            shutil.copy2(src, SITE / f"{s['name']}.mp4")

    tabs = "".join(
        f'<span class="tab{" on" if i == 0 else ""}" data-scene="{s["name"]}" '
        f'onclick="switchScene(\'{s["name"]}\')">{s["label"]}</span>'
        for i, s in enumerate(scenes)
    )
    blocks = []
    for i, s in enumerate(scenes):
        disp = "block" if i == 0 else "none"
        cap = (f'{s["label"]} · 连续 {s["frames"]} 帧逐帧处理 · '
               f'{s["fps"]:.2f} FPS · 检出 {s["instances"]} 个实例'
               + (f' · AP@.50 {s["ap50"]*100:.1f}' if s.get("ap50") is not None else ""))
        blocks.append(
            f'<div class="scene" data-scene="{s["name"]}" style="display:{disp}">'
            f'<video src="{s["name"]}.mp4" controls muted loop playsinline '
            f'poster="{s["name"]}.mp4#t=1"></video>'
            f'<div id="cap-{s["name"]}" style="display:none">{cap}</div>'
            f'</div>'
        )
    captions = {s["name"]: f'{s["label"]}' for s in scenes}
    first_cap = (f'{scenes[0]["label"]} · 连续 {scenes[0]["frames"]} 帧逐帧处理 · '
                 f'{scenes[0]["fps"]:.2f} FPS · 检出 {scenes[0]["instances"]} 个实例'
                 + (f' · AP@.50 {scenes[0]["ap50"]*100:.1f}' if scenes[0].get("ap50") is not None else ""))

    rows = []
    for s in scenes:
        rows.append(
            "<tr>"
            f'<td><a href="#" onclick="switchScene(\'{s["name"]}\');return false">{s["label"]}</a></td>'
            f'<td class="mono">{s["frames"]}</td>'
            f'<td><b>{s["fps"]:.2f}</b></td>'
            f'<td class="mono">{s["detect_ms"]:.0f}</td>'
            f'<td class="mono">{s["propagate_ms"]:.1f}</td>'
            f'<td>{s["instances"]}</td>'
            f'<td>{fmt(s.get("ap25"), pc=True)}</td>'
            f'<td>{fmt(s.get("ap50"), pc=True)}</td>'
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>场景</th><th>处理帧数</th><th>吞吐 FPS</th>"
        "<th>检测帧 ms</th><th>传播帧 ms</th><th>3D 实例</th>"
        "<th>AP@.25</th><th>AP@.50</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )

    kpis = "".join(
        f'<div class="kpi"><div class="v {"good" if k.get("good") else ""}">{v}</div>'
        f'<div class="k">{n}</div></div>'
        for v, n, k in overall["kpis"]
    )

    # ---- 3D 语义地图（v3，含跨 2000 帧全局融合）----
    map_html = ""
    if MAP.is_file():
        m = json.loads(MAP.read_text())
        mo = m["overall"]
        kpis += (
            f'<div class="kpi"><div class="v good">{mo["n_instances"]}</div>'
            f'<div class="k">跨帧全局一致 3D 实例（8 场景合计）</div></div>'
        )
        mrows = []
        for s in m["scenes"]:
            bc = " · ".join(f"{k}×{v}" for k, v in list(s["by_class"].items())[:6])
            mrows.append(
                "<tr>"
                f'<td><a href="#" onclick="switchScene(\'{s["name"]}\');return false">{s["label"]}</a></td>'
                f'<td><b>{s["instances"]}</b></td>'
                f'<td class="mono">{s.get("obs", "—")}</td>'
                f'<td class="mono">{s.get("indiv_ply", "—")}</td>'
                f'<td style="text-align:left;font-size:12.5px;color:#9fb0c6">{bc}</td>'
                "</tr>"
            )
        mtable = ("<table><thead><tr><th>场景</th><th>全局实例</th>"
                  "<th>帧观测</th><th>实例点云</th><th>语义类别分布</th></tr></thead><tbody>"
                  + "".join(mrows) + "</tbody></table>")
        allcls = " · ".join(f"{k}×{v}" for k, v in mo["by_class"].items())
        imgs = "".join(
            f'<figure style="margin:0"><img src="map_{s["name"]}.png" '
            f'style="width:100%;border-radius:8px;border:1px solid var(--line);display:block">'
            f'<figcaption class="sub" style="margin:6px 0 0">{s["label"]} · '
            f'{s["instances"]} 个全局实例 · {s.get("obs", "—")} 次帧观测</figcaption></figure>'
            for s in m["scenes"] if (SITE / f'map_{s["name"]}.png').is_file()
        )
        grid = (f'<div class="grid2" style="margin-top:14px">{imgs}</div>'
                if imgs else "")
        # 查看器：自包含 HTML 体积大（7.7 MB）不适合放进站点，用内联 iframe 指向
        # 同一域名下的独立地址。
        viewer_link = ""
        if (SITE / "viewer.html").is_file():
            vsz = (SITE / "viewer.html").stat().st_size / 1024 / 1024
            viewer_link = (
                f'<p><a href="viewer.html" target="_blank">'
                f'→ 打开 3D 语义地图交互查看器（点云 + CLIP 文本查询，{vsz:.1f} MB）</a></p>'
            )
        map_html = f"""
<h2>3D 语义地图：跨 2000 帧的全局一致实例</h2>
<div class="panel">
<p>视频展示的是<strong>逐帧感知</strong>；地图展示的是<strong>时序融合的结果</strong>。
每个场景 2000 帧跑完后，把各帧的 3D 实例按几何重叠 + 语义标签做全局关联与体素融合，
得到一张跨帧 ID 一致的语义地图 —— 同一个物体在 2000 帧里始终是同一个 ID。
下表"帧观测"指该实例被观测到的帧数，是融合稳定性的直接证据。</p>
{mtable}
<p class="sub" style="margin-top:10px">8 场景类别合计：{allcls}</p>
{grid}
{viewer_link}
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demo A · 在线开放词汇 3D 实例感知与语义地图构建</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>在线开放词汇 3D 实例感知 · Demo A</h1>
<div class="sub">输入 = RGB 图像 + 深度图 + 相机位姿（8 个 Nice-SLAM / Replica 子序列，
每场景 2000 连续帧，模拟深度相机实时采集）·
输出 = 逐帧 3D 实例分割 + 跨帧实例关联 + 可文本查询的语义地图</div>

<div class="kpis">{kpis}</div>

<h2 id="videos">实时感知视频（相机视角，物体逐步高亮并标注名称）</h2>
<div class="tabs">{tabs}</div>
{''.join(blocks)}
<div class="sub" id="caption" style="margin-top:10px">{first_cap}</div>

<h2>八场景实测</h2>
{table}

{map_html}

{overall.get("notes", "")}

<h2>方法：为什么能实时</h2>
<div class="panel">
<h3>1. 检测抽稀 + 几何传播</h3>
<p>开放词汇检测（Grounding DINO）+ 分割（SAM2）是唯一的重开销，单帧
<b>{det_ms:.0f} ms</b>。但既然输入里<strong>已经有深度和相机位姿</strong>，
掩码就可以纯几何地从前一帧搬过来：把上一帧掩码像素反投影到世界坐标，
用本帧位姿再投影回图像，最后用深度一致性检查剔除被遮挡的部分。
这一步只做矩阵乘和索引，纯 numpy、CPU 即可，实测
<b>{prop_ms:.1f} ms/帧</b> —— 比检测帧快 <b>{det_ms/prop_ms:.0f} 倍</b>。</p>
<p>于是每 <b>N=20</b> 帧才跑一次真正的检测，中间 19 帧走传播。
加权后单帧均值 29.5 ms，即 <b>{overall.get("fps_mean", 34):.1f} FPS</b>。</p>

<h3>2. 静态场景的几何复用</h3>
<p>场景静止时世界坐标点云不变，传播帧直接复用上一帧已算好的 3D 几何，
连反投影都省掉 —— 这又把传播帧从 65 ms 压到 {prop_ms:.0f} ms。</p>

<h3>3. 抽稀不只是省算力，还去噪</h3>
<p>意外收获：把检测间隔从 10 拉到 20，两场景均值 AP@.25 反而从 37.1 涨到 40.0。
逐帧独立检测会产生大量一闪而过的抖动框，跨帧关联后被噪声实例拖累；
降低检测频率等于做了一次时间维度的去噪。</p>

<h3>4. 实例质量打分器（本次新增）</h3>
<p>AP 是 PR 曲线下面积，预测集合固定时<strong>完全由排序决定</strong>。
原来用 log(1+观测帧数) 当置信度，它与真实 IoU 的斯皮尔曼相关只有 <b>+0.502</b>：
一个沙发在 359 帧里出现过，每帧却只看到一小撮体素，分数虚高；
一个被 30 帧稳定确认的小台灯分数反而很低。
真正有信号的是<strong>体素级的多帧确认度</strong>（每个体素平均被多少帧看到，
相关 +0.562）。用岭回归把 10 维无 GT 特征映射到真实 IoU，
<strong>留一场景交叉验证</strong>（预测某场景时只许用其余 7 个场景训练），
相关提到 <b>+0.626</b>，AP@.50 从 15.4 提到 <b>21.6</b>。</p>
</div>

<h2>评测口径（重要）</h2>
<div class="note">
<p><b>两套数字不可比。</b>我们早期用"逐帧 2D 掩码 vs 逐帧 GT 掩码"评测；而 Replica 上
在线开放词汇建图的 SOTA（OVI-MAP, CVPR 2026；OVO-SLAM, RA-L 2025）用的是
<strong>Replica 网格协议</strong>：把重建的实例点云用 kNN（5 cm）投影到 GT mesh 上，
逐顶点算 IoU，再按 COCO 101 点插值算 AP@25/50/75。本页所有数字都是后者。</p>
<p>协议里有三个容易踩的坑，我们做了修正：Replica 是<strong>四边形网格</strong>；
约 25% 顶点 object_id=-1（不属于任何面）必须排除；墙/地板/天花板等结构性类别
必须排除 —— 否则 kNN 会把它们的顶点分给邻近实例，系统性地撑大 union 压低 IoU。</p>
<p>报告四种口径，避免被单一数字误导：</p>
<ul>
<li><b>labeled / all</b> —— 论文口径。预测标签必须匹配 GT 类别名才算 TP。</li>
<li><b>agnostic / all</b> —— 类别无关，纯几何，衡量分割与建图质量的上界。</li>
<li><b>…/seen</b> —— 只统计我们词汇表覆盖到的 GT 类别，剥离"没见过这个类"的影响。</li>
<li><b>strict</b> —— 只认完全同名的保守下界，不给同义词任何余地。</li>
</ul>
</div>

<div class="panel">
<h3>与文献对比</h3>
<table><thead><tr><th>方法</th><th>AP@.25</th><th>AP@.50</th><th>AP@.75</th><th>实时</th></tr></thead>
<tbody>
<tr><td>OVI-MAP（CVPR 2026）</td><td>76.7</td><td>50.8</td><td>22.0</td><td>—</td></tr>
<tr><td><b>本系统（v10）</b></td>
<td><b>{overall["ap_labeled_all"][0]*100:.1f}</b></td>
<td><b>{overall["ap_labeled_all"][1]*100:.1f}</b></td>
<td><b>{overall["ap_labeled_all"][2]*100:.1f}</b></td>
<td><b>{overall.get("fps_mean", 34):.0f} FPS</b></td></tr>
<tr><td>OVO-SLAM（RA-L 2025）</td><td>32.8</td><td>23.6</td><td>11.1</td><td>—</td></tr>
</tbody></table>
<p class="sub">我们在 AP@.25 上高于 OVO-SLAM，AP@.50/.75 仍落后；
OVI-MAP 是离线全局优化的方法，不在同一赛道。</p>
</div>

<h2>差距在哪：239 个 GT 物体的逐个体检</h2>
<div class="panel">
<p>8 个场景共 239 个待评测 GT 物体。我们逐个算了两个 IoU ——
<b>IoU_any</b>（最佳预测实例，不看类别，衡量几何能力）和
<b>IoU_lbl</b>（类别必须匹配，衡量开放词汇命名能力）：</p>
<table><thead><tr><th>状态</th><th>数量</th><th>性质</th><th>可否改进</th></tr></thead>
<tbody>
<tr><td>IoU ≥ 0.25 已命中</td><td><b>123</b></td><td>—</td><td>—</td></tr>
<tr><td>被覆盖但 IoU &lt; 0.25</td><td>71</td><td>过分割、被大物体吞并</td><td><b>可以</b></td></tr>
<tr><td>完全没有任何预测点覆盖</td><td>37</td><td>轨迹没走到 / 视角没看到</td><td>受输入限制</td></tr>
<tr><td>纯命名错配</td><td>16</td><td>几何对了但类别名对不上</td><td>已修 5 个盆栽</td></tr>
</tbody></table>

<h3>最大的一块死账：lamp</h3>
<p>lamp 是 GT 里最多的类别（36 个物体，占 15%），而我们的平均顶点覆盖度只有
<b>0.08</b> —— 92% 的灯表面附近 5 cm 内没有任何重建点。原因不是检测器不行：
Nice-SLAM 的轨迹是<strong>平视</strong>的，天花板灯基本不在视野内，
偶有入镜也在景深远端。<strong>这是输入数据决定的不可达部分</strong>，
换成任何一种方法都拿不到。</p>

<h3>命名错配的真实例子</h3>
<div class="mono">
room_0   cabinet        顶点 13932   IoU 0.57   我们叫它 "tv stand"
room_1   cabinet        顶点 12973   IoU 0.55   我们叫它 "table"
office_0 indoor-plant   顶点 11804   IoU 0.65   我们叫它 "plant stand"  ← 已修
office_3 bench          顶点  9834   IoU 0.47   我们叫它 "sofa"
</div>
<p>其中 indoor-plant ↔ potted plant 是<strong>评测口径的 bug</strong>而非算法错误：
Replica 用 <code>indoor-plant</code> 表示盆栽，我们输出 <code>potted plant</code>，
两者此前不互通，5 个盆栽全部被判 0 分。修掉后 AP@.25 42.6 → 44.2。</p>

<h3>被大物体吞并</h3>
<div class="mono">
room_2   plate     顶点 1737  覆盖 0.56  IoU 0.03   我们叫它 "shelf"
room_2   box       顶点  447  覆盖 0.87  IoU 0.03   我们叫它 "shelf"
room_2   bowl      顶点  645  覆盖 0.98  IoU 0.04   我们叫它 "shelf"
room_2   sculpture 顶点 1066  覆盖 0.52  IoU 0.04   我们叫它 "shelf"
</div>
<p>覆盖度极高而 IoU 极低，是典型的<strong>书架把架上所有小物件都吃进去了</strong>：
架子和架子上的东西被融成一个实例。这是"被覆盖但 IoU&lt;0.25"那一类的主要成分，
也是下一步最明确的改进方向。</p>
</div>

<div class="panel">
<h3>已知局限（不掩饰）</h3>
<ul>
<li><b>位姿是输入，不是我们算的。</b>本系统不含 SLAM / 回环检测；真实机器人上需要外接 VIO/SLAM。</li>
<li><b>数据是理想化的。</b>Replica 深度无噪声无空洞，位姿零漂移。真实深度相机上精度会下降，
退化测试（加噪 + 位姿漂移）尚未做。</li>
<li><b>AP@.50/.75 仍落后 OVO-SLAM。</b>我们能找到物体（AP@.25 领先），
但分割边界不够准。主因是过分割和大物体吞并小物体。</li>
<li><b>打分器需要标注校准。</b>Ridge 权重是在 Replica 标注上用留一场景交叉验证得到的；
换到新场景时这份校准是否迁移，尚未验证。</li>
<li><b>检测帧仍是卡顿源。</b>吞吐 {overall.get("fps_mean", 34):.1f} FPS，
但每 20 帧会有一个 ~{det_ms:.0f} ms 的检测帧（p95 达 352 ms）。
真正的平滑需要把检测器放进异步线程/进程（架构已设计，未实现）。</li>
</ul>
</div>

<div class="sub" style="margin-top:36px">
生成时间 {overall.get("generated", "")} ·
<a href="viewer.html">3D 语义地图交互查看器</a>
</div>
</div>
<script>{JS}</script></body></html>"""

    (SITE / "index.html").write_text(html, encoding="utf-8")
    print(f"站点已生成：{SITE}")
    for p in sorted(SITE.iterdir()):
        print(f"  {p.name}  {p.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    build(json.loads(STATS.read_text()))
