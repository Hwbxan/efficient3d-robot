"""生成 Demo A 的「实时建图 + 文本查询」短视频（mp4）。

用离线 matplotlib 渲染：按 first_seen_frame 逐步揭示实例（模拟实时建图），
结尾对某个开放词汇查询（默认 chair）做高亮。最后用 ffmpeg 合成 mp4。

用法（远程 GPU 机器）：
    /miniconda3/bin/python3 -m tools.make_demo_video \
        --data outputs/demo_a_data --scene office_0 \
        --query "chair" --frames 120 --out outputs/demo_video
"""

import argparse
import base64
import json
import os
from pathlib import Path

import numpy as np


def b64_f32(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<f4")


def b64_u8(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="u1")


PALETTE = ['#e6194B','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#42d4f4',
  '#f032e6','#bfef45','#fabed4','#469990','#dcbeff','#9A6324','#fffac8','#800000','#aaffc3',
  '#808000','#ffd8b1','#000075','#a9a9a9','#56ff9e','#e6beff','#ff7ce5','#5c3300','#cfdf38']
LABEL_COLOR = {}
def label_color(label):
    if label in LABEL_COLOR: return LABEL_COLOR[label]
    h = sum(ord(c) for c in label) % len(PALETTE)
    c = PALETTE[h]; LABEL_COLOR[label]=c; return c
def hex2rgb(hexs):
    v=int(hexs[1:],16); return (v>>16&255)/255,(v>>8&255)/255,(v&255)/255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/demo_a_data")
    ap.add_argument("--scene", default="office_0")
    ap.add_argument("--query", default="chair")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--out", default="outputs/demo_video")
    ap.add_argument("--qthr", type=float, default=0.20)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    # 默认 DejaVu Sans 无 CJK 字形，中文标题会渲染成方块。
    # 注意：matplotlib 读取 NotoSansCJK-*.ttc 时只登记首个子字体名 "Noto Sans CJK JP"，
    # 并没有 "SC" 这一条，若写 "Noto Sans CJK SC" 会静默回退到 DejaVu Sans。
    from matplotlib import font_manager as _fm
    _installed = {f.name for f in _fm.fontManager.ttflist}
    for _f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Serif CJK JP", "DejaVu Sans"]:
        if _f in _installed:
            plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
            print(f"  视频中文字体 -> {_f}")
            break
    plt.rcParams["axes.unicode_minus"] = False

    scene = json.loads((Path(args.data)/f"{args.scene}.json").read_text())
    up = scene["up"]
    insts = []
    allmin = np.array([1e9,1e9,1e9]); allmax=np.array([-1e9,-1e9,-1e9])
    for inst in scene["instances"]:
        pos = b64_f32(inst["points"]).reshape(-1,3).astype(float)
        # 必须 reshape 成 (n,3)：否则扁平的 3n 数组会被 matplotlib 当成 3n 个独立颜色，
        # 与 n 个点数量不一致而报错。
        col = b64_u8(inst["colors"]).astype(float).reshape(-1,3)/255
        if len(pos) > 900:
            sel = np.random.RandomState(inst["id"]).choice(len(pos), 900, replace=False)
            pos, col = pos[sel], col[sel]
        if up == "z":
            pos = pos[:,[0,2,1]] * np.array([1,1,-1])  # (x,z,-y)
        insts.append({"id":inst["id"],"label":inst["label"],"fs":inst["first_seen"],
                      "pos":pos,"col":col,
                      # 必须带上 CLIP 嵌入，否则后面的查询相似度全是 -9、高亮一步都不会命中
                      "emb":b64_f32(inst["embedding"]) if inst.get("embedding") else None,
                      "lc":label_color(inst["label"])})
        allmin=np.minimum(allmin,pos.min(0)); allmax=np.maximum(allmax,pos.max(0))
    center=(allmin+allmax)/2
    # 查询嵌入
    q = next((x for x in scene["queries"] if x["text"]==args.query), None)
    if q is None:
        # 取第一个查询
        q = scene["queries"][0] if scene["queries"] else None
    qe = b64_f32(q["embedding"]) if q else None
    scores={}
    if qe is not None:
        for r in insts:
            emb = r.get("emb")
            scores[r["id"]] = float(np.dot(qe, emb)) if emb is not None else -9
    # 自适应阈值：与交互式 viewer 的 pickHits 保持一致。
    # CLIP 图文余弦分布在窄带内，固定阈值会导致「全部命中」或「全部不命中」。
    _sa = np.array(list(scores.values())) if scores else np.array([])
    thr = float(_sa.max() - max(0.02, _sa.std())) if _sa.size else args.qthr
    if _sa.size:
        print(f"  查询 “{q['text'] if q else ''}” 自适应阈值={thr:.3f} "
              f"命中={int((_sa >= thr).sum())}/{len(insts)}")

    out_dir = Path(args.out); fr_dir = out_dir/"frames"; fr_dir.mkdir(parents=True, exist_ok=True)
    N = args.frames
    n_build = int(N*0.7)
    maxfs = max((r["fs"] for r in insts), default=1) or 1
    HIGHLIGHT=np.array([1.0,0.82,0.12])
    # 未命中实例的压暗系数：太小（如 0.18）会让场景几乎全黑、看不清空间结构
    DIM=0.45
    PSIZE=3.0
    plt.ioff()

    for i in range(N):
        fig = plt.figure(figsize=(12.8,7.2), dpi=100); ax=fig.add_subplot(111,projection="3d")
        if i < n_build:
            thr = maxfs*(i+1)/n_build
            title = f"实时建图  t≈{int(thr)} 帧"
            for r in insts:
                if r["fs"] <= thr:
                    ax.scatter(r["pos"][:,0],r["pos"][:,1],r["pos"][:,2],
                               c=r["col"], s=PSIZE, marker='.', linewidths=0, depthshade=False)
        else:
            title = f"文本查询：\"{q['text'] if q else ''}\"  → 高亮命中"
            for r in insts:
                sc = scores.get(r["id"], -9)
                if sc >= thr:
                    # 展开成 (n,3)，与其它分支保持一致的「每点颜色」语义
                    c = np.repeat(HIGHLIGHT[None, :], len(r["pos"]), axis=0)
                else:
                    c = r["col"]*DIM
                ax.scatter(r["pos"][:,0],r["pos"][:,1],r["pos"][:,2],
                           c=c, s=PSIZE, marker='.', linewidths=0, depthshade=False)
        ax.set_xlim(allmin[0],allmax[0]); ax.set_ylim(allmin[1],allmax[1]); ax.set_zlim(allmin[2],allmax[2])
        ax.view_init(elev=22, azim=-60 + i*0.25)
        ax.set_axis_off()
        ax.set_title(title, color="white", fontsize=16, loc="left", pad=-10)
        fig.patch.set_facecolor("#0b0e13"); ax.set_facecolor("#0b0e13")
        p = fr_dir/f"f{ i:04d}.png"
        # 不用 bbox_inches='tight'：它会把画面裁成非 16:9 的小尺寸，直接输出整幅 1280x720
        fig.savefig(p, dpi=100, facecolor="#0b0e13")
        plt.close(fig)
        if i % 20 == 0:
            print(f"  帧 {i}/{N} 完成")
    print("渲染完成，合成 mp4 ...")
    # ffmpeg
    import subprocess
    mp4 = out_dir/f"{args.scene}_demo.mp4"
    # 本环境 ffmpeg 编译时关闭了 libx264，按可用性依次尝试软件/硬件 H.264 编码器
    cands = [("libopenh264", ["-b:v", "4M"]),
             ("h264_nvenc", ["-b:v", "4M"]),
             ("mpeg4", ["-q:v", "4"])]
    ok = False
    for enc, extra in cands:
        cmd = ["ffmpeg","-y","-framerate","15","-i",str(fr_dir/"f%04d.png"),
               "-c:v",enc,"-pix_fmt","yuv420p"] + extra + [str(mp4)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and mp4.exists() and mp4.stat().st_size > 0:
                print(f"  编码器 {enc} 成功，写出 {mp4} {mp4.stat().st_size//1024} KB")
                ok = True; break
            tail = (r.stderr or "").strip().splitlines()
            print(f"  [跳过] {enc} 失败: {tail[-1] if tail else '未知错误'}")
        except Exception as e:
            print(f"  [跳过] {enc} 异常: {e}")
    if not ok:
        print("[警告] mp4 合成失败，保留帧图于", fr_dir)


if __name__ == "__main__":
    main()
