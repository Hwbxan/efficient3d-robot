"""生成「相机第一视角」的在线感知视频（Demo A 主视频）。

与 make_demo_video.py（绕场景旋转的上帝视角点云）不同，这里渲染的是
**相机自己看到的画面**：跟随相机移动，画面里被识别出的物体用跨帧稳定的颜色
**在物体区域内柔和点亮**（不画检测矩形框、不画轮廓描边，底图纹理完全保留），
中心标一行无框文字标签；某个实例首次被确认时高亮更亮并打上 ★NEW，
左下角面板按时间累积列出「已发现物体」，直观呈现在线建图/识别逐步增长的过程。

三种绘制样式（`--style`）：
  highlight（默认）柔和填充高亮，无框无描边
  outline            掩码轮廓描边
  box                传统检测矩形框 + 标签底色条

数据来源：
  RGB         datasets/processed/Replica/<scene>/results/frame%06d.jpg
  逐帧实例    outputs/multiscene_eval_v2/<scene>/segmentation/frame_%06d_instances.json
  跨帧关联    outputs/multiscene_eval_v2/<scene>/association/tracking.json

用法：
    python -m tools.make_egocentric_video --scene office_0 --max-frames 200 \
        --out outputs/demo_video
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216),
    (245, 130, 49), (145, 30, 180), (66, 212, 244), (240, 50, 230),
    (191, 239, 69), (250, 190, 212), (70, 153, 144), (220, 190, 255),
    (154, 99, 36), (255, 250, 200), (128, 0, 0), (170, 255, 195),
]


def scene_render_directory(rgb_root: str, scene: str) -> Path:
    """office_0 -> datasets/processed/Replica/office0（官方命名无下划线）。"""
    for candidate in (scene, scene.replace("_", "")):
        path = Path(rgb_root) / candidate
        if (path / "results").exists():
            return path
    raise FileNotFoundError(f"找不到渲染目录：{rgb_root}/{scene} 或 {scene.replace('_','')}")


# PIL 默认字体不含 CJK 字形，中文会渲染成方块，必须显式指定字体文件。
# NotoSansCJK 同时覆盖拉丁字母与中日韩字形；DroidSansFallback 缺拉丁字形
# （英文/数字会变方块），仅作最后兜底。
CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def load_font(size: int):
    for path in CJK_FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=0)
            except Exception:
                continue
    print("[警告] 未找到中文字体，中文可能显示为方块")
    return ImageFont.load_default()


def hexless(idx: int):
    return PALETTE[idx % len(PALETTE)]


def soften(color, mix: float = 0.32):
    """把调色板颜色朝白色混合，得到柔和的「发光」高亮色。

    直接铺原色会让画面像被色块糊住；混白后叠加在真实影像上，
    既能看清物体范围，又保留底图纹理。
    """
    return tuple(int(c + (255 - c) * mix) for c in color)


def mask_centroid(mask: Image.Image, box, width, height):
    """掩码内像素坐标的中位数（比均值稳，凹形/多块掩码也不会跑出物体外）。"""
    arr = np.asarray(mask)
    ys, xs = np.nonzero(arr > 128)
    if len(xs) == 0:
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    cx = int(np.median(xs))
    cy = int(np.median(ys))
    return min(max(cx, 0), width - 1), min(max(cy, 0), height - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="office_0")
    ap.add_argument("--rgb-root", default="datasets/processed/Replica")
    ap.add_argument("--eval-root", default="outputs/multiscene_eval_v2")
    ap.add_argument("--max-frames", type=int, default=200, help="用多少个已分割帧")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--scale", type=int, default=1,
                    help="输出放大倍数；原始帧已是 1200x680，默认 1 不放大")
    ap.add_argument("--style", default="highlight",
                    choices=["highlight", "outline", "box"],
                    help="highlight=柔和填充高亮，无框无描边（默认）；"
                         "outline=掩码轮廓描边；box=检测矩形框")
    ap.add_argument("--mask-alpha", type=float, default=0.0,
                    help="outline/box 样式下的掩码填充透明度，0 表示不填充")
    ap.add_argument("--fill-alpha", type=float, default=0.38,
                    help="highlight 样式的填充强度（0–1）")
    ap.add_argument("--label-mode", default="text", choices=["text", "none"],
                    help="text=在物体中心画无框文字标签；none=完全不画文字")
    ap.add_argument("--outline-width", type=int, default=3,
                    help="outline 样式的轮廓粗细（像素）")
    ap.add_argument("--new-frames", type=int, default=12,
                    help="首次出现后多少帧内仍标记为 NEW")
    ap.add_argument("--out", default="outputs/demo_video")
    args = ap.parse_args()

    scene_dir = Path(args.eval_root) / args.scene
    rgb_dir = scene_render_directory(args.rgb_root, args.scene)
    print(f"RGB 目录: {rgb_dir}")

    tracking = json.loads((scene_dir / "association" / "tracking.json").read_text())
    tracks = {t["global_id"]: t for t in tracking["tracks"]}
    # (frame_index, local_instance_id) -> global_id
    local_to_global = {}
    for frame in tracking["frames"]:
        for assoc in frame.get("associations", []):
            local_to_global[(frame["frame_index"], assoc["local_instance_id"])] = assoc["global_id"]
    print(f"载入 {len(tracks)} 条轨迹，{len(local_to_global)} 条帧内关联")

    seg_files = sorted((scene_dir / "segmentation").glob("frame_*_instances.json"))
    seg_files = seg_files[: args.max_frames]
    print(f"待渲染 {len(seg_files)} 帧")

    out_dir = Path(args.out)
    frames_dir = out_dir / f"{args.scene}_ego_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    font = load_font(15)
    font_small = load_font(13)
    font_big = load_font(19)

    discovered = []          # 累计发现顺序 [(frame_idx, gid, label)]
    seen_gids = set()
    width = height = None

    for n, seg_path in enumerate(seg_files):
        frame_index = int(seg_path.name.split("_")[1])
        rgb_path = rgb_dir / "results" / f"frame{frame_index:06d}.jpg"
        if not rgb_path.exists():
            print(f"  [跳过] 缺 RGB: {rgb_path}")
            continue

        image = Image.open(rgb_path).convert("RGBA")
        width, height = image.size
        instances = json.loads(seg_path.read_text())

        base = image
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        new_this_frame = []

        for inst in instances:
            gid = local_to_global.get((frame_index, inst["local_instance_id"]))
            if gid is None:
                continue  # 未关联到轨迹的实例不画，避免闪烁噪声
            track = tracks.get(gid, {})
            label = track.get("label") or inst.get("label") or "?"
            color = hexless(gid)
            first_seen = track.get("first_seen_frame", frame_index)
            is_new = (frame_index - first_seen) <= args.new_frames

            # 掩码处理：默认 highlight —— 只在物体区域内叠一层柔和的半透明色，
            # 不画矩形检测框、不画轮廓线，纯粹「把物体点亮」，底图纹理完全保留。
            mask_path = inst.get("mask_path")
            mask = None
            if mask_path and Path(mask_path).exists():
                try:
                    mask = Image.open(mask_path).convert("L")
                    if mask.size != image.size:
                        mask = mask.resize(image.size)
                except Exception:
                    mask = None

            x1, y1, x2, y2 = inst["box_xyxy"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            if mask is not None:
                if args.style == "highlight":
                    # 刚被确认的实例更亮一点，便于看清「新发现」的瞬间
                    a = args.fill_alpha * (1.45 if is_new else 1.0)
                    a = float(min(a, 0.85))
                    alpha = mask.point(lambda v: int(v * a))
                    tint = Image.new("RGBA", image.size, soften(color) + (0,))
                    tint.putalpha(alpha)
                    base = Image.alpha_composite(base, tint)
                else:
                    if args.mask_alpha > 0:
                        alpha = mask.point(lambda v: int(v * args.mask_alpha))
                        tint = Image.new("RGBA", image.size, color + (0,))
                        tint.putalpha(alpha)
                        base = Image.alpha_composite(base, tint)
                    if args.style == "outline":
                        # 形态学梯度 = 膨胀 - 腐蚀，得到掩码边缘轮廓
                        from PIL import ImageFilter
                        k = args.outline_width
                        dilated = np.asarray(
                            mask.filter(ImageFilter.MaxFilter(2 * k + 1)),
                            dtype=np.int16)
                        eroded = np.asarray(
                            mask.filter(ImageFilter.MinFilter(2 * k + 1)),
                            dtype=np.int16)
                        edge = np.clip(dilated - eroded, 0, 255).astype(np.uint8)
                        line = Image.new("RGBA", image.size, color + (0,))
                        line.putalpha(Image.fromarray(edge, mode="L"))
                        base = Image.alpha_composite(base, line)

            if args.style == "box":
                draw.rectangle([x1, y1, x2, y2], outline=color + (255,),
                               width=4 if is_new else 2)

            if args.label_mode == "text":
                text = f"{label} #{gid}"
                if is_new:
                    text = "★ NEW · " + text
                box = draw.textbbox((0, 0), text, font=font)
                tw, th = box[2] - box[0], box[3] - box[1]
                if args.style == "box":
                    ty = max(0, y1 - th - 6)
                    draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 6],
                                   fill=color + (235,))
                    draw.text((x1 + 5, ty + 3), text, font=font,
                              fill=(15, 15, 15, 255))
                else:
                    # 无框标签：放在物体中心，黑描边保证在任何底色上都可读
                    if mask is not None:
                        cx, cy = mask_centroid(mask, (x1, y1, x2, y2),
                                               width, height)
                    else:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    tx = min(max(cx - tw // 2, 2), width - tw - 2)
                    ty = min(max(cy - th // 2, 2), height - th - 2)
                    draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255),
                              stroke_width=3, stroke_fill=(0, 0, 0, 255))

            if gid not in seen_gids:
                seen_gids.add(gid)
                discovered.append((frame_index, gid, label))
                new_this_frame.append((gid, label))

        frame = Image.alpha_composite(base, overlay).convert("RGB")
        hud = ImageDraw.Draw(frame)

        # 顶部信息条
        hud.rectangle([0, 0, width, 30], fill=(12, 16, 22))
        hud.text((10, 6), f"场景 {args.scene} · 第 {frame_index} 帧 · "
                          f"已发现 {len(seen_gids)} 个实例",
                 font=font_big, fill=(255, 255, 255))

        # 左下角：按时间累积的「已发现物体」列表（最近 8 个）
        recent = discovered[-8:]
        panel_h = 20 + 18 * len(recent)
        panel_w = 250
        top = height - panel_h - 10
        hud.rectangle([10, top, 10 + panel_w, height - 10], fill=(12, 16, 22))
        hud.text((18, top + 3), "已发现物体（按出现顺序）", font=font_small,
                 fill=(255, 210, 60))
        for i, (f_idx, gid, label) in enumerate(recent):
            y = top + 20 + 18 * i
            color = hexless(gid)
            hud.rectangle([18, y + 3, 28, y + 13], fill=color)
            hud.text((34, y), f"{label} #{gid}  (帧 {f_idx})", font=font_small,
                     fill=(230, 238, 246))

        if args.scale > 1:
            frame = frame.resize((width * args.scale, height * args.scale),
                                 Image.LANCZOS)
        frame.save(frames_dir / f"f{n:04d}.png")

        if n % 25 == 0:
            print(f"  渲染 {n}/{len(seg_files)} 帧（第 {frame_index} 帧）")

    print("合成 mp4 ...")
    mp4 = out_dir / f"{args.scene}_ego.mp4"
    candidates = [("libopenh264", ["-b:v", "6M"]),
                  ("h264_nvenc", ["-b:v", "6M"]),
                  ("mpeg4", ["-q:v", "4"])]
    ok = False
    for enc, extra in candidates:
        cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
               "-i", str(frames_dir / "f%04d.png"),
               "-c:v", enc, "-pix_fmt", "yuv420p"] + extra + [str(mp4)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and mp4.exists() and mp4.stat().st_size > 0:
            print(f"  编码器 {enc} 成功 -> {mp4} ({mp4.stat().st_size // 1024} KB)")
            ok = True
            break
        tail = (r.stderr or "").strip().splitlines()
        print(f"  [跳过] {enc}: {tail[-1] if tail else '失败'}")
    if not ok:
        print("[警告] mp4 合成失败，帧图保留于", frames_dir)
    print(f"共发现实例 {len(seen_gids)} 个")


if __name__ == "__main__":
    main()
