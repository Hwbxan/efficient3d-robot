"""为交互式 3D 文本查询地图查看器生成紧凑 JSON 数据。

读取每个场景的融合实例地图（点云 + 标签 + 包围盒）、CLIP 图像嵌入、
以及 tracking 里的「首次出现帧」，并用 CLIP 文本编码器烘焙一组预设查询
的文本嵌入。输出 per-scene 的紧凑 JSON（点云用 base64 的 Float32 存储，
查询文本嵌入也烘焙进去），以及 manifest.json（含每场景几何/开放词汇指标）。

查看器（demo_a/index.html）据此渲染可旋转的 3D 实例点云，支持：
  - 文本查询高亮（预设按钮用烘焙分数即时高亮；自由文本用浏览器内 CLIP 文本塔）
  - 「建图回放」滑块（按 first_seen_frame 逐步揭示实例，模拟实时建图）

用法（在远程 GPU 机器上）：
    /miniconda3/bin/python3 -m tools.build_viewer_data \
        --eval-root outputs/multiscene_eval_v2 --all \
        --output outputs/demo_a_data --max-points 2000
"""

import argparse
import base64
import glob
import json
import os
import struct
from pathlib import Path

import numpy as np

# 预设查询：既覆盖 Replica 常见类别，也包含短语以展示「开放词汇」能力。
# 文本嵌入用模板集成（与 build_text_embeddings.py 同一套），与图像嵌入同空间。
PRESET_QUERIES = [
    "chair", "desk", "sofa", "door", "trash can", "computer monitor",
    "book", "pillow", "picture", "wall", "floor", "cabinet", "rug",
    "lamp", "blinds", "refrigerator", "bathtub", "bed", "nightstand",
    "cushion", "plant", "sink", "toilet", "bottle", "clock", "whiteboard",
    "something to sit on", "a place to work", "a screen", "a piece of furniture",
]
TEXT_TEMPLATES = ("{}", "a photo of a {}", "a photo of {}", "a close-up photo of a {}")

NICE_SLAM_SCENES = [
    "office_0", "office_1", "office_2", "office_3", "office_4",
    "room_0", "room_1", "room_2",
]


def parse_ply_binary(path):
    """纯 numpy 解析 Open3D 导出的 binary_little_endian PLY。
    期望属性：double x,y,z + uchar red,green,blue。"""
    with open(path, "rb") as f:
        data = f.read()
    # 找到 end_header
    idx = data.find(b"end_header")
    if idx < 0:
        raise ValueError(f"{path} 不是合法 PLY")
    header = data[:idx].decode("utf-8", "ignore")
    body = data[idx + len(b"end_header"):]
    # 去掉 end_header 后的换行
    body = body.lstrip(b"\n")
    n_vert = 0
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n_vert = int(line.split()[-1])
    dtype = np.dtype([
        ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    verts = np.frombuffer(body, dtype=dtype, count=n_vert)
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype("<f4")
    rgb = np.stack([verts["r"], verts["g"], verts["b"]], axis=1).astype("u1")
    return xyz, rgb


def b64_f32(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def b64_u8(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype="u1").tobytes()).decode("ascii")


def load_embeddings(emb_path):
    if not os.path.exists(emb_path):
        return None, None
    d = np.load(emb_path, allow_pickle=True)
    gids = [int(x) for x in d["global_ids"]]
    embs = d["embeddings"].astype("<f4")
    # L2 归一化，保证与文本嵌入同尺度
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-8)
    return gids, embs


def build_scene(eval_root, scene, max_points, clip_encoder):
    scene_dir = Path(eval_root) / scene
    im_path = scene_dir / "fusion_attempt_01" / "instance_map.json"
    track_path = scene_dir / "association" / "tracking.json"
    emb_path = scene_dir / "embeddings" / "instance_embeddings.npz"
    if not im_path.exists():
        print(f"  [跳过] {scene}: 缺 {im_path}")
        return None

    im = json.loads(im_path.read_text())
    tracks = {}
    if track_path.exists():
        t = json.loads(track_path.read_text())
        for tr in t.get("tracks", []):
            tracks[int(tr["global_id"])] = int(tr.get("first_seen_frame", 0))
    gids, emb_matrix = load_embeddings(str(emb_path))
    emb_by_gid = {}
    if gids is not None:
        for i, g in enumerate(gids):
            emb_by_gid[int(g)] = emb_matrix[i]

    up_hint = str(im.get("coordinate_frame", "y_up")).lower()
    up = "z" if "z" in up_hint else "y"

    instances = []
    for inst in im["instances"]:
        gid = int(inst["global_id"])
        label = inst.get("label", "unknown")
        ply_rel = inst.get("point_cloud_path")
        if not ply_rel:
            continue
        ply_path = scene_dir / "fusion_attempt_01" / ply_rel
        if not ply_path.exists():
            continue
        xyz, rgb = parse_ply_binary(str(ply_path))
        if len(xyz) > max_points:
            sel = np.random.RandomState(gid).choice(len(xyz), max_points, replace=False)
            xyz = xyz[sel]
            rgb = rgb[sel]
        bbox = inst.get("bbox_min_world")
        bbox_max = inst.get("bbox_max_world")
        emb = emb_by_gid.get(gid)
        rec = {
            "id": gid,
            "label": label,
            "first_seen": tracks.get(gid, 0),
            "points": b64_f32(xyz),
            "colors": b64_u8(rgb),
            "n": int(len(xyz)),
        }
        if bbox is not None and bbox_max is not None:
            rec["bbox"] = [bbox, bbox_max]
        if emb is not None:
            rec["embedding"] = b64_f32(emb)
        instances.append(rec)

    if not instances:
        print(f"  [跳过] {scene}: 无可用实例点云")
        return None

    # 烘焙预设查询文本嵌入
    queries = []
    if clip_encoder is not None:
        for q in PRESET_QUERIES:
            prompts = [t.format(q) for t in TEXT_TEMPLATES]
            raw = np.asarray(clip_encoder.encode_texts(prompts), dtype="<f4")
            averaged = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8)
            avg = averaged.mean(axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-8)
            queries.append({"text": q, "embedding": b64_f32(avg.astype("<f4"))})

    # 场景级指标
    metrics = {}
    summary_path = Path(eval_root) / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        item = s.get("scenes", {}).get(scene)
        if item:
            d = item.get("detection", {})
            a = item.get("association", {})
            metrics.update({
                "ap25": round(d.get("iou_0.25", {}).get("ap", 0.0), 4),
                "ap50": round(d.get("iou_0.5", {}).get("ap", 0.0), 4),
                "fragmentation": round(a.get("fragmentation_mean", 0.0), 3),
                "single_track": round(a.get("single_track_ratio", 0.0), 3),
                "label_consistency": round(item.get("label_consistency", {}).get("ratio", 0.0), 3),
            })
    ov_path = Path(eval_root) / "open_vocab_summary.json"
    if ov_path.exists():
        ov = json.loads(ov_path.read_text())
        ss = ov.get("scene_summaries", {}).get(scene)
        if ss:
            metrics["ov_mean_ap"] = round(ss.get("mean_ap", 0.0), 4)
        metrics["ov_macro_present"] = round(ov.get("mean_ap_present_scenes", 0.0), 4)

    max_first = max((i["first_seen"] for i in instances), default=0)
    out = {
        "scene": scene,
        "up": up,
        "frame_count": int(max_first),
        "num_instances": len(instances),
        "metrics": metrics,
        "instances": instances,
        "queries": queries,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/multiscene_eval_v2")
    ap.add_argument("--scenes", default=None, help="逗号分隔；不填则 --all")
    ap.add_argument("--all", action="store_true", help="使用 8 个 Nice-SLAM 场景")
    ap.add_argument("--output", default="outputs/demo_a_data")
    ap.add_argument("--max-points", type=int, default=2000)
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.all or not args.scenes:
        scenes = NICE_SLAM_SCENES
    else:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 CLIP 文本编码器（仅用于烘焙查询文本嵌入）
    clip_encoder = None
    try:
        from src.perception.open_vocab_encoder import OpenVocabularyEncoder
        clip_encoder = OpenVocabularyEncoder(args.clip_model, device=args.device)
        print(f"CLIP 文本编码器就绪：{args.clip_model} @ {clip_encoder.device}")
    except Exception as e:  # noqa
        print(f"[警告] 无法加载 CLIP 文本编码器，仅输出点云（查询高亮不可用）：{e}")

    manifest = {"scenes": [], "clip_model": args.clip_model}
    for scene in scenes:
        print(f"处理 {scene} ...")
        out = build_scene(args.eval_root, scene, args.max_points, clip_encoder)
        if out is None:
            continue
        scene_file = out_dir / f"{scene}.json"
        scene_file.write_text(json.dumps(out, separators=(",", ":")))
        size_mb = scene_file.stat().st_size / 1e6
        print(f"  写出 {scene_file.name} ({size_mb:.2f} MB, {out['num_instances']} 实例, "
              f"{len(out['queries'])} 查询)")
        manifest["scenes"].append({
            "scene": scene,
            "file": scene_file.name,
            "num_instances": out["num_instances"],
            "frame_count": out["frame_count"],
            "metrics": out["metrics"],
        })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n完成。manifest 含 {len(manifest['scenes'])} 个场景 -> {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
