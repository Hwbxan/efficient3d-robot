"""离线给每个融合实例（track）提取 PointEncoder 嵌入。

目的：把跨帧关联判据从"纯手工几何"升级为"学习表示 + 几何"。
PointEncoder 是 Stage 5 训练好的点式骨干（outputs/stage5/run_s5b_metricfix/best.pt），
有 instance_embedding（16 维）分支，理论上同一物体的点应有相近嵌入。

这里不需要重跑序列：直接读已落盘的 individual/G*.ply 提实例级嵌入。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/efficient3d_robot")
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v4")
    ap.add_argument("--scenes", default="office_0,office_1")
    ap.add_argument("--weights", default="outputs/stage5/run_s5b_metricfix/best.pt")
    ap.add_argument("--max-points", type=int, default=20000,
                    help="每个实例最多采样多少点（防止显存爆）")
    a = ap.parse_args()

    from src.models.point_encoder import (
        PointEncoder, PointEncoderConfig, encode_single_instance,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ROOT / a.weights), map_location=device)
    cfg = PointEncoderConfig(**ckpt["config"])
    model = PointEncoder(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"已加载 {a.weights}  设备={device}  "
          f"instance_dim={cfg.instance_dim}  num_classes={cfg.num_classes}")

    import open3d as o3d
    run_root = ROOT / a.run_root
    for scene in [s.strip() for s in a.scenes.split(",")]:
        run_dir = run_root / scene.replace("_", "")
        fusion = run_dir / "fusion_attempt_01"
        imap = json.loads((fusion / "instance_map.json").read_text())
        out = {}
        for inst in imap["instances"]:
            gid = int(inst["global_id"])
            p = fusion / inst.get("point_cloud_path",
                                  f"individual/G{gid:03d}.ply")
            if not p.is_file():
                continue
            pcd = o3d.io.read_point_cloud(str(p))
            xyz = np.asarray(pcd.points)
            rgb = np.asarray(pcd.colors)
            if len(xyz) == 0:
                continue
            if rgb.shape[0] != xyz.shape[0]:
                rgb = np.zeros_like(xyz)
            if len(xyz) > a.max_points:
                sel = np.random.default_rng(0).choice(
                    len(xyz), a.max_points, replace=False)
                xyz, rgb = xyz[sel], rgb[sel]
            emb = encode_single_instance(model, points=xyz, colors=rgb,
                                         device=device, pool="max")
            out[gid] = dict(
                label=inst.get("label", "?"),
                obs=int(inst.get("observation_count", 0)),
                shape=emb["shape_embedding"].astype(np.float32),
                clip=emb["clip_embedding"].astype(np.float32),
                logits=emb["semantic_logits"].astype(np.float32),
            )
        dest = run_dir / "track_embeddings.npz"
        np.savez_compressed(
            str(dest),
            **{f"s_{g}": v["shape"] for g, v in out.items()},
            **{f"c_{g}": v["clip"] for g, v in out.items()},
            **{f"l_{g}": v["logits"] for g, v in out.items()},
            meta=json.dumps({str(g): {"label": v["label"], "obs": v["obs"]}
                             for g, v in out.items()}),
        )
        sh = np.stack([v["shape"] for v in out.values()])
        print(f"{scene:<10} {len(out):>3} 个实例  shape_dim={sh.shape[1]}  "
              f"clip_dim={out[list(out)[0]]['clip'].shape[0]}  → {dest.name}")


if __name__ == "__main__":
    main()
