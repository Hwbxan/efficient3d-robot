"""实例标签：DINO 逐帧投票 vs CLIP 多视角聚合，谁更准？

动机
----
v6 的 class-agnostic AP25(seen) 已到 79.7，labeled 只有 53.0 —— 掉了 26.7 分，
全是被"标签和 GT 类名对不上"扣掉的。而现有标签来自 Grounding DINO 逐帧检测的
多数投票：DINO 是在**检测那一刻**顺手给的标签，没有跨视角聚合证据。

每个融合实例其实已经有了一份跨帧聚合的 CLIP 嵌入（crops_used 常达数百），
直接用它跟 38 个提示词的文本嵌入比余弦相似度取 argmax，就是零成本的重标注
——不用重跑序列，不用训练。

本脚本在**已有嵌入**的场景上离线比较两种标签，并直接给出对应的 labeled AP，
决定要不要把它接进主流程。

用法：
  /miniconda3/bin/python3 relabel_eval.py --run-root outputs/rt8_v5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path("/data/efficient3d_robot")
sys.path.insert(0, str(ROOT))
import eval_mesh_protocol as E  # noqa: E402

SCENES = ["office_0", "office_1", "office_2", "office_3", "office_4",
          "room_0", "room_1", "room_2"]

CLASSES = [
    "tv screen", "monitor", "tablet", "chair", "stool", "sofa",
    "desk", "table", "desk organizer", "trash can", "basket",
    "tissue box", "door",
    "lamp", "cushion", "book", "pillow", "blanket", "vase",
    "bottle", "plate", "clock", "camera", "bench",
    "potted plant", "plant stand", "cabinet", "picture frame", "nightstand",
    "sculpture", "cloth", "tv stand", "candle", "pot",
    "comforter", "bed", "bowl", "shelf",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/rt8_v5")
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--min-vert", type=int, default=50)
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    args = ap.parse_args()

    # 走项目自己的封装（use_safetensors=True，torch 2.5 下能正常加载）
    from src.perception.open_vocab_encoder import OpenVocabularyEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_model = OpenVocabularyEncoder(args.clip_model, device=device)
    txt = enc_model.encode_texts([f"a photo of a {c}" for c in CLASSES])
    txt = np.asarray(txt, dtype=np.float32).reshape(len(CLASSES), -1)
    print(f"文本嵌入 {txt.shape}  设备={device}")
    print("=" * 100)
    print(f"标签对比  run={args.run_root}  kNN={args.dist}")
    print("=" * 100)

    struct_norm = {E.norm(s) for s in E.STRUCTURAL}
    tall = {"dino": [], "clip": []}
    flip = Counter()

    for scene in [s.strip() for s in args.scenes.split(",")]:
        scene_dir = ROOT / args.run_root / scene.replace("_", "")
        emb_p = scene_dir / "embeddings" / "instance_embeddings.npz"
        if not emb_p.is_file():
            print(f"  [跳过] {scene}: 无 CLIP 嵌入（先跑 extract_instance_embeddings）")
            continue
        z = np.load(emb_p)
        gids_e = z["global_ids"].astype(int)
        emb = z["embeddings"].astype(np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sim = emb @ txt.T
        top = np.argmax(sim, axis=1)
        clip_label = {int(g): CLASSES[int(t)] for g, t in zip(gids_e, top)}

        # ---- GT ----
        mesh_p = None
        for base in (ROOT / "datasets/raw/replica_v1", ROOT / "datasets/processed/Replica"):
            for cand in (base / scene / "habitat" / "mesh_semantic.ply",
                         base / scene.replace("_", "") / "habitat" / "mesh_semantic.ply"):
                if cand.is_file():
                    mesh_p = cand
                    break
            if mesh_p:
                break
        info = json.loads((mesh_p.parent / "info_semantic.json").read_text())
        obj_meta = {int(o["id"]): o for o in info["objects"]}
        xyz, tri, face_obj = E.read_semantic_ply(mesh_p)
        vobj = E.vertex_object_ids(len(xyz), tri, face_obj)
        is_struct = np.zeros(len(xyz), dtype=bool)
        for gid in np.unique(vobj):
            gid = int(gid)
            if gid < 0:
                continue
            cname = (obj_meta.get(gid) or {}).get("class_name", "?")
            if E.norm(cname) in struct_norm:
                is_struct[vobj == gid] = True
        valid = (~is_struct) & (vobj >= 0)
        idx = np.where(valid)[0]

        pred = E.load_prediction(scene_dir)
        if pred is None:
            continue
        pxyz, pids, id_score, id_label = pred
        tree = cKDTree(pxyz)
        dist, nn = tree.query(xyz[idx], distance_upper_bound=args.dist)
        hit = np.isfinite(dist)
        pid_at = np.where(hit, pids[np.minimum(nn, len(pids) - 1)], -1)
        gsub = vobj[idx]

        gids = sorted({int(g) for g in np.unique(gsub) if int(g) >= 0})
        gids = [g for g in gids if int((gsub == g).sum()) >= args.min_vert]
        n_gt = len(gids)
        gid_set = {g: k for k, g in enumerate(gids)}

        uniq = [int(p) for p in np.unique(pids)]
        best_g, best_iou = [], []
        for p in uniq:
            bi, bg = 0.0, -1
            for g in gids:
                gm = (gsub == g)
                n_g = int(gm.sum())
                inter = int(((pid_at == p) & gm).sum())
                if inter == 0:
                    continue
                n_p = int((pid_at == p).sum())
                iou = inter / max(n_g + n_p - inter, 1)
                if iou > bi:
                    bi, bg = iou, g
            best_iou.append(bi)
            best_g.append(bg)
        best_iou = np.array(best_iou)
        gt_names = [(obj_meta.get(g) or {}).get("class_name", "?") if g >= 0 else "?"
                    for g in best_g]

        dino_lab = [str(id_label.get(p, "?")) for p in uniq]
        clip_lab = [clip_label.get(p, "?") for p in uniq]
        scores = np.array([id_score.get(p, 1.0) for p in uniq])

        def aps(labels):
            okk = np.array([E.label_match(l, g) for l, g in zip(labels, gt_names)])
            gtid = np.array([gid_set[g] if g >= 0 else -1 for g in best_g])
            return {t: E.average_precision(best_iou.copy(),
                                           np.where(okk, gtid, -1), scores, t, n_gt)
                    for t in (0.25, 0.5, 0.75)}

        a_d, a_c = aps(dino_lab), aps(clip_lab)
        tall["dino"].append(a_d)
        tall["clip"].append(a_c)

        m = best_iou >= 0.25
        n_d = sum(1 for l, g, k in zip(dino_lab, gt_names, m)
                  if k and E.label_match(l, g))
        n_c = sum(1 for l, g, k in zip(clip_lab, gt_names, m)
                  if k and E.label_match(l, g))
        for d, c in zip(dino_lab, clip_lab):
            if d != c:
                flip[(d, c)] += 1

        print(f"{scene:<9} GT {n_gt:>3} 预测 {len(uniq):>3} 压中 {int(m.sum()):>3}  "
              f"标签对(DINO) {n_d:>3}  标签对(CLIP) {n_c:>3}   |  "
              f"DINO {a_d[0.25]*100:5.1f}/{a_d[0.5]*100:5.1f}/{a_d[0.75]*100:5.1f}   "
              f"CLIP {a_c[0.25]*100:5.1f}/{a_c[0.5]*100:5.1f}/{a_c[0.75]*100:5.1f}")

    print("-" * 100)
    for k in ("dino", "clip"):
        v = tall[k]
        if not v:
            continue
        f = lambda t: float(np.mean([x[t] for x in v])) * 100  # noqa: E731
        print(f"{k:<6} labeled AP25 {f(0.25):5.1f}   AP50 {f(0.5):5.1f}   AP75 {f(0.75):5.1f}")
    print("\nCLIP 相对 DINO 的改判（前 15，DINO -> CLIP）：")
    for (d, c), n in flip.most_common(15):
        print(f"  {d:<18} -> {c:<18} {n}")


if __name__ == "__main__":
    main()
