"""对比"所有 crop 嵌入平均" vs "只聚合投给多数类的 crop"的区分度。

区分度用两个量衡量：
  top1 相似度、top1-top2 差距（越大越能区分）；
  以及和融合标签的一致性（CLIP 投票标签 vs DINO 投票标签是否一致）。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/efficient3d_robot")
sys.path.insert(0, str(ROOT))
from src.perception.open_vocab_encoder import OpenVocabularyEncoder  # noqa: E402

CLASSES = json.loads((ROOT / "canonical_classes.json").read_text())
d = Path(sys.argv[1])
z = np.load(d / "instance_embeddings.npz")
meta = json.loads((d / "instance_embeddings.json").read_text())
gids = z["global_ids"].astype(int)
enc = OpenVocabularyEncoder("openai/clip-vit-base-patch32", device="cuda")
txt = np.asarray(enc.encode_texts([f"a photo of a {c}" for c in CLASSES]),
                 dtype=np.float32).reshape(len(CLASSES), -1)
txt = txt / np.linalg.norm(txt, axis=1, keepdims=True)

dino_lab = {int(m["global_id"]): m.get("label") for m in meta["instances"]}
clip_lab = {int(m["global_id"]): m.get("clip_label") for m in meta["instances"]}


def stats(E, name):
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    s = E @ txt.T
    top = s.max(axis=1)
    part = np.sort(s, axis=1)
    gap = part[:, -1] - part[:, -2]
    idx = np.argmax(s, axis=1)
    agree = sum(1 for g, i in zip(gids, idx) if CLASSES[int(i)] == dino_lab.get(int(g)))
    print(f"{name:<12} top1={top.mean():.3f}  gap={gap.mean():.4f}  "
          f"与融合标签一致 {agree}/{len(gids)}")
    return idx


i_old = stats(z["embeddings"], "平均(旧)")
if "embeddings_refined" in z:
    i_new = stats(z["embeddings_refined"], "精炼(新)")

print(f"\n{'gid':>4} {'融合标签':<14} {'旧 top1':<12} {'新 top1':<12} {'CLIP投票标签':<14} 票数分布")
for k, g in enumerate(gids):
    old = CLASSES[int(i_old[k])]
    new = CLASSES[int(i_new[k])] if "embeddings_refined" in z else "-"
    votes = next((m.get("clip_votes") for m in meta["instances"]
                  if int(m["global_id"]) == int(g)), {})
    vs = ",".join(f"{a}:{b}" for a, b in list(votes.items())[:3])
    print(f"{g:>4} {str(dino_lab.get(int(g))):<14} {old:<12} {new:<12} "
          f"{str(clip_lab.get(int(g))):<14} {vs}")
