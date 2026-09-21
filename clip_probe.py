"""CLIP 实例嵌入体检：到底是提示词模板问题，还是嵌入本身没区分度。

对若干实例打印：DINO 标签、三种模板下的 CLIP top-3、相似度分布的尖锐程度。
如果 sim 分布很平（top1 与 top2 差 <0.02），说明嵌入没有类别区分度，
再怎么换模板也救不回来。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/efficient3d_robot")
sys.path.insert(0, str(ROOT))
from src.perception.open_vocab_encoder import OpenVocabularyEncoder  # noqa: E402

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

TEMPLATES = {
    "a photo of a {}": "a photo of a {}",
    "{}": "{}",
    "a {} in a room": "a {} in a room",
    "a cropped photo of a {}": "a cropped photo of a {}",
}

scene_dir = ROOT / sys.argv[1]
z = np.load(scene_dir / "embeddings" / "instance_embeddings.npz")
gids = z["global_ids"].astype(int)
emb = z["embeddings"].astype(np.float32)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
imap = json.loads((scene_dir / "fusion_attempt_01" / "instance_map.json").read_text())
dino = {int(i["global_id"]): i.get("label", "?") for i in imap["instances"]}

enc = OpenVocabularyEncoder("openai/clip-vit-base-patch32", device="cuda")
res = {}
for name, tpl in TEMPLATES.items():
    txt = np.asarray(enc.encode_texts([tpl.format(c) for c in CLASSES]),
                     dtype=np.float32).reshape(len(CLASSES), -1)
    txt = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-9)
    res[name] = emb @ txt.T

print(f"{sys.argv[1]}  实例 {len(gids)}  嵌入范数均值 {np.linalg.norm(z['embeddings'],axis=1).mean():.3f}")
print(f"{'gid':>4} {'DINO':<16}" + "".join(f"{n[:14]:>16}" for n in TEMPLATES))
for i, g in enumerate(gids):
    row = f"{g:>4} {dino.get(int(g),'?'):<16}"
    for name in TEMPLATES:
        s = res[name][i]
        k = int(np.argmax(s))
        row += f"  {CLASSES[k][:12]:>12}({s[k]:.2f})"
    print(row)

print("\n相似度尖锐度（top1 - top2，越大越有区分度）：")
for name in TEMPLATES:
    s = res[name]
    part = np.sort(s, axis=1)
    gap = part[:, -1] - part[:, -2]
    print(f"  {name:<28} 平均 top1={part[:,-1].mean():.3f}  gap={gap.mean():.4f}  "
          f"最大 gap={gap.max():.4f}")
