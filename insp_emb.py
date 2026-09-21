import json
import sys
from pathlib import Path
import numpy as np

p = Path(sys.argv[1]) / "embeddings" / "instance_embeddings.json"
d = json.loads(p.read_text())
for k in ("clip_model", "min_mask_area", "context_ratio", "crops_used",
          "instance_count", "embedding_dim"):
    print(k, "=", d.get(k))
print("instances sample:", json.dumps(d["instances"][:2], ensure_ascii=False)[:500])
z = np.load(p.parent / "instance_embeddings.npz")
print("npz:", {k: z[k].shape for k in z})
