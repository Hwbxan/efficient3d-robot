import collections
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(f"{path} 缺失")
    sys.exit(0)

data = json.loads(path.read_text(encoding="utf-8"))
items = data["instances"] if isinstance(data, dict) else data
counter = collections.Counter(item["label"] for item in items)
print(
    f"实例 {len(items)} | unknown {counter.get('unknown', 0)} | "
    f"{dict(counter.most_common(10))}"
)
