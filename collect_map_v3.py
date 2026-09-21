"""汇总 v3 八场景 3D 语义地图（跨 2000 帧全局融合后的实例）。"""
import json
import collections
from pathlib import Path

ROOT = Path("/data/efficient3d_robot")
RUNROOT = ROOT / "outputs" / "rt8_v3"
OUT = ROOT / "demo_a_map_v3.json"

SCENES = ["office0", "office1", "office2", "office3",
          "office4", "room0", "room1", "room2"]
LABEL = {"office0": "Office 0", "office1": "Office 1", "office2": "Office 2",
         "office3": "Office 3", "office4": "Office 4", "room0": "Room 0",
         "room1": "Room 1", "room2": "Room 2"}

scenes = []
for s in SCENES:
    run = RUNROOT / s
    fmap = run / "fusion_attempt_01" / "instance_map.json"
    if not fmap.is_file():
        print(f"[跳过] {s} 无 instance_map.json")
        continue
    d = json.loads(fmap.read_text())
    insts = d.get("instances", [])
    if isinstance(insts, dict):
        insts = list(insts.values())
    by_cls = collections.Counter(
        (i.get("label") or i.get("class") or i.get("class_name") or "?").strip()
        for i in insts)
    pts = 0
    for i in insts:
        for key in ("point_count", "num_points", "n_points", "points"):
            v = i.get(key)
            if isinstance(v, int):
                pts += v
                break
    # 个体点云文件数（融合实际写出的实例）
    indiv = run / "fusion_attempt_01" / "individual"
    n_indiv = len(list(indiv.glob("*.ply"))) if indiv.is_dir() else 0
    lat = run / "latency.json"
    fps = None
    if lat.is_file():
        try:
            L = json.loads(lat.read_text())
            fps = L.get("throughput_fps") or L.get("fps")
        except Exception:
            pass
    scenes.append({
        "name": s, "label": LABEL[s], "instances": len(insts),
        "by_class": dict(by_cls.most_common()), "points": pts,
        "indiv_ply": n_indiv, "fps": fps,
        "processed_frames": d.get("processed_frames"),
    })
    print(f"{s:8s} 实例 {len(insts):3d}  个体点云 {n_indiv:3d}  "
          f"{dict(by_cls.most_common())}")

total_cls = collections.Counter()
for s in scenes:
    total_cls.update(s["by_class"])

out = {
    "scenes": scenes,
    "overall": {
        "n_scenes": len(scenes),
        "n_instances": sum(s["instances"] for s in scenes),
        "n_points": sum(s["points"] for s in scenes),
        "by_class": dict(total_cls.most_common()),
    },
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\n合计：{out['overall']['n_instances']} 个跨帧全局实例，"
      f"{out['overall']['n_points']} 点")
print("写入", OUT)
