"""把 Replica 语义网格转成带标签的点云数据集（Stage 5a）。

**为什么用网格而不是渲染帧**：18 个场景里只有 office0 有渲染好的 RGB-D，
其余场景只有 `habitat/` 网格。而网格一次性免费提供了**语义标签 + 实例标签**
（面级 `object_id`），26.84 M 顶点不需要任何渲染成本。

网格布局（binary_little_endian，已在 `generate_gt_instance_masks.py` 中验证）：

    vertex: float x,y,z / float nx,ny,nz / uchar r,g,b   → stride 27 B
    face:   uint8 顶点数 n + uint32×n 索引 + uint16 object_id → 3 + 4n B

⚠️ 面是**变长多边形**（平均 ~4.7 顶点/面），不能按固定步长解析，
必须顺序扫描；多边形按扇形三角化，三角形继承 object_id。

顶点本身**没有** object_id，标签在面上，所以逐顶点标签用相邻面的
`object_id` **多数投票**得到（同一顶点可能被多个物体共享）。

输出（每个场景一个 npz）：

    xyz      (N,3) float32   世界坐标（Z 轴向上）
    normal   (N,3) float32   单位法线
    rgb      (N,3) uint8
    instance (N,)  int32     原始 object_id（场景内唯一）
    class    (N,)  int32     全局类别表索引

用法（在项目根目录）：

    # 先看有哪些场景、多大
    python -m tools.build_mesh_point_dataset --replica-root datasets/raw/replica_v1 --dry-run

    # 真正构建
    python -m tools.build_mesh_point_dataset \
        --replica-root datasets/raw/replica_v1 \
        --output-directory outputs/point_dataset \
        --voxel-size 0.01 --max-points-per-scene 200000
"""

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

# 体素下采样时，标签用多数投票而不是平均——平均会把类别索引搅成无意义的中间值。
LABEL_AGGREGATION = "majority"

# Replica 的 info_semantic.json 里这几类不是语义，而是"标注缺口"或隐私模糊区域：
#   undefined           —— 标注者没能归类（18 个场景里全都出现）
#   anonymize_picture   —— 人脸/照片模糊块
#   anonymize_text      —— 文字模糊块
# 它们占着类别槽位会同时干两件坏事：把语义头撑大，以及在 mIoU 里贡献一个
# "预测成别的语义就扣分、预测成 undefined 才加分"的伪类别。
# 所以默认把它们映射成无标注（-1），由 losses 的 ignore_index 跳过。
DEFAULT_EXCLUDED_CLASSES = ["undefined", "anonymize_picture", "anonymize_text"]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replica-root", default="datasets/raw/replica_v1",
                        help="原始 Replica 根目录（其下每个场景含 habitat/mesh_semantic.ply）")
    parser.add_argument("--output-directory", default="outputs/point_dataset")
    parser.add_argument("--voxel-size", type=float, default=0.01,
                        help="体素边长（米）；0 表示不做体素下采样")
    parser.add_argument("--max-points-per-scene", type=int, default=200000,
                        help="每场景点数上限（按 Morton 序等距抽稀）")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="只处理这些场景（默认扫描全部）")
    parser.add_argument("--scene-limit", type=int, default=None,
                        help="只处理前 N 个场景，用于快速验证流程")
    parser.add_argument("--val-scenes", type=int, default=3)
    parser.add_argument("--test-scenes", type=int, default=3)
    parser.add_argument("--exclude-classes", nargs="+", default=DEFAULT_EXCLUDED_CLASSES,
                        help="这些类别名**不占类别槽位**，其点被当作无标注（class=-1）。"
                             "默认剔除 Replica 里的噪声类：undefined（标注者未能归类）、"
                             "anonymize_picture / anonymize_text（隐私模糊区域）。"
                             "它们不是语义，训练它们只会稀释 mIoU。传空列表可关闭。")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出发现的场景与网格大小，不解析面数据")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# PLY 解析（与 generate_gt_instance_masks.py 同一套已验证实现）
# --------------------------------------------------------------------------- #


def read_ply_header(handle):
    """读取 PLY 头部，返回 (头部文本, 数据区起始偏移)。"""

    buffer = b""
    while b"end_header" not in buffer:
        chunk = handle.read(65536)
        if not chunk:
            raise ValueError("PLY 头部缺少 end_header")
        buffer += chunk
    end = buffer.find(b"end_header\n") + len(b"end_header\n")
    handle.seek(end - len(buffer), 1)
    return buffer[:end].decode("ascii"), end


def element_count(header, name):
    """从头部文本中解析 element 的数量。"""

    for line in header.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == name:
            return int(parts[2])
    raise ValueError(f"PLY 头部缺少 element {name}")


VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ]
)


def parse_polygon_faces(face_bytes, face_count):
    """顺序解析变长多边形面并扇形三角化，返回 (三角形 M×3, object_id M)。

    面记录是变长的（3 + 4n 字节），必须顺序扫描——这是唯一无法向量化的地方。
    用 memoryview + struct.unpack_from 把每次迭代的开销压到最低。

    ⚠️ **三角形数多于面数**：一个 n 边形扇形三角化出 n-2 个三角形
    （Replica 平均 ~4.7 顶点/面 → 约 2.7 个三角形/面）。所以不能按
    `face_count` 预分配，否则四边形就会越界。这里按 3 倍预留并在需要时翻倍扩容。
    """

    capacity = max(face_count * 3, 16)
    triangles = np.empty((capacity, 3), dtype=np.int64)
    object_ids = np.empty(capacity, dtype=np.int64)

    view = memoryview(face_bytes)
    size = len(face_bytes)
    offset = 0
    written = 0

    for _ in range(face_count):
        if offset + 3 > size:
            raise ValueError("面数据不完整")
        vertex_total = view[offset]
        if not 3 <= vertex_total <= 16:
            raise ValueError(f"非法面顶点数 {vertex_total}（偏移 {offset}），布局解析失败")
        end = offset + 3 + 4 * vertex_total
        if end > size:
            raise ValueError("面数据不完整")

        needed = vertex_total - 2
        if written + needed > capacity:
            grown = max(capacity, needed)
            triangles = np.concatenate(
                [triangles, np.empty((grown, 3), dtype=np.int64)], axis=0
            )
            object_ids = np.concatenate(
                [object_ids, np.empty(grown, dtype=np.int64)], axis=0
            )
            capacity += grown

        indices = struct.unpack_from("<%dI" % vertex_total, face_bytes, offset + 1)
        object_id = struct.unpack_from("<H", face_bytes, offset + 1 + 4 * vertex_total)[0]

        first = indices[0]
        for j in range(1, vertex_total - 1):
            triangles[written, 0] = first
            triangles[written, 1] = indices[j]
            triangles[written, 2] = indices[j + 1]
            object_ids[written] = object_id
            written += 1

        offset = end

    if offset != size:
        raise ValueError(f"面区解析后剩余 {size - offset} 字节未消费，布局可能不正确")
    return triangles[:written], object_ids[:written]


def read_semantic_mesh(ply_path):
    """返回 (顶点结构数组, 三角形 M×3, 三角形级 object_id M)。"""

    with open(ply_path, "rb") as handle:
        header, _ = read_ply_header(handle)
        if "binary_little_endian" not in header:
            raise ValueError("仅支持 binary_little_endian 格式的语义网格")

        vertex_count = element_count(header, "vertex")
        face_count = element_count(header, "face")

        raw_vertices = handle.read(vertex_count * VERTEX_DTYPE.itemsize)
        if len(raw_vertices) != vertex_count * VERTEX_DTYPE.itemsize:
            raise ValueError("语义网格顶点数据不完整")
        vertices = np.frombuffer(raw_vertices, dtype=VERTEX_DTYPE, count=vertex_count)

        triangles, object_ids = parse_polygon_faces(handle.read(), face_count)

    return vertices, triangles, object_ids


# --------------------------------------------------------------------------- #
# 逐顶点标签 / 体素下采样
# --------------------------------------------------------------------------- #


def majority_by_group(group, value, group_count):
    """对每个 group 取 value 的众数，返回长度 group_count 的数组。

    用 `np.lexsort((-counts, group))` 实现：先按 group 升序，组内按计数降序，
    于是每组的第一条就是众数。比 Python 循环快两个数量级。

    ⚠️ **负值必须先平移到非负**。编码 `group * scale + value` 要求
    `value ∈ [0, scale)` 才能保证唯一；否则 `group*scale + (-1)` 会和
    `(group-1)*scale + (scale-1)` 撞键。类别的 -1（无标注）正好踩这个坑，
    实测会把 -1 解成别的组、别的类别。平局时取较小的原值（`lexsort` 稳定）。
    """

    if value.size == 0:
        return np.zeros(group_count, dtype=np.int64)

    low = int(value.min())
    offset = -low if low < 0 else 0
    shifted = value.astype(np.int64) + offset
    scale = int(shifted.max()) + 1

    key = group.astype(np.int64) * scale + shifted
    unique_key, counts = np.unique(key, return_counts=True)
    group_of_key = unique_key // scale
    value_of_key = unique_key % scale

    order = np.lexsort((-counts, group_of_key))
    sorted_group = group_of_key[order]
    sorted_value = value_of_key[order]

    is_first = np.ones(sorted_group.shape[0], dtype=bool)
    is_first[1:] = sorted_group[1:] != sorted_group[:-1]

    result = np.zeros(group_count, dtype=np.int64)
    result[sorted_group[is_first]] = sorted_value[is_first] - offset
    return result


def vertex_labels_from_faces(triangles, face_object_ids, vertex_count):
    """逐顶点 object_id：对相邻面的 object_id 做多数投票。"""

    vertex_of_corner = triangles.reshape(-1)
    object_of_corner = np.repeat(face_object_ids, 3)
    return majority_by_group(vertex_of_corner, object_of_corner, vertex_count)


def voxel_downsample(points, normals, colors, instance, class_id, voxel_size):
    """按体素聚合：位置/法线/颜色取均值，实例/类别取众数。"""

    keys = np.floor(points / voxel_size).astype(np.int64)
    low = keys.min(axis=0)
    dims = (keys.max(axis=0) - low + 1).astype(np.int64)

    # 把三维体素坐标压成一维整数键，避免 np.unique(axis=0) 的开销。
    # 房间尺寸下 dims 乘积约 1e9 量级，int64 完全够用。
    flat = ((keys[:, 0] - low[0]) * dims[1] + (keys[:, 1] - low[1])) * dims[2] + (keys[:, 2] - low[2])
    _, inverse = np.unique(flat, return_inverse=True)
    inverse = inverse.reshape(-1)
    voxel_count = int(inverse.max()) + 1

    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)
    inverse_f = inverse.astype(np.float64)

    def voxel_mean(values):
        channels = values.shape[1]
        summed = np.zeros((voxel_count, channels), dtype=np.float64)
        for channel in range(channels):
            summed[:, channel] = np.bincount(
                inverse, weights=values[:, channel].astype(np.float64), minlength=voxel_count
            )
        return (summed / counts[:, None]).astype(values.dtype)

    # 颜色用众数而不是均值：均值会让相邻物体的颜色互相污染，
    # 而颜色是开放词汇头的重要线索。
    colour_key = (colors[:, 0].astype(np.int64) << 16) | (colors[:, 1].astype(np.int64) << 8) | colors[:, 2]
    colour_value = majority_by_group(inverse, colour_key, voxel_count)
    downsampled_colour = np.stack(
        [(colour_value >> 16) & 0xFF, (colour_value >> 8) & 0xFF, colour_value & 0xFF], axis=1
    ).astype(np.uint8)

    return (
        voxel_mean(points),
        voxel_mean(normals),
        downsampled_colour,
        majority_by_group(inverse, instance, voxel_count).astype(np.int32),
        majority_by_group(inverse, class_id, voxel_count).astype(np.int32),
        counts.astype(np.int64),
    )


def morton_stride_subsample(points, target_count, seed):
    """按 Morton（Z-order）序等距抽稀到 target_count 个点。

    比随机采样空间覆盖更均匀，比 FPS 便宜几个数量级。
    """

    total = points.shape[0]
    if target_count >= total:
        return np.arange(total)

    low = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - low, 1e-9)
    bits = 10
    limit = (1 << bits) - 1
    quantised = np.clip(np.round((points - low) / span * limit), 0, limit).astype(np.int64)

    code = np.zeros(total, dtype=np.int64)
    for axis in range(3):
        for bit in range(bits):
            code |= ((quantised[:, axis] >> bit) & 1) << (3 * bit + axis)

    order = np.argsort(code, kind="stable")
    picks = np.linspace(0, total - 1, target_count).round().astype(np.int64)
    return order[picks]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def discover_scenes(replica_root, only=None):
    """扫描 `<root>/*/habitat/mesh_semantic.ply`，返回 [(场景名, 网格路径, info 路径)]。"""

    root = Path(replica_root)
    if not root.is_dir():
        raise SystemExit(f"Replica 根目录不存在：{root}")

    found = []
    for mesh_path in sorted(root.glob("*/habitat/mesh_semantic.ply")):
        scene = mesh_path.parent.parent.name
        info_path = mesh_path.parent / "info_semantic.json"
        found.append((scene, mesh_path, info_path))

    if only:
        wanted = set(only)
        found = [item for item in found if item[0] in wanted]
        missing = wanted - {item[0] for item in found}
        if missing:
            raise SystemExit(f"这些场景没有找到语义网格：{sorted(missing)}")
    if not found:
        raise SystemExit(f"在 {root} 下没有找到 */habitat/mesh_semantic.ply")
    return found


def load_class_names(info_path):
    """读取 info_semantic.json，返回 (object_id → 类别名, 类别名列表)。"""

    with open(info_path, "r", encoding="utf-8") as handle:
        info = json.load(handle)

    id_to_name = {}
    for item in info.get("objects", []):
        id_to_name[int(item["id"])] = item["class_name"]
    if not id_to_name:
        raise ValueError(f"{info_path} 中没有 objects")

    names = sorted({name for name in id_to_name.values()})
    return id_to_name, names


def build_global_class_table(scene_infos, exclude=()):
    """所有场景类别名的并集，排序后作为全局类别表。

    `exclude` 里的名字**不进入类别表**，所以不占类别槽位；
    对应的物体在 `process_scene` 里会被留成无标注（-1）。
    """

    excluded = set(exclude)
    names = set()
    for id_to_name, _ in scene_infos:
        names |= set(id_to_name.values())
    return sorted(names - excluded)


def process_scene(scene, mesh_path, id_to_name, class_index, args):
    """解析一个场景，返回 (数组字典, 统计信息)。"""

    started = time.time()
    vertices, triangles, face_object_ids = read_semantic_mesh(mesh_path)
    parsed = time.time()

    vertex_count = vertices.shape[0]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    normals = np.stack([vertices["nx"], vertices["ny"], vertices["nz"]], axis=1).astype(np.float32)
    colours = np.stack([vertices["r"], vertices["g"], vertices["b"]], axis=1).astype(np.uint8)

    instance = vertex_labels_from_faces(triangles, face_object_ids, vertex_count)
    labelled = time.time()

    unknown = sorted(set(instance.tolist()) - set(id_to_name.keys()))
    class_id = np.full(vertex_count, -1, dtype=np.int32)
    excluded_objects = 0
    for object_id, name in id_to_name.items():
        index = class_index.get(name)
        if index is None:
            # 被 --exclude-classes 剔掉的噪声类别：保持 -1（无标注），
            # 既不给它类别槽位，也不让它把周围点"投票"成伪类别。
            excluded_objects += 1
            continue
        class_id[instance == object_id] = index

    total_before = vertex_count
    voxel_count = None
    if args.voxel_size > 0:
        points, normals, colours, instance, class_id, counts = voxel_downsample(
            points, normals, colours, instance, class_id, args.voxel_size
        )
        voxel_count = points.shape[0]

    after_voxel = points.shape[0]
    picked = morton_stride_subsample(points, args.max_points_per_scene, args.seed)
    points = points[picked]
    normals = normals[picked]
    colours = colours[picked]
    instance = instance[picked]
    class_id = class_id[picked]

    # 抽稀后可能丢掉小物体，记录最终留下的实例
    kept_instances = sorted(set(instance.tolist()))
    kept_classes = sorted(set(class_id.tolist()) - {-1})
    # 直方图只统计已标注点；-1 单独由 unlabelled_points 报告，
    # 否则会在 main 里被当成 class_names[-1]（最后一个类别）而污染计数。
    histogram = {}
    for value in class_id[class_id >= 0]:
        histogram[int(value)] = histogram.get(int(value), 0) + 1

    arrays = {
        "xyz": points,
        "normal": normals,
        "rgb": colours,
        "instance": instance.astype(np.int32),
        "class": class_id.astype(np.int32),
    }
    stats = {
        "scene": scene,
        "vertices": int(total_before),
        "triangles": int(triangles.shape[0]),
        "voxels": int(voxel_count) if voxel_count is not None else None,
        "points_after_voxel": int(after_voxel),
        "points": int(points.shape[0]),
        "instances_in_mesh": len(set(face_object_ids.tolist())),
        "instances_kept": len(kept_instances),
        "classes_kept": len(kept_classes),
        "unlabelled_points": int((class_id < 0).sum()),
        "excluded_objects": excluded_objects,
        "unknown_object_ids": unknown[:20],
        "class_histogram": histogram,
        "seconds": {
            "parse": round(parsed - started, 2),
            "labels": round(labelled - parsed, 2),
            "total": round(time.time() - started, 2),
        },
    }
    return arrays, stats


def main():
    args = parse_arguments()
    scenes = discover_scenes(args.replica_root, args.scenes)
    if args.scene_limit:
        scenes = scenes[:args.scene_limit]

    if args.dry_run:
        print(f"发现 {len(scenes)} 个场景（{args.replica_root}）")
        print("%-20s %12s %12s" % ("scene", "vertices", "faces"))
        total_vertices = total_faces = 0
        for scene, mesh_path, info_path in scenes:
            with open(mesh_path, "rb") as handle:
                header, _ = read_ply_header(handle)
            vertices = element_count(header, "vertex")
            faces = element_count(header, "face")
            total_vertices += vertices
            total_faces += faces
            has_info = "有" if info_path.exists() else "缺"
            print("%-20s %12d %12d   info:%s" % (scene, vertices, faces, has_info))
        print("%-20s %12d %12d" % ("合计", total_vertices, total_faces))
        return

    scene_infos = []
    for scene, _, info_path in scenes:
        if not info_path.exists():
            raise SystemExit(f"{scene} 缺少 info_semantic.json，无法得到类别名")
        scene_infos.append(load_class_names(info_path))

    class_names = build_global_class_table(scene_infos, args.exclude_classes)
    class_index = {name: index for index, name in enumerate(class_names)}
    print(f"全局类别表：{len(class_names)} 类")
    if args.exclude_classes:
        print(f"已剔除的噪声类别（其点记为无标注）：{list(args.exclude_classes)}")

    output_directory = Path(args.output_directory)
    scene_directory = output_directory / "scenes"
    scene_directory.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for position, ((scene, mesh_path, _), (id_to_name, _)) in enumerate(zip(scenes, scene_infos), 1):
        print(f"[{position}/{len(scenes)}] {scene} ...", flush=True)
        arrays, stats = process_scene(scene, mesh_path, id_to_name, class_index, args)
        np.savez_compressed(scene_directory / f"{scene}.npz", **arrays)
        all_stats.append(stats)
        print(
            "    %d 顶点 → %d 点（体素 %s）  实例 %d  类别 %d  未标注 %d  用时 %.1fs"
            % (
                stats["vertices"],
                stats["points"],
                stats["voxels"],
                stats["instances_kept"],
                stats["classes_kept"],
                stats["unlabelled_points"],
                stats["seconds"]["total"],
            ),
            flush=True,
        )

    # 场景级划分：同一个场景不能同时出现在 train 和 val/test 里，
    # 否则逐点指标会被"见过同一房间"严重高估。
    names = [stats["scene"] for stats in all_stats]
    test_count = min(args.test_scenes, max(0, len(names) - 1))
    val_count = min(args.val_scenes, max(0, len(names) - test_count - 1))
    splits = {
        "train": names[: len(names) - val_count - test_count],
        "val": names[len(names) - val_count - test_count: len(names) - test_count],
        "test": names[len(names) - test_count:],
    }

    class_histogram = np.zeros(len(class_names), dtype=np.int64)
    for stats in all_stats:
        for key, value in stats["class_histogram"].items():
            class_histogram[int(key)] += value

    manifest = {
        "replica_root": str(args.replica_root),
        "output_directory": str(output_directory),
        "voxel_size": args.voxel_size,
        "max_points_per_scene": args.max_points_per_scene,
        "label_aggregation": LABEL_AGGREGATION,
        "class_names": class_names,
        "num_classes": len(class_names),
        "excluded_classes": list(args.exclude_classes),
        "splits": splits,
        "total_points": int(sum(stats["points"] for stats in all_stats)),
        "total_instances_kept": int(sum(stats["instances_kept"] for stats in all_stats)),
        "class_histogram": {class_names[i]: int(class_histogram[i]) for i in range(len(class_names))},
        "scenes": all_stats,
        "note": (
            "逐点标签来自网格面级 object_id（顶点用相邻面多数投票）；"
            "class = -1 表示该 object_id 不在 info_semantic.json 里，"
            "或属于 excluded_classes 里被剔除的噪声类别。"
            "划分是场景级的，避免同房间泄漏。"
        ),
    }
    manifest_path = output_directory / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"\n合计 {manifest['total_points']:,} 点 / {manifest['total_instances_kept']} 实例"
          f" / {len(class_names)} 类")
    print(f"划分：train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])} 场景")
    print("点数最多的 10 个类别：")
    ordered = sorted(manifest["class_histogram"].items(), key=lambda kv: -kv[1])[:10]
    for name, count in ordered:
        share = count / max(manifest["total_points"], 1)
        print("  %-24s %10d  (%5.2f%%)" % (name, count, share * 100))
    print(f"清单：{manifest_path}")


if __name__ == "__main__":
    main()
