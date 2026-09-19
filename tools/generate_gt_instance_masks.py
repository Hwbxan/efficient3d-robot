"""从 Replica 语义网格生成逐帧 GT 实例标签。

对选定帧沿相机光线对 habitat/mesh_semantic.ply 做射线投射，
命中三角形携带的 object_id 即该像素的 GT 实例编号；同时把
射线命中的 z 深度与传感器深度图对比，校验位姿与内参是否对齐。

输出：
- instance_masks/instance%06d.png    uint16 实例编号，0 表示无标注
- previews/preview%06d.png           前若干帧的彩色可视化
- gt_manifest.json                   object_id → 类别映射与逐帧统计

用法（在项目根目录）：
python -m tools.generate_gt_instance_masks \
    --scene-directory datasets/processed/Replica/office0 \
    --replica-scene office_0 \
    --output-directory outputs/gt/office0 \
    --start-frame 0 --end-frame 600 --frame-stride 10
"""

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from src.datasets.replica_sequence import (
    CX,
    CY,
    FX,
    FY,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ReplicaSequence,
)


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


def read_semantic_mesh(ply_path):
    """解析 Replica 实例网格，返回 (顶点 N×3, 三角形 M×3, 三角形级 object_id M)。

    网格布局（binary_little_endian）：
    vertex: float x,y,z / float nx,ny,nz / uchar r,g,b        → 24 字节
    face:   uint8 顶点数 n + uint32×n 索引 + uint16 object_id  → 3 + 4n 字节

    Replica 实例网格是**混合多边形网格**（平均约 4.7 顶点/面），
    面记录变长，必须顺序解析；多边形按扇形三角化，三角形继承 object_id。
    """

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ]
    )

    with open(ply_path, "rb") as handle:
        header, _ = read_ply_header(handle)
        if "binary_little_endian" not in header:
            raise ValueError("仅支持 binary_little_endian 格式的语义网格")

        vertex_count = element_count(header, "vertex")
        face_count = element_count(header, "face")

        raw_vertices = handle.read(vertex_count * vertex_dtype.itemsize)
        if len(raw_vertices) != vertex_count * vertex_dtype.itemsize:
            raise ValueError("语义网格顶点数据不完整")
        vertices = np.frombuffer(raw_vertices, dtype=vertex_dtype, count=vertex_count)

        face_bytes = handle.read()
        triangles, object_ids = parse_polygon_faces(face_bytes, face_count)

    points = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]], axis=1
    ).astype(np.float32)
    return points, triangles, object_ids


def parse_polygon_faces(face_bytes, face_count):
    """顺序解析变长多边形面并扇形三角化，返回 (三角形 M×3, object_id M)。"""

    triangles = []
    object_ids = []
    offset = 0
    size = len(face_bytes)

    for _ in range(face_count):
        if offset + 3 > size:
            raise ValueError("面数据不完整")
        vertex_total = face_bytes[offset]
        if not 3 <= vertex_total <= 16:
            raise ValueError(f"非法面顶点数 {vertex_total}（偏移 {offset}），布局解析失败")
        end = offset + 3 + 4 * vertex_total
        if end > size:
            raise ValueError("面数据不完整")

        indices = struct.unpack_from("<%dI" % vertex_total, face_bytes, offset + 1)
        object_id = struct.unpack_from("<H", face_bytes, offset + 1 + 4 * vertex_total)[0]

        first = indices[0]
        for j in range(1, vertex_total - 1):
            triangles.append((first, indices[j], indices[j + 1]))
            object_ids.append(object_id)
        offset = end

    if offset != size:
        raise ValueError(f"面区解析后剩余 {size - offset} 字节未消费，布局可能不正确")
    return np.asarray(triangles, dtype=np.int64), np.asarray(object_ids, dtype=np.int64)


def load_class_mapping(info_path):
    """读取 info_semantic.json，返回 object_id → 类别名 映射。"""

    with open(info_path, "r", encoding="utf-8") as handle:
        info = json.load(handle)

    mapping = {}
    for item in info.get("objects", []):
        mapping[int(item["id"])] = item["class_name"]
    if not mapping:
        raise ValueError("info_semantic.json 中没有 objects")
    return mapping


def build_raycasting_scene(points, triangles):
    """构建 Open3D 射线投射场景。"""

    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(points.astype(np.float32))
    mesh.triangle.indices = o3d.core.Tensor(triangles.astype(np.uint32))
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    return scene


def pixel_rays(camera_to_world, directions_camera):
    """把相机系光线变换到世界系，返回 (像素数, 6) 的 origin+direction 数组。

    direction 保持 z 分量为 1（未归一化），因此 t_hit 直接等于 z 深度。
    """

    rotation = camera_to_world[:3, :3].astype(np.float64)
    origin = camera_to_world[:3, 3].astype(np.float64)
    world_directions = directions_camera @ rotation.T
    origins = np.broadcast_to(origin, world_directions.shape)
    return np.concatenate([origins, world_directions], axis=1).astype(np.float32)


def render_frame(scene, face_object_ids, camera_to_world, directions_camera):
    """投射一帧，返回 (uint16 实例图 H×W, 命中 z 深度 H×W)。"""

    rays = pixel_rays(camera_to_world, directions_camera)
    result = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = result["t_hit"].numpy()
    primitive_ids = result["primitive_ids"].numpy().astype(np.int64)

    hit = np.isfinite(t_hit)
    instance = np.zeros(len(t_hit), dtype=np.uint16)
    instance[hit] = face_object_ids[primitive_ids[hit]].astype(np.uint16)
    z_hit = np.where(hit, t_hit, 0.0).astype(np.float32)
    return (
        instance.reshape(IMAGE_HEIGHT, IMAGE_WIDTH),
        z_hit.reshape(IMAGE_HEIGHT, IMAGE_WIDTH),
    )


def depth_alignment_stats(z_hit, depth_m, tolerance_m):
    """对比射线命中深度与传感器深度，用于校验位姿与内参对齐。"""

    valid = depth_m > 0
    if not valid.any():
        return {"valid_pixels": 0, "median_abs_diff_m": None, "fraction_within_tolerance": None}

    diff = np.abs(z_hit[valid] - depth_m[valid])
    return {
        "valid_pixels": int(valid.sum()),
        "median_abs_diff_m": round(float(np.median(diff)), 4),
        "fraction_within_tolerance": round(float(np.mean(diff <= tolerance_m)), 4),
    }


def colorize_instance_map(instance_map):
    """把实例编号渲染成固定颜色的可视化图。"""

    unique_ids = sorted(int(value) for value in np.unique(instance_map) if value != 0)
    lookup = np.zeros((max(unique_ids) + 1 if unique_ids else 1, 3), dtype=np.uint8)
    for index, object_id in enumerate(unique_ids):
        hue = int(index * 179 / max(len(unique_ids), 1))
        color = cv2.cvtColor(
            np.array([[[hue, 220, 235]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        lookup[object_id] = color
    rgb = lookup[np.clip(instance_map.astype(np.int64), 0, len(lookup) - 1)]
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene-directory", required=True, help="处理后的场景目录（含 results/ 与 traj.txt）")
    parser.add_argument("--replica-scene", required=True, help="原始 Replica 场景名（如 office_0）")
    parser.add_argument("--replica-root", default="datasets/raw/replica_v1", help="原始 Replica 根目录")
    parser.add_argument("--output-directory", required=True, help="GT 输出目录")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="默认到最后一帧")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--preview-count", type=int, default=3, help="彩色可视化帧数")
    parser.add_argument("--depth-tolerance", type=float, default=0.05, help="深度对齐判定阈值（米）")
    return parser.parse_args()


def main():
    args = parse_arguments()

    scene_directory = Path(args.scene_directory)
    replica_scene_directory = Path(args.replica_root) / args.replica_scene
    semantic_mesh_path = replica_scene_directory / "habitat" / "mesh_semantic.ply"
    semantic_info_path = replica_scene_directory / "habitat" / "info_semantic.json"
    output_directory = Path(args.output_directory)
    mask_directory = output_directory / "instance_masks"
    preview_directory = output_directory / "previews"
    mask_directory.mkdir(parents=True, exist_ok=True)
    preview_directory.mkdir(parents=True, exist_ok=True)

    sequence = ReplicaSequence(scene_directory)
    class_mapping = load_class_mapping(semantic_info_path)
    points, triangles, face_object_ids = read_semantic_mesh(semantic_mesh_path)
    scene = build_raycasting_scene(points, triangles)

    unknown_ids = sorted(set(face_object_ids.tolist()) - set(class_mapping.keys()))
    if unknown_ids:
        print(f"警告：网格中出现 info_semantic.json 未覆盖的 object_id：{unknown_ids[:10]}")

    us, vs = np.meshgrid(np.arange(IMAGE_WIDTH), np.arange(IMAGE_HEIGHT))
    directions_camera = np.stack(
        [(us - CX) / FX, (vs - CY) / FY, np.ones_like(us, dtype=np.float64)], axis=2
    ).reshape(-1, 3)

    end_frame = args.end_frame if args.end_frame is not None else len(sequence) - 1
    frame_stats = {}
    preview_done = 0
    selected = 0

    for index in range(len(sequence)):
        frame_id = int(sequence.rgb_paths[index].stem[5:])
        if frame_id < args.start_frame or frame_id > end_frame:
            continue
        if args.frame_stride > 1 and frame_id % args.frame_stride != 0:
            continue

        item = sequence[index]
        instance_map, z_hit = render_frame(
            scene, face_object_ids, item["camera_to_world"], directions_camera
        )
        stats = depth_alignment_stats(z_hit, item["depth_m"], args.depth_tolerance)

        labeled = instance_map > 0
        visible = sorted(int(value) for value in np.unique(instance_map[labeled]))
        stats.update(
            labeled_pixels=int(labeled.sum()),
            label_coverage=round(float(labeled.mean()), 4),
            visible_objects=visible,
        )
        frame_stats[str(frame_id)] = stats
        selected += 1

        mask_path = mask_directory / f"instance{frame_id:06d}.png"
        if not cv2.imwrite(str(mask_path), instance_map):
            raise RuntimeError(f"无法写入实例掩码：{mask_path}")

        if preview_done < args.preview_count:
            preview_path = preview_directory / f"preview{frame_id:06d}.png"
            if not cv2.imwrite(str(preview_path), cv2.cvtColor(colorize_instance_map(instance_map), cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"无法写入预览：{preview_path}")
            preview_done += 1

        if stats["median_abs_diff_m"] is None or stats["median_abs_diff_m"] > args.depth_tolerance:
            print(f"警告：帧 {frame_id} 深度对齐中位误差 {stats['median_abs_diff_m']} m")

    all_visible = sorted({obj for stats in frame_stats.values() for obj in stats["visible_objects"]})
    manifest = {
        "scene_directory": str(scene_directory),
        "replica_scene": args.replica_scene,
        "semantic_mesh": str(semantic_mesh_path),
        "object_id_to_class": {str(k): v for k, v in class_mapping.items()},
        "visible_object_count": len(all_visible),
        "visible_objects": all_visible,
        "frame_count": selected,
        "depth_tolerance_m": args.depth_tolerance,
        "frames": frame_stats,
        "note": "instance 0 表示无标注；depth_check 用射线命中 z 深度与传感器深度对比校验对齐。",
    }
    manifest_path = output_directory / "gt_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    diffs = [s["median_abs_diff_m"] for s in frame_stats.values() if s["median_abs_diff_m"] is not None]
    print(f"完成：{selected} 帧，可见物体 {len(all_visible)} 个")
    if diffs:
        print(f"深度对齐中位误差：最小 {min(diffs)} m / 最大 {max(diffs)} m")
    print(f"清单：{manifest_path}")


if __name__ == "__main__":
    main()
