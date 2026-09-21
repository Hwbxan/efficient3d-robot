"""单进程在线感知序列推理器（模型只加载一次，进程内循环逐帧）。

为什么需要这个脚本
------------------
`tools/run_instance_sequence.py` 默认**逐帧 spawn 子进程**（`inspect_grounded_sam2`
+ `inspect_3d_instances`），每帧都要重新 import torch/open3d/matplotlib 并**重新加载
DINO 与 SAM2 权重**，实测 30.7 s/帧，而真实推理只有约 1.9 s——**约 90% 是脚手架开销**
（见 `tools/benchmark_latency.py` 的对照）。

本脚本把逐帧的 检测→分割→3D 提升 改成**进程内循环、模型只加载一次**，把 18 场景评测
从"不可行"变"可行"，同时是性能目标（≥6 FPS）的根基。

产物兼容性
----------
逐帧写出的 `segmentation/frame_*_instances.json` + `frame_*_masks/*.png` 以及
`instances_3d/frame_*/instances_3d.json` + `individual/*.ply` 与 `inspect_grounded_sam2`
/ `inspect_3d_instances` **字节级一致**，因此 `evaluate_against_gt`、
`inspect_instance_tracking`、`preview_instance_tracking`、`replay_instance_fusion` 可直接消费。

关联 / 预览 / 融合只在末尾对整条序列跑一次（复用 `run_instance_sequence.finish_sequence`
的 subprocess 编排，这些步骤不重载 2D 模型）。

用法
----
    python -m tools.run_sequence_efficient \
        --scene-directory datasets/processed/Replica/office0 \
        --run-directory outputs/experiments/office0_efficient \
        --frames 0 10 20 30 40 50 60 \
        --classes chair desk monitor "trash can" door
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import re
import statistics
import sys
import time
from dataclasses import replace
from dataclasses import replace as _dc_replace
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.replica_sequence import ReplicaSequence  # noqa: E402
from src.geometry.instance_lifting import (  # noqa: E402
    lift_mask_to_world,
    voxel_indices,
)
from src.geometry.instance_merging import (  # noqa: E402
    plan_merges,
    voxel_key_set,
)
from src.mapping import geometric_instance_tracker as _git  # noqa: E402
from src.geometry.mask_propagation import (  # noqa: E402
    densify_label_map,
    propagate_instance_masks,
    visible_pixel_counts,
)
from src.perception.grounding_dino_detector import (  # noqa: E402
    Detection,
    GroundingDinoDetector,
)
from src.perception.sam2_box_segmenter import (  # noqa: E402
    InstanceMask,
    Sam2BoxSegmenter,
)
from tools.run_instance_sequence import finish_sequence  # noqa: E402

MASK_COLORS_RGB = [
    (255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 180, 80),
    (220, 80, 255), (80, 255, 220), (255, 100, 180), (180, 255, 80),
    (120, 120, 255), (255, 220, 80),
]


def safe_filename(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_").lower()


def save_point_cloud(output_path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if not o3d.io.write_point_cloud(str(output_path), cloud, write_ascii=False):
        raise RuntimeError(f"点云保存失败：{output_path}")


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def nms_detections(detections, iou_threshold=0.60):
    """跨提示词组 NMS：不同组可能把同一物体用不同标签检出两遍。

    按分数降序贪心保留，IoU 超过阈值的弱框丢弃。返回保留的检测列表。
    """
    if len(detections) <= 1:
        return list(detections)
    import torch
    from torchvision.ops import nms as _nms
    boxes = torch.as_tensor(
        np.stack([np.asarray(d.box_xyxy, dtype=np.float32) for d in detections]))
    scores = torch.as_tensor(
        np.asarray([float(d.score) for d in detections], dtype=np.float32))
    keep = _nms(boxes, scores, float(iou_threshold)).cpu().numpy().tolist()
    return [detections[i] for i in keep]


def merge_frame_instances(rgb, depth_m, camera_matrix, camera_to_world,
                          detections, masks, geometries,
                          voxel_size, coverage_threshold, max_center_distance,
                          pixel_stride, erosion_iterations,
                          allow_cross_label=False, known_labels=None):
    """把单帧内重复 / 过分割的观测合成一个，返回过滤后的三份列表。

    做法：先用 3D 体素覆盖率找出应合并的下标组，把组内其余成员的 2D 掩码并到
    代表上、检测框取并集、分数取最高，再对代表重新做一次 3D 提升，最后把被
    吸收的成员从三份列表里剔除。重新提升是必须的——合并后的掩码比原来大，
    沿用旧的质心 / 包围盒会让后续的关联与评估都基于一半的物体。
    """

    entries = []
    for detection, geometry in zip(detections, geometries):
        entries.append({
            "label": detection.label,
            "voxels": voxel_key_set(geometry.points_world, voxel_size),
            "centroid": np.asarray(geometry.centroid, dtype=np.float64),
        })

    groups = plan_merges(entries, coverage_threshold, max_center_distance,
                         allow_cross_label=allow_cross_label,
                         known_labels=known_labels)
    if not groups:
        return detections, masks, geometries

    # InstanceMask / Detection 都是 frozen dataclass，合并后要换成新对象
    # 而不是改字段。
    merged_masks = list(masks)
    merged_detections = list(detections)
    absorbed = set()
    for group in groups:
        # 代表取分数最高的一个；跨标签合并时这意味着「哪个提示词更自信用哪个」。
        keeper = max(group, key=lambda i: detections[i].score)
        others = [index for index in group if index != keeper]
        for index in others:
            absorbed.add(index)
            merged_masks[keeper] = replace(
                merged_masks[keeper],
                mask=np.logical_or(
                    merged_masks[keeper].mask, merged_masks[index].mask
                ),
            )
            keep_box = merged_detections[keeper].box_xyxy
            other_box = merged_detections[index].box_xyxy
            merged_detections[keeper] = replace(
                merged_detections[keeper],
                box_xyxy=np.concatenate([
                    np.minimum(keep_box[:2], other_box[:2]),
                    np.maximum(keep_box[2:], other_box[2:]),
                ]),
                score=max(
                    merged_detections[keeper].score,
                    merged_detections[index].score,
                ),
            )

    merged_masks = [
        replace(mask, area_pixels=int(np.count_nonzero(mask.mask)))
        for mask in merged_masks
    ]

    merged_geometries = list(geometries)
    for group in groups:
        keeper = group[0]
        merged_geometries[keeper] = lift_mask_to_world(
            rgb=rgb, depth_m=depth_m, mask=merged_masks[keeper].mask,
            camera_matrix=camera_matrix, camera_to_world=camera_to_world,
            pixel_stride=pixel_stride, erosion_iterations=erosion_iterations,
        )

    kept = [index for index in range(len(detections)) if index not in absorbed]
    return (
        [merged_detections[index] for index in kept],
        [merged_masks[index] for index in kept],
        [merged_geometries[index] for index in kept],
    )


def detections_from_label_map(label_map, instance_info, min_area_pixels=150):
    """把几何传播得到的标签图还原成 detections / masks 列表。

    传播帧不跑检测，所以标签和分数只能从上一帧继承（存在 `instance_info` 里），
    框则由传播后的掩码直接取包围盒。可见像素过少的实例视为已出画或被完全遮挡，
    直接丢弃——否则会把「只剩一条边」的残影继续往下传。
    """

    detections = []
    masks = []
    local_ids = []
    counts = visible_pixel_counts(label_map)

    for local_id, area in sorted(counts.items()):
        if area < min_area_pixels:
            continue
        info = instance_info.get(local_id)
        if info is None:
            continue

        mask = label_map == local_id
        rows, cols = np.nonzero(mask)
        box = np.array(
            [cols.min(), rows.min(), cols.max(), rows.max()],
            dtype=np.float32,
        )

        detections.append(
            Detection(
                label=info["label"],
                score=info["score"],
                box_xyxy=box,
            )
        )
        masks.append(
            InstanceMask(
                mask=mask,
                predicted_iou=info.get("predicted_iou", 0.0),
                area_pixels=int(area),
            )
        )
        local_ids.append(local_id)

    return detections, masks, local_ids


def make_autocast(enabled: bool):
    """半精度推理上下文。

    Grounding DINO / SAM2 都是 transformer，fp16 在 RTX 3090 上能拿到接近
    两倍的吞吐，而检测框与掩码的数值精度需求远低于 fp32。关闭时返回一个
    空上下文，保证结果与原流水线逐位一致。
    """

    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-directory", default="datasets/processed/Replica/office0")
    parser.add_argument("--dino-model", default="checkpoints/grounding-dino-tiny")
    parser.add_argument("--sam-model", default="checkpoints/sam2.1-hiera-tiny")
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--classes", nargs="+",
                        default=["computer monitor", "chair", "desk", "trash can", "door", "sofa"])
    parser.add_argument("--class-groups-file", type=Path, default=None,
                        help="JSON：[[短语...], [短语...], ...]。给定时每个检测帧只查"
                             "其中一组并按检测帧序号轮转，降低同批短语间 softmax 竞争。")
    parser.add_argument("--label-alias-file", type=Path, default=None,
                        help="JSON：{查询短语: 规范标签}。检出后立刻把标签归一到规范名，"
                             "使同义说法检出的同一物体能被合并而不是变成重复实例。")
    # PATCH_LABEL_GATE_V1 / SUPPORT_GATE_V1
    parser.add_argument("--label-gate", default="off",
                        choices=["off", "strict", "support"],
                        help="跨帧关联的标签否决档位：off=不否决；"
                             "strict=已知标签不同即否决；"
                             "support=只在家具与非家具之间否决"
                             "（放在桌子上的纸箱不是桌子的一部分）")
    # PATCH_GROUPS_ALL_V1
    parser.add_argument("--groups-per-frame", default="rotate",
                        choices=["rotate", "all"],
                        help="rotate=每检测帧只查一组并轮转（物体观测率降为 1/N）；"
                             "all=每检测帧各组都查一遍并跨组 NMS 去重")
    parser.add_argument("--group-nms-iou", type=float, default=0.60,
                        help="跨组 NMS 的框 IoU 阈值")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--group-box-thresholds", default="",
                        help="每个提示词组单独的框阈值，逗号分隔，如 0.30,0.20；"
                             "留空则所有组都用 --box-threshold")
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-box-area-fraction", type=float, default=0.40)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--erosion-iterations", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--voxel-assoc-size", type=float, default=0.05,
                        help="跨帧关联用的体素边长（米）；0 表示不写 voxel_keys")
    parser.add_argument("--detect-interval", type=int, default=1,
                        help="每隔多少帧跑一次 Grounding DINO + SAM2；"
                             "1 表示逐帧检测（原有行为）。>1 时中间帧用深度+位姿"
                             "把上一帧掩码几何传播过来，不再跑检测与分割")
    parser.add_argument("--propagate-stride", type=int, default=2,
                        help="几何传播的像素采样步长；2 时约 11 ms/帧、IoU 0.925，"
                             "1 为逐像素（约 73 ms/帧、IoU 0.931）")
    parser.add_argument("--propagate-min-area", type=int, default=150,
                        help="传播后可见像素少于该值的实例视为已出画/被完全遮挡，丢弃")
    parser.add_argument("--skip-ply", action="store_true",
                        help="跳过每帧每实例的点云落盘（纯 IO 开销），测量真实"
                             "在线性能时应打开")
    parser.add_argument("--merge-coverage", type=float, default=0.0,
                        help="单帧内同标签观测体素覆盖率达到该值即合并；0 表示不合并（默认）")
    parser.add_argument("--merge-max-center-distance", type=float, default=0.80,
                        help="合并时额外的质心距离上限（米）")
    parser.add_argument("--cross-label-merge", dest="cross_label_merge",
                        action="store_true", default=False,
                        help="允许不同标签的观测跨标签合并（同义提示词把同一物体检出"
                             "两遍时去重），代表取分数更高的标签")
    parser.add_argument("--encoder-weights", type=Path, default=None,
                        help="Stage-5 点式 encoder 权重（提供则提取 shape/clip 嵌入，增强关联）")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True,
                        help="2D 前端用 fp16 autocast 推理（默认开，约 1.8x 加速）")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="关闭 fp16，用 fp32 复现基线数值")
    parser.add_argument("--write-previews", action="store_true",
                        help="额外写逐帧 matplotlib 3D 预览（慢，默认关以提速）")
    parser.add_argument("--no-preview", dest="write_previews", action="store_false",
                        help="显式关闭逐帧 3D 预览（默认已关，保留兼容）")
    parser.add_argument("--no-fusion", action="store_true",
                        help="跳过末尾的跨帧实例融合（A/B 实验用，省时间）")
    args = parser.parse_args()
    # PATCH_PROMPT_GROUPS_V1：别名归一 + 分组轮转
    args.label_alias = {}
    if args.label_alias_file:
        args.label_alias = json.loads(
            Path(args.label_alias_file).read_text(encoding="utf-8"))
        print(f"标签别名：{len(args.label_alias)} 条 -> "
              f"{len(set(args.label_alias.values()))} 个规范名")
    args.class_groups = None
    if args.class_groups_file:
        groups = json.loads(Path(args.class_groups_file).read_text(encoding="utf-8"))
        args.class_groups = [g for g in groups if g]
        flat = []
        for g in args.class_groups:
            for c in g:
                canon = args.label_alias.get(c, c)
                if canon not in flat:
                    flat.append(canon)
        args.classes = flat
        print(f"提示词分组：{len(args.class_groups)} 组，各组 "
              f"{[len(g) for g in args.class_groups]} 个短语，"
              f"规范标签并集 {len(flat)} 类；每个检测帧只查一组并轮转")
    elif args.label_alias:
        flat = []
        for c in args.classes:
            canon = args.label_alias.get(c, c)
            if canon not in flat:
                flat.append(canon)
        args.classes = flat
    from src.mapping import geometric_instance_tracker as _git
    _git.set_known_labels(args.classes)
    print(f"跟踪器标签集合（{len(_git.KNOWN_LABELS)} 类）：{sorted(_git.KNOWN_LABELS)}")
    return args


def main():
    args = parse_arguments()

    scene_directory = Path(args.scene_directory)
    if not scene_directory.is_absolute():
        scene_directory = (ROOT / scene_directory).resolve()
    run_directory = Path(args.run_directory)
    if not run_directory.is_absolute():
        run_directory = (ROOT / run_directory).resolve()

    # ---- 轻量校验（对齐 run_instance_sequence 的防护）----
    if args.frames[0] < 0 or any(b <= a for a, b in zip(args.frames, args.frames[1:])):
        parser_error = "帧编号必须非负且严格递增"
        raise SystemExit(f"错误：{parser_error}")
    results = scene_directory / "results"
    for frame in args.frames:
        for filename in [f"frame{frame:06d}.jpg", f"depth{frame:06d}.png"]:
            if not (results / filename).is_file():
                raise SystemExit(f"错误：缺少输入 {results / filename}")
    run_directory.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("单进程在线感知序列推理器")
    print("=" * 78)
    print(f"场景      {scene_directory}")
    print(f"帧        {args.frames[0]}..{args.frames[-1]}（{len(args.frames)} 帧）")
    print(f"设备      {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU       {torch.cuda.get_device_name(0)}")

    # ---------------- 加载（一次性，不计入每帧）----------------
    print("\n加载数据与模型……")
    sync()
    load_start = time.perf_counter()
    sequence = ReplicaSequence(scene_directory)
    detector = GroundingDinoDetector(model_id=args.dino_model)
    detect_round = 0
    _group_thresholds = None
    if args.group_box_thresholds:
        _group_thresholds = [float(x) for x in args.group_box_thresholds.split(",")]
        if args.class_groups and len(_group_thresholds) != len(args.class_groups):
            raise SystemExit("--group-box-thresholds 的个数必须与分组数一致")
    segmenter = Sam2BoxSegmenter(model_id_or_path=args.sam_model)

    encoder = None
    device = None
    if args.encoder_weights is not None:
        from src.models.point_encoder import (
            PointEncoder, PointEncoderConfig, encode_single_instance,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.encoder_weights, map_location=device)
        cfg = PointEncoderConfig(**ckpt["config"])
        encoder = PointEncoder(cfg).to(device).eval()
        encoder.load_state_dict(ckpt["model"])
        print(f"  已加载 encoder：{args.encoder_weights}")
    sync()
    print(f"  加载完成：{time.perf_counter() - load_start:.2f} s（一次性）")

    segmentation_root = run_directory / "segmentation"
    instances_root = run_directory / "instances_3d"
    segmentation_root.mkdir(parents=True, exist_ok=True)
    instances_root.mkdir(parents=True, exist_ok=True)

    # ---------------- 逐帧（进程内）----------------
    frame_records = []
    no_depth_skipped = 0
    print(f"\n开始逐帧推理……\n")
    header = f"{'帧':>6} {'DINO':>9} {'SAM2':>9} {'提升':>9} {'合计':>9}  {'实例':>4}"
    print(header)
    print("-" * len(header))

    autocast = make_autocast(args.fp16)
    print(f"2D 前端精度：{'fp16' if args.fp16 and torch.cuda.is_available() else 'fp32'}")

    prev_state = None   # 上一帧的标签图 / 实例元信息 / 深度+位姿，供几何传播使用

    for index, frame_index in enumerate(args.frames):
        stages = {}
        frame_start = time.perf_counter()

        frame = sequence[frame_index]
        rgb = frame["rgb"]

        # 检测降频：中间帧不跑 DINO+SAM2，改用深度+位姿把上一帧掩码传播过来。
        # 传播是纯 CPU numpy，不占 GPU，因此可以与后台检测并行（在线化阶段再拆线程）。
        use_propagation = (
            args.detect_interval > 1
            and prev_state is not None
            and (index % args.detect_interval) != 0
        )

        if use_propagation:
            sync(); t0 = time.perf_counter()
            propagated = propagate_instance_masks(
                prev_depth_m=prev_state["depth_m"],
                prev_label_map=prev_state["label_map"],
                cur_depth_m=frame["depth_m"],
                prev_camera_to_world=prev_state["camera_to_world"],
                cur_camera_to_world=frame["camera_to_world"],
                camera_matrix=frame["camera_matrix"],
                pixel_stride=args.propagate_stride,
            )
            propagated = densify_label_map(propagated, args.propagate_stride)
            sync(); stages["propagate"] = (time.perf_counter() - t0) * 1000.0
            stages["detect"] = 0.0
            stages["segment"] = 0.0

            detections, masks, local_ids = detections_from_label_map(
                propagated,
                prev_state["instances"],
                min_area_pixels=args.propagate_min_area,
            )
            propagated_count = len(detections)

            # 静态场景里世界坐标点云不随相机移动而变，所以传播帧**不需要重新
            # 做 3D 提升**：直接复用上一帧的几何（含 voxel_keys）。
            # 这一项就从 ~50 ms 降到接近 0，是达到实时帧率的关键。
            reused_geometries = prev_state.get("geometries", {})
            geometries = [
                reused_geometries[lid]
                for lid in local_ids
                if lid in reused_geometries
            ]
            # 个别实例可能拿不到上一帧几何（例如刚被合并过），剔除保持三者对齐
            if len(geometries) != len(detections):
                keep = [i for i, lid in enumerate(local_ids)
                        if lid in reused_geometries]
                detections = [detections[i] for i in keep]
                masks = [masks[i] for i in keep]
                local_ids = [local_ids[i] for i in keep]
            stages["lift"] = 0.0
        else:
            sync(); t0 = time.perf_counter()
            with autocast:
                # PATCH_GROUPS_ALL_V1：all=各组都查一遍再跨组 NMS；rotate=轮转
                if args.class_groups and args.groups_per_frame == "all":
                    _collected = []
                    for _gi, _g in enumerate(args.class_groups):
                        _bt = (args.box_threshold if _group_thresholds is None
                               else _group_thresholds[_gi])
                        _collected.extend(detector.predict(
                            rgb=rgb, text_queries=_g,
                            box_threshold=_bt,
                            text_threshold=args.text_threshold,
                            max_box_area_fraction=args.max_box_area_fraction,
                        ))
                    detections = nms_detections(_collected, args.group_nms_iou)
                    detect_round += 1
                else:
                    if args.class_groups:
                        _g = args.class_groups[detect_round % len(args.class_groups)]
                    else:
                        _g = args.classes
                    detect_round += 1
                    detections = detector.predict(
                        rgb=rgb, text_queries=_g,
                        box_threshold=args.box_threshold,
                        text_threshold=args.text_threshold,
                        max_box_area_fraction=args.max_box_area_fraction,
                    )
                # PATCH_PROMPT_GROUPS_V1：同义说法 -> 规范标签
                # Detection 是 frozen dataclass，只能用 replace 重建
                if args.label_alias:
                    detections = [_dc_replace(
                        _d, label=args.label_alias.get(_d.label, _d.label))
                        for _d in detections]
            sync(); stages["detect"] = (time.perf_counter() - t0) * 1000.0

            if detections:
                boxes = np.stack([d.box_xyxy for d in detections])
                sync(); t0 = time.perf_counter()
                with autocast:
                    masks = segmenter.predict(rgb=rgb, boxes_xyxy=boxes)
                sync(); stages["segment"] = (time.perf_counter() - t0) * 1000.0
            else:
                masks = []
                stages["segment"] = 0.0
            propagated_count = 0

        # ---- 3D 提升（进程内）----
        # 传播帧已直接复用上一帧几何，跳过反投影。
        if use_propagation:
            # 传播帧：几何已复用，不再反投影
            pass
        else:
            sync(); t0 = time.perf_counter()
            geometries = []
            kept_detections = []
            kept_masks = []
            for detection, instance_mask in zip(detections, masks):
                try:
                    geometry = lift_mask_to_world(
                        rgb=rgb, depth_m=frame["depth_m"], mask=instance_mask.mask,
                        camera_matrix=frame["camera_matrix"],
                        camera_to_world=frame["camera_to_world"],
                        pixel_stride=args.pixel_stride,
                        erosion_iterations=args.erosion_iterations,
                    )
                except ValueError:
                    # 掩码区域内没有有效深度（窗户/玻璃外的远景、反光面、超出
                    # 量程的平面）。扩展类别词表后会稳定遇到这类观测，不能让整条
                    # 序列崩掉：直接丢弃，并保持 detections/masks/geometries 对齐。
                    no_depth_skipped += 1
                    continue
                kept_detections.append(detection)
                kept_masks.append(instance_mask)
                geometries.append(geometry)
            detections = kept_detections
            masks = kept_masks
            sync(); stages["lift"] = (time.perf_counter() - t0) * 1000.0

        # ---- 合并单帧内的重复 / 过分割观测 ----
        if args.merge_coverage > 0 and len(geometries) > 1:
            detections, masks, geometries = merge_frame_instances(
                rgb=rgb, depth_m=frame["depth_m"],
                camera_matrix=frame["camera_matrix"],
                camera_to_world=frame["camera_to_world"],
                detections=detections, masks=masks, geometries=geometries,
                voxel_size=args.voxel_assoc_size,
                coverage_threshold=args.merge_coverage,
                max_center_distance=args.merge_max_center_distance,
                pixel_stride=args.pixel_stride,
                erosion_iterations=args.erosion_iterations,
                allow_cross_label=args.cross_label_merge,
                known_labels=_git.KNOWN_LABELS,
            )

        # ---- 记录本帧状态，供下一帧几何传播 ----
        # 必须在单帧合并之后构建：masks 已是最终版本。
        if args.detect_interval > 1:
            label_map = np.zeros(frame["depth_m"].shape, dtype=np.int32)
            instance_info = {}
            for ii, (detection, instance_mask) in enumerate(zip(detections, masks)):
                label_map[instance_mask.mask] = ii + 1   # 0 留给背景
                instance_info[ii + 1] = {
                    "label": detection.label,
                    "score": detection.score,
                    "predicted_iou": instance_mask.predicted_iou,
                }
            # 几何按 local_id 存档，供下一传播帧直接复用（世界坐标不变）
            geometry_by_local_id = {
                ii + 1: geometry
                for ii, geometry in enumerate(geometries)
            }
            prev_state = {
                "label_map": label_map,
                "instances": instance_info,
                "geometries": geometry_by_local_id,
                "depth_m": frame["depth_m"],
                "camera_to_world": frame["camera_to_world"],
            }

        # ---- 写 segmentation 产物（与 inspect_grounded_sam2 一致）----
        name = f"frame_{frame_index:06d}"
        masks_directory = segmentation_root / f"{name}_masks"
        masks_directory.mkdir(parents=True, exist_ok=True)
        seg_records = []
        for ii, (detection, instance_mask) in enumerate(zip(detections, masks)):
            label_name = safe_filename(detection.label)
            mask_path = (masks_directory / f"{ii:02d}_{label_name}.png").resolve()
            cv2.imwrite(str(mask_path), instance_mask.mask.astype(np.uint8) * 255)
            seg_records.append({
                "local_instance_id": ii,
                "label": detection.label,
                "detection_score": detection.score,
                "sam_predicted_iou": instance_mask.predicted_iou,
                "area_pixels": instance_mask.area_pixels,
                "box_xyxy": detection.box_xyxy.tolist(),
                "mask_path": str(mask_path),
            })
        (segmentation_root / f"{name}_instances.json").write_text(
            json.dumps(seg_records, indent=2, ensure_ascii=False), encoding="utf-8")

        # ---- 写 instances_3d 产物（与 inspect_3d_instances 一致）----
        frame_dir = instances_root / name
        individual_dir = frame_dir / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        lift_records = []
        for ii, (detection, geometry) in enumerate(zip(detections, geometries)):
            label_name = safe_filename(detection.label)
            ply_path = (individual_dir / f"{ii:02d}_{label_name}.ply").resolve()
            # 在线/实时场景不该在关键路径上写每帧每实例的 ply：纯 IO 开销，
            # 与感知无关。需要离线产物时才打开。
            if not args.skip_ply:
                save_point_cloud(ply_path, geometry.points_world, geometry.colors)
            entry = {
                "local_instance_id": ii,
                "label": detection.label,
                "detection_score": detection.score,
                "sam_predicted_iou": None,
                "point_count": len(geometry.points_world),
                "valid_depth_ratio": geometry.valid_depth_ratio,
                "median_depth_m": geometry.median_depth_m,
                "centroid_world": geometry.centroid.tolist(),
                "bbox_min_world": geometry.bbox_min.tolist(),
                "bbox_max_world": geometry.bbox_max.tolist(),
                "bbox_extent_m": geometry.bbox_extent.tolist(),
                "point_cloud_path": str(ply_path),
            }
            if args.voxel_assoc_size > 0:
                # 跨帧关联用的体素索引：同一物体不同视角会落到几乎相同的体素
                # 集合上，比质心距离稳得多（质心会随可见部分漂移）。
                entry["voxel_keys"] = voxel_indices(
                    geometry.points_world, args.voxel_assoc_size
                ).tolist()
            if encoder is not None and len(geometry.points_world) >= 30:
                emb = encode_single_instance(
                    encoder, points=geometry.points_world, colors=geometry.colors,
                    device=device, pool="max")
                entry["shape_embedding"] = emb["shape_embedding"].tolist()
                entry["clip_embedding"] = emb["clip_embedding"].tolist()
                entry["semantic_logits"] = emb["semantic_logits"].tolist()
            lift_records.append(entry)

        (frame_dir / "instances_3d.json").write_text(
            json.dumps(lift_records, indent=2, ensure_ascii=False), encoding="utf-8")

        total = sum(stages.values())
        mode = "propagate" if use_propagation else "detect"
        frame_records.append({"frame": frame_index, "stages": stages,
                               "total_ms": total, "instances": len(geometries),
                               "mode": mode,
                               "propagated": propagated_count})
        mark = "  ← 传播" if use_propagation else ""
        print(f"{frame_index:>6} {stages.get('detect', 0):>9.1f} "
              f"{stages.get('segment', 0):>9.1f} {stages.get('lift', 0):>9.1f} "
              f"{total:>9.1f}  {len(geometries):>4}{mark}")

    # ---------------- 汇总 ----------------
    measured = frame_records
    total_mean = statistics.fmean([f["total_ms"] for f in measured]) if measured else 0.0
    print(f"\n{'=' * 78}")
    print(f"逐帧推理汇总（n={len(measured)}）")
    print(f"{'=' * 78}")

    latency = None
    if total_mean:
        det = statistics.fmean([f["stages"]["detect"] for f in measured])
        seg = statistics.fmean([f["stages"]["segment"] for f in measured])
        lif = statistics.fmean([f["stages"]["lift"] for f in measured])
        totals = sorted(f["total_ms"] for f in measured)
        latency = {
            "measured_frames": len(measured),
            "mean_total_ms": round(total_mean, 1),
            "median_total_ms": round(statistics.median(totals), 1),
            "p95_total_ms": round(totals[min(len(totals) - 1, int(0.95 * len(totals)))], 1),
            "fps_mean": round(1000.0 / total_mean, 2),
            "detect_ms": round(det, 1),
            "segment_ms": round(seg, 1),
            "lift_ms": round(lif, 1),
            "pure_2d_ms": round(det + seg, 1),
            "note": "单进程、模型只加载一次、torch.cuda.synchronize() 计时、不写预览",
        }
        print(f"单帧均值 {total_mean:.1f} ms → {1000.0 / total_mean:.2f} FPS")
        print(f"  Grounding DINO {det:.1f} ms | SAM2 {seg:.1f} ms | 3D 提升 {lif:.1f} ms")
        print(f"纯 2D 推理 {det + seg:.1f} ms —— 占单帧 {100.0 * (det + seg) / total_mean:.1f}%")

        # 检测降频时，均值和 FPS 会被两类帧混合掩盖，必须分开看：
        # 真正决定能否跟上采集帧率的是「总耗时 / 总帧数」。
        detect_frames = [f for f in measured if f["mode"] == "detect"]
        propagate_frames = [f for f in measured if f["mode"] == "propagate"]
        if detect_frames and propagate_frames:
            d_mean = statistics.fmean([f["total_ms"] for f in detect_frames])
            p_mean = statistics.fmean([f["total_ms"] for f in propagate_frames])
            p_prop = statistics.fmean(
                [f["stages"].get("propagate", 0) for f in propagate_frames])
            print(f"\n检测帧 {len(detect_frames)} 个，均值 {d_mean:.1f} ms；"
                  f"传播帧 {len(propagate_frames)} 个，均值 {p_mean:.1f} ms"
                  f"（其中几何传播 {p_prop:.1f} ms）")
            print(f"折算每帧平均成本 {total_mean:.1f} ms → "
                  f"可持续帧率 {1000.0 / total_mean:.2f} FPS")
            latency["detect_frames"] = len(detect_frames)
            latency["propagate_frames"] = len(propagate_frames)
            latency["detect_frame_mean_ms"] = round(d_mean, 1)
            latency["propagate_frame_mean_ms"] = round(p_mean, 1)
            latency["propagate_only_ms"] = round(p_prop, 1)
            latency["detect_interval"] = args.detect_interval
    else:
        print("无帧")

    # 延迟落盘，便于跨实验/跨场景对比与审计
    latency_path = run_directory / "latency.json"
    with latency_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"frames": measured, "summary": latency},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"延迟明细：{latency_path}")
    if no_depth_skipped:
        print(f"因掩码内无有效深度而丢弃的观测：{no_depth_skipped} 个"
              f"（多为窗户/远景/反光面，扩展类别词表后会出现）")

    # ---------------- 关联 / 预览 / 融合（复用现有 finish_sequence）----------------
    # 把需要的属性挂到 args 上（finish_sequence 用到）
    args.scene_directory = str(scene_directory)
    args.voxel_size = args.voxel_size
    # PATCH_NOPREVIEW_FIX_V1：--no-preview 的 dest 是 write_previews，
    # 而 finish_sequence 读的是 args.no_preview，两边对不上导致预览从未被跳过。
    args.no_preview = not getattr(args, 'write_previews', True)
    finish_sequence(args, run_directory)


if __name__ == "__main__":
    main()
