"""用 habitat-sim 把 Replica 的 18 个场景渲染成 RGB-D 序列。

为什么需要这个
--------------
`datasets/raw/replica_v1/` 里 18 个场景**只有 mesh + semantic**，
没有 `results/frame*.jpg` / `depth*.png` / `traj.txt` —— 全盘搜索确认
只有 office0 有真实的 RGB-D 序列。这让「在线流水线的多场景评测」一直做不了。

但场景目录里其实带着 habitat 格式的 stage：
    <scene>/habitat/replica_stage.stage_config.json
    <scene>/habitat/mesh_semantic.ply
    <scene>/habitat/info_semantic.json
所以 habitat-sim 可以直接加载全部 18 个场景，RGB-D 自己渲染即可。

**PTex 坑**：stage config 的 `render_asset` 指向 `../mesh.ply`，那是 PTex
格式，走几何着色器。在无头 EGL + 新驱动（580.x）下 `PTexMeshShader::link()`
必定失败：

    (0) : error C6022: No input primitive type

解法：把 `render_asset` 换成 `habitat/mesh_semantic.ply`。Replica 的语义网格
**带真实顶点色**（不是类别着色，实测单帧唯一颜色数可达 5 万），所以画质够用。

产出格式（与 `datasets/processed/Replica/office0/` 完全一致）
------------------------------------------------------------
    results/frame%06d.jpg    1200x680 RGB
    results/depth%06d.png    1200x680 uint16，米 * 6553.5
    traj.txt                 每行 16 个数，4x4 camera-to-world，行主序

内参必须匹配 `src/datasets/replica_sequence.py`：
    FX = FY = 600, CX = 599.5, CY = 339.5
即水平 FOV 90°、宽 1200 -> f = (1200/2)/tan(45°) = 600。所以渲染时
`hfov` 取 habitat 默认的 90°。

坐标系转换（关键）
------------------
Replica 的 `traj.txt` 是**场景自身坐标系**（Z 朝上）。habitat 的世界是 Y 朝上，
stage config 用 `up=[0,0,1], front=[0,1,0]` 描述这个差异，于是

    world = R_stage_to_world @ stage,   R_stage_to_world = [[1,0,0],
                                                           [0,0,1],
                                                           [0,-1,0]]

相机在 habitat agent 局部系里：forward = -Z，right = +X，down = -Y。
转成 OpenCV 的 camera-to-world（x 右、y 下、z 前）后，再左乘 `R^T` 回到场景系。

脚本自带**平面性自检**：地面在场景系里应当是一个 z ≈ 常数的水平面。
若坐标系搞错了，地面点的 z 会散开，自检会直接报出来。

用法：
    /miniconda3/envs/e3d-habitat/bin/python -m tools.render_replica_sequences \
        --scene office_1 --frames 300 --output-root datasets/processed/Replica_rendered
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 与 src/datasets/replica_sequence.py 保持一致
IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 680
DEPTH_SCALE = 6553.5
CAMERA_HEIGHT = 1.25

# stage(Z-up) -> habitat world(Y-up)
R_STAGE_TO_WORLD = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def quaternion_to_matrix(q) -> np.ndarray:
    """quaternion.quaternion -> 3x3 旋转矩阵。"""

    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_to_world_stage(state, sensor_offset) -> np.ndarray:
    """把 habitat 的 agent 状态转成**场景坐标系**下的 4x4 camera-to-world。

    相机轴映射是**实测标定**出来的，不是推导的。做法（见
    `_diag_joint_calib.py`）：用语义传感器挑出真正的 floor 像素，反投影到
    世界系，要求地面高度等于导航网格高度。联合搜索 basis × 传感器偏移轴，
    最优解是

        basis = [[1,0,0],[0,0,1],[0,-1,0]]   （恰好等于 R_STAGE_TO_WORLD）
        偏移轴 = 1（即 [0, 1.25, 0]）
        地面高度偏差 +0.037 m

    也就是说相机在 **agent 局部系**里的三个轴是
        x_cam = (1,0,0)   y_cam = (0,0,1)   z_cam = (0,-1,0)
    而不是 Y-up 帧里的 OpenCV 约定 —— 早先用错这套映射，地面高度偏了 0.845 m。

    注意：地板在这个场景里**不是单一水平面**（导航点 y 从 -0.85 到 -0.35），
    所以「平整度」不能当判据，只有高度一致性可用。
    """

    rotation = quaternion_to_matrix(state.rotation)          # agent -> world
    position = np.asarray(state.position, dtype=np.float64)

    centre_world = position + rotation @ np.asarray(sensor_offset, dtype=np.float64)

    # 每列是一个相机轴（在 agent 局部系里表达）
    axes_local = R_STAGE_TO_WORLD.T
    axes_world = rotation @ axes_local

    # 世界 -> 场景系
    world_to_stage = R_STAGE_TO_WORLD.T
    axes_stage = world_to_stage @ axes_world
    centre_stage = world_to_stage @ centre_world

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = axes_stage
    pose[:3, 3] = centre_stage
    return pose


def make_stage_config(scene_dir: Path, output_directory: Path) -> Path:
    """生成绕开 PTex 的 stage config（渲染网格换成带顶点色的语义网格）。

    **坑**：habitat 会把 stage config 里的资源路径**相对于该 config 所在目录**解析，
    所以这里的路径必须是绝对的。用相对路径（例如默认的 `datasets/raw/...`）会得到
    `<work_dir>/datasets/raw/...` 这种不存在的路径，报错却是
    "does not correspond to any existing file or primitive render asset"。
    """

    render_asset = scene_dir / "habitat/mesh_semantic.ply"
    if not render_asset.is_file():
        raise FileNotFoundError(f"语义网格不存在：{render_asset}")

    stage = {
        "render_asset": str(render_asset.resolve()),
        "collision_asset": str(render_asset.resolve()),
        "semantic_asset": str(render_asset.resolve()),
        "nav_asset": str((scene_dir / "habitat/mesh_semantic.navmesh").resolve()),
        "semantic_descriptor_filename": str(
            (scene_dir / "habitat/info_semantic.json").resolve()
        ),
        "shader_type": "flat",
        "up": [0, 0, 1],
        "front": [0, 1, 0],
        "origin": [0, 0, 0],
    }
    path = output_directory / f"{scene_dir.name}.stage_config.json"
    path.write_text(json.dumps(stage, indent=2), encoding="utf-8")
    return path


def build_simulator(scene_dir: Path, work_directory: Path):
    """构建 habitat Simulator（非 PTex）。"""

    import habitat_sim

    stage_config = make_stage_config(scene_dir, work_directory)

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(stage_config)
    backend.enable_physics = False
    backend.gpu_device_id = 0

    sensors = []
    for uuid, sensor_type in (
        ("color_sensor", habitat_sim.SensorType.COLOR),
        ("depth_sensor", habitat_sim.SensorType.DEPTH),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [IMAGE_HEIGHT, IMAGE_WIDTH]
        spec.position = [0.0, CAMERA_HEIGHT, 0.0]
        # habitat 默认 hfov=90°，正好对应 FX=600（宽 1200）
        spec.hfov = 90.0
        sensors.append(spec)

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = sensors

    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_config]))


def plan_trajectory(sim, count: int, step: float, seed: int):
    """在导航网格上走一条平滑路径，返回 [(position, yaw_deg), ...]。"""

    rng = np.random.default_rng(seed)
    start = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
    positions = [start]
    yaw = float(rng.uniform(0.0, 360.0))

    while len(positions) < count:
        previous = positions[-1]
        candidate = None
        for _ in range(24):
            # 签名是 (circle_center, radius, max_tries, island_index)
            # 半径取 step 的 2.5 倍：太小会让相机原地打转，
            # 而且容易卡在「贴着墙」的退化视角上（实测零深度占比 100%）。
            point = sim.pathfinder.get_random_navigable_point_near(
                previous, step * 2.5, 64,
            )
            if point is None:
                continue
            point = np.asarray(point, dtype=np.float32)
            if float(np.linalg.norm(point - previous)) > step * 0.5:
                candidate = point
                break
        if candidate is None:
            break

        # 朝向移动方向，并加一点抖动，避免完全是直线
        delta = candidate - previous
        yaw = float(np.degrees(np.arctan2(delta[0], -delta[2]))) + float(rng.normal(0.0, 4.0))
        positions.append(candidate)

    return positions, yaw


def floor_height_check(depth_m: np.ndarray, pose: np.ndarray, reference_height: float):
    """自检：图像底部应当落在地面上。

    场景系是 Z 朝上，导航网格给的高度就是地面高度。把图像最下面几行反投影
    到场景系，返回 (地面 z 中位数, 地面 z 标准差, 相对参考高度的偏差)。

    若坐标系或内参搞错，偏差会很大（实测错误约定下标准差可达 0.3 m）。
    """

    fx = fy = 600.0
    cx, cy = 599.5, 339.5
    height, width = depth_m.shape

    rows = np.arange(height - 40, height, 4)
    columns = np.arange(0, width, 8)
    grid_rows, grid_columns = np.meshgrid(rows, columns, indexing="ij")

    z = depth_m[grid_rows, grid_columns]
    valid = z > 0.2
    if valid.sum() < 30:
        return None

    x = (grid_columns[valid] - cx) * z[valid] / fx
    y = (grid_rows[valid] - cy) * z[valid] / fy
    points_camera = np.stack([x, y, z[valid], np.ones_like(x)], axis=0)
    points_stage = (pose @ points_camera)[:3]

    median = float(np.median(points_stage[2]))
    spread = float(np.std(points_stage[2]))
    return {
        "floor_z_median": median,
        "floor_z_std": spread,
        "offset_from_navmesh_m": median - reference_height,
    }


def frame_is_usable(depth_m: np.ndarray, rgb: np.ndarray) -> tuple[bool, str]:
    """过滤退化视角。

    随机采样导航点经常会给出「相机贴在墙里 / 被包在几何体内」的视角，
    表现为大面积零深度或整幅过暗。这类帧留着会污染数据集，直接丢掉。
    """

    empty_fraction = float((depth_m <= 0.05).mean())
    if empty_fraction > 0.45:
        return False, f"零深度占比 {empty_fraction:.2f}"

    if float(rgb.mean()) < 25.0:
        return False, f"过暗（均值 {float(rgb.mean()):.1f}）"

    if float(np.median(depth_m[depth_m > 0.05])) > 8.0:
        return False, "中位深度超过 8 m"

    return True, ""


def render_scene(args) -> dict:
    # 注意：`e3d-habitat` 环境里**没有 cv2**（只有 habitat-sim + numpy + PIL），
    # 所以这里用 PIL 写图。uint16 的 PNG 用 PIL 直接存会得到 mode 'I;16'，
    # 与官方发布的一致。
    from PIL import Image
    from habitat_sim.utils.common import quat_from_angle_axis

    # 全部转绝对路径：habitat 解析 stage config 内的资源路径时以 config 所在目录为基准
    raw_root = Path(args.raw_root).resolve()
    scene_dir = raw_root / args.scene
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"场景不存在：{scene_dir}")

    output_directory = (Path(args.output_root) / args.scene).resolve()
    results_directory = output_directory / "results"
    results_directory.mkdir(parents=True, exist_ok=True)

    work_directory = Path(args.work_directory).resolve()
    work_directory.mkdir(parents=True, exist_ok=True)

    print(f"场景 {args.scene}：构建 Simulator", flush=True)
    sim = build_simulator(scene_dir, work_directory)
    agent = sim.get_agent(0)
    sensor_offset = [0.0, CAMERA_HEIGHT, 0.0]

    print(f"规划 {args.frames} 帧轨迹（步长 {args.step} m）", flush=True)
    positions, _ = plan_trajectory(sim, args.frames, args.step, args.seed)
    print(f"实际得到 {len(positions)} 个位姿", flush=True)
    if len(positions) < 2:
        raise RuntimeError("轨迹规划失败：导航点太少")

    poses = []
    checks = []
    skipped = []
    written = 0
    for index, position in enumerate(positions):
        # 朝向下一段，形成连续运动。
        #
        # 偏航轴是 **Z**（场景系的上方向），不是 Y。相机前向在场景系里是
        # `R^T @ rot @ (0,1,0)`；只有绕 Z 旋转才能让前向保持水平 ——
        # 绕 Y 旋转等于俯仰，会把相机怼到天花板上（这正是最初的 bug）。
        if index < len(positions) - 1:
            delta = positions[index + 1] - position
        else:
            delta = position - positions[index - 1]
        yaw = np.degrees(np.arctan2(-float(delta[0]), float(delta[2])))

        state = agent.get_state()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = quat_from_angle_axis(np.deg2rad(yaw), np.array([0.0, 0.0, 1.0]))
        agent.set_state(state)

        observations = sim.get_sensor_observations()
        rgb = np.asarray(observations["color_sensor"])[..., :3]
        depth_m = np.asarray(observations["depth_sensor"]).astype(np.float32)

        usable, reason = frame_is_usable(depth_m, rgb)
        if not usable:
            skipped.append({"index": index, "reason": reason})
            continue

        pose = camera_to_world_stage(agent.get_state(), sensor_offset)
        poses.append(pose)

        # 自检：底部应当落在地面上（导航网格高度 = 场景系里的地面 z）
        if written % 20 == 0:
            check = floor_height_check(depth_m, pose, float(position[1]))
            if check is not None:
                checks.append(check)

        # 深度写成与官方一致的 uint16（米 * 6553.5）
        depth_raw = np.clip(depth_m * DEPTH_SCALE, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_raw).save(results_directory / f"depth{written:06d}.png")
        Image.fromarray(rgb.astype(np.uint8)).save(
            results_directory / f"frame{written:06d}.jpg", quality=92,
        )
        written += 1

        if written % 20 == 0:
            print(f"  已写 {written} 帧（扫描 {index + 1}/{len(positions)}），"
                  f"深度中位数 {float(np.median(depth_m[depth_m > 0.05])):.3f} m", flush=True)

    sim.close()

    if written < 10:
        raise RuntimeError(f"可用帧太少（{written}），轨迹质量不合格")

    with (output_directory / "traj.txt").open("w", encoding="utf-8") as handle:
        for pose in poses:
            handle.write(" ".join(f"{value:.18e}" for value in pose.reshape(-1)) + "\n")

    if checks:
        offsets = np.array([item["offset_from_navmesh_m"] for item in checks])
        spreads = np.array([item["floor_z_std"] for item in checks])
        floor_check = {
            "samples": len(checks),
            "offset_median_m": float(np.median(offsets)),
            "offset_abs_max_m": float(np.abs(offsets).max()),
            "spread_median_m": float(np.median(spreads)),
            "passed": bool(
                abs(float(np.median(offsets))) < 0.30
                # 底部 40 行里除了地面还有家具/墙脚，spread 天然偏大，
                # 所以这里只用来兜底（>0.6 m 说明反投影明显崩了）
                and float(np.median(spreads)) < 0.60
            ),
        }
    else:
        floor_check = {"samples": 0, "passed": False}

    report = {
        "scene": args.scene,
        "frames_written": written,
        "frames_skipped": len(skipped),
        "skip_reasons": skipped[:10],
        "resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "depth_scale": DEPTH_SCALE,
        "step_m": args.step,
        "seed": args.seed,
        "render_asset": "habitat/mesh_semantic.ply",
        "ptex_workaround": "PTexMeshShader link 失败，改用带顶点色的语义网格",
        "floor_height_check": floor_check,
        "note": (
            "自检口径：图像底部反投影到场景系后应落在地面（导航网格高度）上。"
            "offset_median 反映坐标系/内参是否正确，spread 反映地面是否平整。"
        ),
    }
    (output_directory / "render_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--raw-root", default="datasets/raw/replica_v1")
    parser.add_argument("--output-root", default="datasets/processed/Replica_rendered")
    parser.add_argument("--work-directory", default="/tmp/replica_render_work")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_arguments()
    report = render_scene(args)
    check = report["floor_height_check"]
    if not check.get("passed"):
        print(
            f"\n自检未通过：地面高度自检 {check}，坐标系或内参可能有问题",
            file=sys.stderr,
        )
        return 1
    print(
        f"\n自检通过：地面偏移中位数 {check['offset_median_m']:+.3f} m，"
        f"平整度 {check['spread_median_m']:.3f} m"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
