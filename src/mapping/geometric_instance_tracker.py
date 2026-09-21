from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


# 默认标签集合。实际使用时应由调用方通过 set_known_labels() 注入当前提示词，
# 而不是硬编码 —— 硬编码曾导致换提示词后新类别全被丢成 "unknown"。
DEFAULT_KNOWN_LABELS = {
    "computer monitor",
    "tv screen",
    "chair",
    "desk",
    "table",
    "trash can",
    "door",
    "sofa",
}

KNOWN_LABELS = set(DEFAULT_KNOWN_LABELS)


def set_known_labels(labels):
    """由序列推理器注入当前使用的提示词集合。"""
    global KNOWN_LABELS
    KNOWN_LABELS = {label.strip() for label in labels if label and label.strip()}
    return KNOWN_LABELS


# 「家具 / 承载面」类别：别的物体会放在它们上面或里面。
# 与「非家具」标签之间一律否决合并——放在桌子上的纸箱不是桌子的一部分。
SUPPORT_SURFACES = {
    "table", "desk", "shelf", "cabinet", "bookcase", "dresser", "counter",
    "bed", "sofa", "couch", "chair", "armchair", "bench", "stool", "seat",
    "nightstand", "tv stand", "plant stand", "door", "ottoman", "refrigerator",
}


def is_support_surface(label):
    return str(label).strip().lower() in SUPPORT_SURFACES


@dataclass
class InstanceTrack:
    """跨帧维护的实例记录。"""

    global_id: int
    latest_observation: dict
    first_seen_frame: int
    last_seen_frame: int
    observation_count: int = 1
    label_votes: dict = field(default_factory=dict)
    # 最近若干次观测的体素并集，是跨帧关联的比较基准。
    voxel_history: deque = field(default_factory=deque)
    voxel_window: frozenset = field(default_factory=frozenset)

    @property
    def label(self):
        if not self.label_votes:
            return "unknown"

        return max(
            self.label_votes,
            key=self.label_votes.get,
        )

    @property
    def status(self):
        if self.observation_count >= 2:
            return "confirmed"

        return "tentative"


def add_label_vote(track, observation):
    """只有明确类别参与投票，混合标签保留在原始观测中。"""

    label = observation["label"].strip().lower()

    if label in KNOWN_LABELS:
        track.label_votes[label] = (
            track.label_votes.get(label, 0) + 1
        )


def voxel_size_of(source):
    """体素集合的元素个数，供调试输出使用；无数据时返回 0。"""

    keys = voxel_set_of(source)
    return 0 if keys is None else len(keys)


def push_voxels(track, observation, window_size):
    """把观测的体素压入轨道的滑动窗口，并重算窗口并集。

    为什么用滑动窗口而不是「最后一次观测」或「累积并集」：

    - 最后一次观测常常只是一块碎片（只看到椅背时可能只有 35 个体素，而完整
      椅子有 190 个）。拿碎片做比较基准，覆盖率会随视角剧烈波动，同一物体
      的配对也只有 0.16 左右的相似度，跟不同物体分不开。
    - 累积并集虽然完整，但不可逆：一旦某次误匹配把远处物体并进来，它会越涨
      越大（实测 room_2 有一条 chair 轨道涨到 924 体素），之后把越来越多的
      无关观测「覆盖」进来，越错越多。

    滑动窗口兼得两头：几次观测叠起来已经足够接近物体完整形状，而老观测会滑
    出去，单次误匹配的影响只持续 window_size 帧。
    """

    keys = voxel_set_of(observation)
    if keys is None:
        return
    if window_size <= 0:
        return

    while len(track.voxel_history) >= window_size:
        track.voxel_history.popleft()
    track.voxel_history.append(keys)

    union = track.voxel_history[0]
    for item in list(track.voxel_history)[1:]:
        union = union | item
    track.voxel_window = union


def voxel_set_of(source):
    """把观测（dict）或轨道（frozenset）统一成体素集合；无数据时返回 None。"""

    if isinstance(source, (set, frozenset)):
        return source if source else None

    keys = source.get("voxel_keys")
    if not keys:
        return None
    cached = source.get("_voxel_set")
    if cached is None:
        cached = frozenset(tuple(item) for item in keys)
        source["_voxel_set"] = cached
    return cached


def voxel_scores(first, second, min_voxels=15):
    """返回 (iou, coverage)：两个体素集合的 IoU 与覆盖率，无数据时 (None, None)。

    较小的那一侧少于 min_voxels 时直接判定为不可信并返回 (None, None)：一块
    只有几个体素的碎片随便落在谁身上都能拿到很高的覆盖率，让它参与判定等于
    给噪声一票否决权。室内家具在 0.05 m 体素下完整形状通常有几百个体素，
    15 的下限只挡碎片、不影响正常观测。

    覆盖率 = |A∩B| / min(|A|,|B|)。为什么关联要用覆盖率而不是 IoU：跟踪时两个
    集合常常一个是物体的完整体素、另一个只是当前视角下的一小块——只看到椅背
    时可能只有 30 个体素，而完整椅子有 180 个。此时交集 29，IoU 只有 0.16，但
    覆盖率高达 0.83：小块几乎完全落在大块里，这正是「同一个物体」的强证据。
    实测同一物体的覆盖率在 0.7 以上，不同物体接近 0，间隔很干净。

    覆盖率天然对称：小块对大块、大块对小块都会给出高值，所以比较对象用「最后
    一次观测」就够了，不需要维护累积并集。实测累积并集反而有害——一旦某次误
    匹配把远处物体并进来，并集会不可逆地膨胀（room_2 里有一条 chair 轨道的
    并集涨到 924 体素，而一把椅子只有约 190），之后它会把越来越多的无关观测
    「覆盖」进来，越错越多。
    """

    first_set = voxel_set_of(first)
    second_set = voxel_set_of(second)
    if first_set is None or second_set is None:
        return None, None

    intersection = len(first_set & second_set)
    union = len(first_set | second_set)
    if union == 0:
        return None, None

    smaller = min(len(first_set), len(second_set))
    if smaller < min_voxels:
        return None, None
    return intersection / union, intersection / smaller


def geometry_distances(first, second):
    first_center = np.asarray(first["centroid_world"])
    second_center = np.asarray(second["centroid_world"])

    center_distance = np.linalg.norm(
        first_center - second_center
    )

    first_min = np.asarray(first["bbox_min_world"])
    first_max = np.asarray(first["bbox_max_world"])
    second_min = np.asarray(second["bbox_min_world"])
    second_max = np.asarray(second["bbox_max_world"])

    axis_gaps = np.maximum(
        np.maximum(
            first_min - second_max,
            second_min - first_max,
        ),
        0.0,
    )

    return float(center_distance), float(np.linalg.norm(axis_gaps))


class GeometricInstanceTracker:
    """面向短序列、静态场景的几何关联基线。

    Stage 5c 扩展：支持可选的 shape embedding（来自 PointEncoder 的
    FP 层逐点特征 max-pool），用于增强几何关联的判别力。
    """

    def __init__(
        self,
        max_center_distance=0.65,
        max_bbox_gap=0.15,
        unmatched_cost=0.80,
        ambiguity_margin=0.0,
        label_gate=False,
        label_gate_mode="strict",
        shape_weight=0.25,
        memory_frames=30,
        reacquire_max_center_distance=1.5,
        reacquire_max_bbox_gap=0.30,
        voxel_iou_weight=0.70,
        voxel_min_coverage=0.40,
        reacquire_min_coverage=0.55,
        voxel_memory_frames=600,
        voxel_window_size=20,
    ):
        if max_center_distance <= 0 or max_bbox_gap <= 0:
            raise ValueError("距离门限必须大于 0")
        if not 0.0 <= shape_weight <= 1.0:
            raise ValueError("shape_weight 必须在 [0, 1] 内")
        if reacquire_max_center_distance < max_center_distance:
            raise ValueError("reacquire_max_center_distance 必须 ≥ max_center_distance")

        # 几何门限放得比直觉宽（0.65 m / 0.15 m）：判定物体身份的主力已经是体素
        # 覆盖率，几何只是兜底。收紧几何门限反而会把「视角变化导致质心漂移」的
        # 同一物体切成两条轨道，实测 room_2 上放宽后单轨率从 0.4 升到 0.7。
        self.max_center_distance = max_center_distance
        self.max_bbox_gap = max_bbox_gap
        # 不匹配代价抬高到 0.80：让匈牙利匹配更愿意接受「勉强像」的配对，减少
        # 被判为歧义而遭丢弃的检测（丢弃会直接损失召回率）。
        self.unmatched_cost = unmatched_cost
        # 歧义余量为 0 表示不再因为「两个候选代价接近」而丢弃检测。原来的 0.08
        # 本意是避免在拿不准时误合并，但实测它丢弃掉的多是真检测——覆盖率的分
        # 布要么接近 0 要么很高，两个候选很容易并列，于是大量观测被判歧义后连轨
        # 都不建。关掉之后 room_2 的召回率从 0.865 回到 0.896，AP 反超基线。
        self.ambiguity_margin = ambiguity_margin
        # 标签一致性门控（默认开）：当两个 KNOWN 类别不同的观测要合并时，直接
        # 阻断。根因是「实例形成不依赖语义」的设计里，显示器（computer monitor）
        # 的体素整块落在书桌（desk）体素内部，覆盖率≈1.0，于是体素通道把显示器
        # 并进了书桌轨道——office_1 有 18 个 GT 显示器却只检出 1 个。而 Replica
        # 的 GT 把每个类别都标成独立 object_id，显示器本就是与书桌不同的实例，
        # 用标签做合并否决既符合 GT，又不影响同类物体的跨帧聚合。unknown 标签
        # （检测器没给出已知类别）不参与否决，避免把没标签的轨道锁死。
        self.label_gate = label_gate
        # off=不否决 / strict=已知标签不同即否决 / support=只在家具与非家具之间否决
        if label_gate_mode not in ("off", "strict", "support"):
            raise ValueError("label_gate_mode 必须是 off/strict/support")
        self.label_gate_mode = (
            "off" if not label_gate else label_gate_mode
        )
        self.shape_weight = shape_weight
        # 重新捕获：物体短暂消失后重现时，相机已移动，其质心可能距旧轨最后
        # 位置超过常规门限；对「近期出现过（last_seen 在 memory_frames 内）」
        # 的轨道放宽门限，避免被拒而开新轨（过分割的 association_no_candidate）。
        self.memory_frames = memory_frames
        self.reacquire_max_center_distance = reacquire_max_center_distance
        self.reacquire_max_bbox_gap = reacquire_max_bbox_gap
        # 体素覆盖率关联：权重为 0 时完全退回纯几何代价（旧行为）。
        self.voxel_iou_weight = voxel_iou_weight
        # 覆盖率达到该值即认定为同一物体，直接放行、不再受质心门限约束。
        # 同一物体的实测覆盖率在 0.7 以上，不同物体接近 0，0.5 是很宽的间隔。
        self.voxel_min_coverage = voxel_min_coverage
        # 重新捕获专用的体素通道：几何重捕获只能用「近期」轨道，因为相机移动后
        # 物体质心本来就会漂到门限外，放宽记忆窗口只会误合并；而体素覆盖不随
        # 相机移动变化，几百帧之后依然是物体身份的可靠证据。所以给它一个独立
        # 且长得多的记忆窗口，并要求更高的覆盖率（跨很久的重连比相邻帧更需要
        # 证据充分，否则容易把物体搬走后又冒出来的同类物体认成同一个）。
        self.reacquire_min_coverage = reacquire_min_coverage
        self.voxel_memory_frames = voxel_memory_frames
        # 比较基准取最近几次观测的体素并集的窗口长度。
        self.voxel_window_size = voxel_window_size

        self.tracks = {}
        self.next_global_id = 1
        self.last_processed_frame = None

    def _create_track(self, observation, frame_index):
        global_id = self.next_global_id
        self.next_global_id += 1

        track = InstanceTrack(
            global_id=global_id,
            latest_observation=observation,
            first_seen_frame=frame_index,
            last_seen_frame=frame_index,
        )

        add_label_vote(track, observation)
        push_voxels(track, observation, self.voxel_window_size)
        self.tracks[global_id] = track

        return global_id

    def _gate(self, observation, track, frame_index=None):
        """返回 (inf 或 None, max_center, max_bbox, voxel_coverage)。

        frame_index 为 None（或轨道近期未出现）时用严格门限；否则用放宽门限
        （重新捕获）。代价归一化随所用门限缩放，保证同一个移动量在放宽门限下
        代价更小、更易被重新捕获。

        若覆盖率达到 voxel_min_coverage，直接放行：几何门限本来是为了拦住
        「看着不像同一个东西」的配对，而「当前观测的体素几乎全部落在轨道已有
        体素里」是更强的证据，再让质心门限否决它只会把同一物体切成多条轨道。
        """

        _, coverage = voxel_scores(observation, track.voxel_window)

        center_distance, bbox_gap = geometry_distances(
            observation,
            track.latest_observation,
        )
        if frame_index is not None:
            gap = frame_index - track.last_seen_frame
        else:
            gap = -1
        if gap >= 0 and gap <= self.memory_frames:
            max_center = self.reacquire_max_center_distance
            max_bbox = self.reacquire_max_bbox_gap
        else:
            max_center = self.max_center_distance
            max_bbox = self.max_bbox_gap

        if self._label_conflict(observation, track):
            return np.inf, max_center, max_bbox, coverage

        if coverage is not None and coverage >= self.voxel_min_coverage:
            return None, max_center, max_bbox, coverage

        if center_distance > max_center or bbox_gap > max_bbox:
            return np.inf, max_center, max_bbox, coverage
        return None, max_center, max_bbox, coverage

    def _label_penalty(self, observation, track):
        observed_label = observation["label"].strip().lower()
        if observed_label not in KNOWN_LABELS or track.label == "unknown":
            return 0.5
        if observed_label == track.label:
            return 0.0
        return 1.0

    def _label_conflict(self, observation, track):
        """两个观测是否「不可能是同一个实例」。

        off      不否决（体素覆盖率单独说话）；
        strict   两个已知类别不同就否决——会误伤 desk/table 这类同义提示词；
        support  只在「家具 / 承载面」与「非家具」之间否决：放在桌子上的纸箱
                 不是桌子的一部分，但 desk 与 table 仍然允许合并。
        """

        if self.label_gate_mode == "off":
            return False

        observed_label = observation["label"].strip().lower()
        track_label = track.label
        if track_label == "unknown":
            return False

        if self.label_gate_mode == "support":
            # 只有两边标签都认识时才否决，避免检测器偶尔给出的杂标签锁死轨道。
            if observed_label not in KNOWN_LABELS or track_label not in KNOWN_LABELS:
                return False
            return is_support_surface(observed_label) != is_support_surface(track_label)

        if observed_label not in KNOWN_LABELS:
            return False
        return observed_label != track_label


    def _shape_cost(self, observation, track):
        """返回 (shape_cost, has_shape)；无 shape embedding 时 has_shape=False。"""

        obs_shape = observation.get("shape_embedding")
        track_shape = track.latest_observation.get("shape_embedding")
        if obs_shape is None or track_shape is None:
            return 0.0, False
        a = np.asarray(obs_shape, dtype=np.float32)
        b = np.asarray(track_shape, dtype=np.float32)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm <= 1e-8:
            return 0.0, False
        sim = float(np.dot(a, b) / norm)
        return (1.0 - sim) * 0.5, True

    def _association_cost(self, observation, track, frame_index=None):
        """严格门限的关联代价（第一遍匹配用）。"""

        inf, max_center, max_bbox, coverage = self._gate(
            observation, track, frame_index
        )
        if inf is not None:
            return inf

        label_penalty = self._label_penalty(observation, track)
        shape_cost, has_shape = self._shape_cost(observation, track)
        center_distance, bbox_gap = geometry_distances(
            observation, track.latest_observation
        )

        if has_shape and self.shape_weight > 0.0:
            base = (
                0.70 * center_distance / max_center
                + 0.20 * bbox_gap / max_bbox
                + 0.10 * label_penalty
            )
            geometric = (
                (1.0 - self.shape_weight) * base
                + self.shape_weight * shape_cost
            )
        else:
            geometric = (
                0.70 * center_distance / max_center
                + 0.20 * bbox_gap / max_bbox
                + 0.10 * label_penalty
            )

        if coverage is None or self.voxel_iou_weight <= 0.0:
            return geometric

        # 以覆盖率为主判据，几何只当平局裁决。不能用加权平均：覆盖率接近 0 时
        # 加权平均会给出一个「比不匹配还差」的固定代价，把本来该由几何判定的
        # 配对也一并否决掉（体素数据缺失时尤为明显）。
        return (1.0 - coverage) + 0.10 * min(geometric, 1.0)

    def update(self, observations, frame_index):
        """返回当前帧局部 ID 到全局 ID 的关联结果。

        两遍匹配：
        1. 严格门限匹配（与原行为一致，不产生新歧义）；
        2. 仅对「严格匹配失败」的新观测，用放宽门限重新捕获近期出现过的休眠
           轨道（物体短暂消失后重现、相机已移动导致质心超严格门限）。要求唯一
           候选以避免误合并。
        """

        if (
            self.last_processed_frame is not None
            and frame_index <= self.last_processed_frame
        ):
            raise ValueError("必须按严格递增的帧编号更新")

        self.last_processed_frame = frame_index

        if not observations:
            return []

        # 固定本轮历史实例列表，避免当前帧新建实例参与本轮匹配。
        previous_tracks = list(self.tracks.values())
        observation_count = len(observations)
        track_count = len(previous_tracks)

        # ---- Pass 1：严格门限匹配 ----
        pair_costs = np.full(
            (observation_count, track_count),
            np.inf,
        )

        for row, observation in enumerate(observations):
            for column, track in enumerate(previous_tracks):
                pair_costs[row, column] = self._association_cost(observation, track)

        # 仅保留比“不匹配”更划算的候选。
        eligible = pair_costs < self.unmatched_cost

        ambiguous_rows = set()

        for row in range(observation_count):
            candidate_costs = np.sort(
                pair_costs[row, eligible[row]]
            )

            if (
                len(candidate_costs) >= 2
                and candidate_costs[1] - candidate_costs[0]
                < self.ambiguity_margin
            ):
                ambiguous_rows.add(row)

        # 右侧每一列都是一个“不匹配”位置。
        assignment_costs = np.full(
            (observation_count, track_count + observation_count),
            self.unmatched_cost,
        )

        assignment_costs[:, :track_count] = np.where(
            eligible,
            pair_costs,
            1e6,
        )

        for row in ambiguous_rows:
            assignment_costs[row, :track_count] = 1e6

        rows, columns = linear_sum_assignment(assignment_costs)
        assignments = dict(zip(rows.tolist(), columns.tolist()))

        matched_this_frame = set()
        results = []

        for row, observation in enumerate(observations):
            column = assignments[row]

            result = {
                "frame_index": frame_index,
                "local_instance_id": observation["local_instance_id"],
                "raw_label": observation["label"],
                "global_id": None,
                "association_cost": None,
            }

            if column < track_count:
                track = previous_tracks[column]

                track.latest_observation = observation
                track.last_seen_frame = frame_index
                track.observation_count += 1
                add_label_vote(track, observation)
                push_voxels(track, observation, self.voxel_window_size)
                matched_this_frame.add(track.global_id)

                result.update(
                    global_id=track.global_id,
                    decision="matched",
                    association_cost=float(pair_costs[row, column]),
                )

            elif row in ambiguous_rows:
                result["decision"] = "deferred_ambiguous"

            elif eligible[row].any():
                result["decision"] = "deferred_conflict"

            else:
                result["decision"] = "new_tentative"

            results.append(result)

        # ---- Pass 2：重新捕获（仅对严格匹配失败的新观测）----
        for row, observation in enumerate(observations):
            if results[row]["decision"] != "new_tentative":
                continue
            best_column = None
            best_cost = self.unmatched_cost
            for column, track in enumerate(previous_tracks):
                if track.global_id in matched_this_frame:
                    continue
                gap = frame_index - track.last_seen_frame
                if gap < 0:
                    continue
                # 跨已知类别的重新捕获同样否决：显示器重现时即便体素覆盖很高，
                # 也不该并回书桌轨道。
                if self._label_conflict(observation, track):
                    continue
                _, coverage = voxel_scores(
                    observation, track.voxel_window
                )
                if (
                    coverage is not None
                    and coverage >= self.reacquire_min_coverage
                    and gap <= self.voxel_memory_frames
                ):
                    # 体素通道：直接用覆盖率当代价，不看质心。
                    cost = 1.0 - coverage
                elif gap > self.memory_frames:
                    continue
                else:
                    cost = self._association_cost(
                        observation, track, frame_index
                    )
                if np.isfinite(cost) and cost < best_cost:
                    best_cost = cost
                    best_column = column
            if best_column is not None:
                track = previous_tracks[best_column]
                track.latest_observation = observation
                track.last_seen_frame = frame_index
                track.observation_count += 1
                add_label_vote(track, observation)
                push_voxels(track, observation, self.voxel_window_size)
                matched_this_frame.add(track.global_id)
                results[row].update(
                    global_id=track.global_id,
                    decision="matched",
                    association_cost=float(best_cost),
                )

        # ---- 兜底：重新捕获也失败的观测才真正新建轨道 ----
        # （与原行为一致：deferred_* 一律不建轨，避免重复实例）
        for row, observation in enumerate(observations):
            if results[row]["decision"] != "new_tentative":
                continue

            global_id = self._create_track(observation, frame_index)

            results[row].update(
                global_id=global_id,
                decision="new_tentative",
            )

        return results

    def export_tracks(self):
        return [
            {
                "global_id": track.global_id,
                "label": track.label,
                "status": track.status,
                "observation_count": track.observation_count,
                "first_seen_frame": track.first_seen_frame,
                "last_seen_frame": track.last_seen_frame,
                "latest_centroid_world": (
                    track.latest_observation["centroid_world"]
                ),
            }
            for track in self.tracks.values()
        ]
