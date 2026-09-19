from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


KNOWN_LABELS = {
    "computer monitor",
    "chair",
    "desk",
    "trash can",
    "door",
    "sofa",
}


@dataclass
class InstanceTrack:
    """跨帧维护的实例记录。"""

    global_id: int
    latest_observation: dict
    first_seen_frame: int
    last_seen_frame: int
    observation_count: int = 1
    label_votes: dict = field(default_factory=dict)

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
        max_center_distance=0.50,
        max_bbox_gap=0.10,
        unmatched_cost=0.65,
        ambiguity_margin=0.08,
        shape_weight=0.25,
        memory_frames=30,
        reacquire_max_center_distance=1.5,
        reacquire_max_bbox_gap=0.30,
    ):
        if max_center_distance <= 0 or max_bbox_gap <= 0:
            raise ValueError("距离门限必须大于 0")
        if not 0.0 <= shape_weight <= 1.0:
            raise ValueError("shape_weight 必须在 [0, 1] 内")
        if reacquire_max_center_distance < max_center_distance:
            raise ValueError("reacquire_max_center_distance 必须 ≥ max_center_distance")

        self.max_center_distance = max_center_distance
        self.max_bbox_gap = max_bbox_gap
        self.unmatched_cost = unmatched_cost
        self.ambiguity_margin = ambiguity_margin
        self.shape_weight = shape_weight
        # 重新捕获：物体短暂消失后重现时，相机已移动，其质心可能距旧轨最后
        # 位置超过常规门限；对「近期出现过（last_seen 在 memory_frames 内）」
        # 的轨道放宽门限，避免被拒而开新轨（过分割的 association_no_candidate）。
        self.memory_frames = memory_frames
        self.reacquire_max_center_distance = reacquire_max_center_distance
        self.reacquire_max_bbox_gap = reacquire_max_bbox_gap

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
        self.tracks[global_id] = track

        return global_id

    def _gate(self, observation, track, frame_index=None):
        """返回 (cost, max_center, max_bbox)。

        frame_index 为 None（或轨道近期未出现）时用严格门限；否则用放宽门限
        （重新捕获）。代价归一化随所用门限缩放，保证同一个移动量在放宽门限下
        代价更小、更易被重新捕获。
        """

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

        if center_distance > max_center or bbox_gap > max_bbox:
            return np.inf, max_center, max_bbox
        return None, max_center, max_bbox

    def _label_penalty(self, observation, track):
        observed_label = observation["label"].strip().lower()
        if observed_label not in KNOWN_LABELS or track.label == "unknown":
            return 0.5
        if observed_label == track.label:
            return 0.0
        return 1.0

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

        inf, max_center, max_bbox = self._gate(observation, track, frame_index)
        if inf is not None:
            return inf

        label_penalty = self._label_penalty(observation, track)
        shape_cost, has_shape = self._shape_cost(observation, track)

        if has_shape and self.shape_weight > 0.0:
            base = (
                0.70 * geometry_distances(observation, track.latest_observation)[0] / max_center
                + 0.20 * geometry_distances(observation, track.latest_observation)[1] / max_bbox
                + 0.10 * label_penalty
            )
            return (1.0 - self.shape_weight) * base + self.shape_weight * shape_cost

        center_distance, bbox_gap = geometry_distances(observation, track.latest_observation)
        return (
            0.70 * center_distance / max_center
            + 0.20 * bbox_gap / max_bbox
            + 0.10 * label_penalty
        )

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
                if gap < 0 or gap > self.memory_frames:
                    continue
                cost = self._association_cost(observation, track, frame_index)
                if np.isfinite(cost) and cost < best_cost:
                    best_cost = cost
                    best_column = column
            if best_column is not None:
                track = previous_tracks[best_column]
                track.latest_observation = observation
                track.last_seen_frame = frame_index
                track.observation_count += 1
                add_label_vote(track, observation)
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