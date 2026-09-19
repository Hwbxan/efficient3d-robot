"""静态场景关联 V2：只对无几何候选的观测增加历史表面匹配。

不修改原 GeometricInstanceTracker。表面缓存只在整帧分配结束后更新，
因此不会使用当前帧其他观测或未来帧的信息来证明本帧匹配。
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from src.mapping.geometric_instance_tracker import (
    KNOWN_LABELS,
    GeometricInstanceTracker,
    geometry_distances,
)


def downsample_surface(points, voxel_size, max_points):
    """每个占用体素保留均值点；超限后确定性采样，限制参考表面大小。"""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("表面点云必须是非空 N×3 数组")
    if not np.isfinite(points).all():
        raise ValueError("表面点云含 NaN 或无穷值")
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    representatives = sums / counts[:, None]
    if len(representatives) > max_points:
        indices = np.linspace(0, len(representatives) - 1, max_points, dtype=int)
        representatives = representatives[indices]
    return representatives


@dataclass
class SurfaceReference:
    """用于关联的有界参考点集，不是用于导出的完整融合地图。"""

    points: np.ndarray
    last_frame: int
    tree: object = None

    def nearest_distances(self, points):
        if self.tree is None:
            self.tree = cKDTree(self.points)
        distances, _ = self.tree.query(points, k=1)
        return distances


class SurfaceInstanceTracker(GeometricInstanceTracker):
    def __init__(
        self,
        point_loader,
        surface_voxel_size=0.02,
        surface_distance=0.04,
        minimum_coverage=0.80,
        minimum_surface_points=80,
        max_fallback_center_distance=1.50,
        max_reference_points=60000,
        max_query_points=12000,
        **geometric_parameters,
    ):
        super().__init__(**geometric_parameters)
        numeric = [surface_voxel_size, surface_distance, max_fallback_center_distance]
        if not all(np.isfinite(value) and value > 0 for value in numeric):
            raise ValueError("表面距离、体素大小及补充搜索距离必须为有限正数")
        if not 0 < minimum_coverage <= 1:
            raise ValueError("minimum_coverage 必须位于 (0, 1]")
        if minimum_surface_points < 3 or min(max_reference_points, max_query_points) < minimum_surface_points:
            raise ValueError("参考点及查询点上限不得低于最低表面点数")
        self.point_loader = point_loader
        self.surface_voxel_size = surface_voxel_size
        self.surface_distance = surface_distance
        self.minimum_coverage = minimum_coverage
        self.minimum_surface_points = minimum_surface_points
        self.max_fallback_center_distance = max_fallback_center_distance
        self.max_reference_points = max_reference_points
        self.max_query_points = max_query_points
        self.references = {}
        self._extra_costs = {}

    def surface_parameters(self):
        names = [
            "surface_voxel_size", "surface_distance", "minimum_coverage",
            "minimum_surface_points", "max_fallback_center_distance",
            "max_reference_points", "max_query_points",
        ]
        return {name: getattr(self, name) for name in names}

    def _association_cost(self, observation, track):
        key = (observation["local_instance_id"], track.global_id)
        if key in self._extra_costs:
            return self._extra_costs[key]
        return super()._association_cost(observation, track)

    def _surface_candidate(self, observation, query, track, frame_index, reserved):
        center_distance, latest_bbox_gap = geometry_distances(observation, track.latest_observation)
        detail = {
            "global_id": track.global_id,
            "track_label": track.label,
            "track_last_seen_frame": track.last_seen_frame,
            "center_distance_m": center_distance,
            "latest_bbox_gap_m": latest_bbox_gap,
            "reserved_by_geometry": track.global_id in reserved,
            "surface_passed": False,
        }
        label = observation["label"].strip().lower()
        if label not in KNOWN_LABELS or label != track.label:
            detail["reason"] = "requires_exact_known_label"
            return detail
        if track.status != "confirmed":
            detail["reason"] = "history_not_confirmed"
            return detail
        if center_distance > self.max_fallback_center_distance:
            detail["reason"] = "outside_fallback_search_radius"
            return detail

        reference = self.references[track.global_id]
        if reference.last_frame >= frame_index:
            raise RuntimeError("参考表面必须严格来自过去帧")
        detail["reference_last_frame"] = reference.last_frame
        detail["query_point_count"] = len(query)
        detail["reference_point_count"] = len(reference.points)
        if min(len(query), len(reference.points)) < self.minimum_surface_points:
            detail["reason"] = "insufficient_surface_points"
            return detail

        # 粗筛使用历史参考表面的包围盒，不把 bbox 相交当作匹配证据。
        gaps = np.maximum(np.maximum(
            query.min(axis=0) - reference.points.max(axis=0),
            reference.points.min(axis=0) - query.max(axis=0),
        ), 0.0)
        reference_gap = float(np.linalg.norm(gaps))
        detail["reference_bbox_gap_m"] = reference_gap
        if reference_gap > self.surface_distance:
            detail["reason"] = "reference_surface_too_far"
            return detail

        distances = reference.nearest_distances(query)
        inliers = distances <= self.surface_distance
        coverage = float(np.mean(inliers))
        mean_inlier_distance = float(np.mean(distances[inliers])) if inliers.any() else None
        detail.update(
            coverage=coverage,
            inlier_point_count=int(inliers.sum()),
            mean_inlier_distance_m=mean_inlier_distance,
        )
        if coverage < self.minimum_coverage:
            detail["reason"] = "insufficient_surface_coverage"
            return detail

        # 补充分支自己的代价；不改变正常几何分支的代价或筛选条件。
        cost = (
            0.40 * (1.0 - coverage)
            + 0.15 * mean_inlier_distance / self.surface_distance
            + 0.10 * center_distance / self.max_fallback_center_distance
        )
        detail["surface_cost"] = float(cost)
        detail["surface_passed"] = cost < self.unmatched_cost
        detail["reason"] = "surface_supported" if detail["surface_passed"] else "surface_cost_too_high"
        return detail

    def update(self, observations, frame_index):
        if self.last_processed_frame is not None and frame_index <= self.last_processed_frame:
            raise ValueError("必须按严格递增的帧编号更新")
        ids = [item["local_instance_id"] for item in observations]
        if len(ids) != len(set(ids)):
            raise ValueError("同帧局部实例 ID 不得重复")
        if not observations:
            return super().update(observations, frame_index)
        for observation in observations:
            geometry = [np.asarray(observation[name], dtype=float) for name in (
                "centroid_world", "bbox_min_world", "bbox_max_world",
            )]
            if any(value.shape != (3,) or not np.isfinite(value).all() for value in geometry):
                raise ValueError("观测的中心及包围盒必须是有限三维坐标")
            if np.any(geometry[1] > geometry[2]):
                raise ValueError("包围盒最小值不能超过最大值")

        # 先读完本帧点云；读取失败时不推进关联器状态。
        surfaces = {
            item["local_instance_id"]: downsample_surface(
                self.point_loader(item), self.surface_voxel_size, self.max_reference_points,
            )
            for item in observations
        }
        previous_tracks = list(self.tracks.values())
        normal_candidates = {}
        reserved = set()
        baseline_cost = super()._association_cost
        for observation in observations:
            eligible_ids = {
                track.global_id for track in previous_tracks
                if baseline_cost(observation, track) < self.unmatched_cost
            }
            normal_candidates[observation["local_instance_id"]] = eligible_ids
            reserved.update(eligible_ids)

        diagnostics = {}
        blocked = set()
        extra_costs = {}
        for observation in observations:
            local_id = observation["local_instance_id"]
            if normal_candidates[local_id]:
                continue
            query = surfaces[local_id]
            if len(query) > self.max_query_points:
                query = query[np.linspace(0, len(query) - 1, self.max_query_points, dtype=int)]
            candidates = [
                self._surface_candidate(observation, query, track, frame_index, reserved)
                for track in previous_tracks
            ]
            diagnostics[local_id] = candidates
            supported = [item for item in candidates if item["surface_passed"]]

            # 不让补充分支抢走正常分支的候选；有强重合时暂缓，而非再建重复 ID。
            if any(item["reserved_by_geometry"] for item in supported):
                blocked.add(local_id)
                continue
            for candidate in supported:
                extra_costs[(local_id, candidate["global_id"])] = candidate["surface_cost"]

        active = [item for item in observations if item["local_instance_id"] not in blocked]
        self._extra_costs = extra_costs
        try:
            results = super().update(active, frame_index)
        finally:
            self._extra_costs = {}
        by_local_id = {result["local_instance_id"]: result for result in results}
        ordered_results = []

        for observation in observations:
            local_id = observation["local_instance_id"]
            if local_id in blocked:
                result = {
                    "frame_index": frame_index, "local_instance_id": local_id,
                    "raw_label": observation["label"], "global_id": None,
                    "association_cost": None, "decision": "deferred_conflict",
                    "association_source": "surface_reserved_conflict",
                }
            else:
                result = by_local_id[local_id]
                key = (local_id, result["global_id"])
                if result["decision"] == "matched":
                    result["association_source"] = "surface" if key in extra_costs else "geometry"
                elif any(key[0] == local_id for key in extra_costs):
                    result["association_source"] = "surface_unassigned"
                else:
                    result["association_source"] = "geometry"
            result["surface_candidates"] = diagnostics.get(local_id, [])
            ordered_results.append(result)

        # 到此本帧决策全部完成；只有已分配观测才能进入下一帧的参考表面。
        for result in ordered_results:
            global_id = result["global_id"]
            if global_id is None:
                continue
            points = surfaces[result["local_instance_id"]]
            if global_id in self.references:
                points = downsample_surface(
                    np.concatenate([self.references[global_id].points, points]),
                    self.surface_voxel_size, self.max_reference_points,
                )
            self.references[global_id] = SurfaceReference(points=points, last_frame=frame_index)
        return ordered_results
