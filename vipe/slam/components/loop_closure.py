# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Appearance-retrieved, geometry-verified loop closure for single-view ViPE.

The stock DROID backend proposes long-range factors from the *current* pose and
depth estimate.  A genuine revisit can therefore be missed once drift makes the
two observations geometrically distant.  This module adds an independent place
recognition path, verifies candidates with RGB features and keyframe depth, and
uses the resulting SE(3) constraints to initialize a final dense DROID BA.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from vipe.ext.lietorch import SE3
from vipe.utils.geometry import se3_matrix_to_se3

from .buffer import GraphBuffer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopClosureOptions:
    min_frame_gap: int = 300
    retrieval_top_k: int = 6
    similarity_threshold: float = 0.52
    spatial_similarity_threshold: float = 0.35
    spatial_search_radius: float = 2.0
    max_candidates: int = 120
    candidate_nms: int = 2
    max_features: int = 2048
    ratio_test: float = 0.78
    min_matches: int = 35
    min_inliers: int = 24
    min_inlier_ratio: float = 0.28
    max_reprojection_error: float = 2.5
    max_cycle_rotation_deg: float = 5.0
    max_cycle_translation: float = 0.30
    max_pose_correction_rotation_deg: float = 35.0
    max_pose_correction_translation: float = 3.0
    cluster_radius: int = 8
    min_cluster_support: int = 2
    max_loop_edges: int = 24
    pose_graph_max_nfev: int = 40

    @classmethod
    def from_config(cls, config: Any) -> "LoopClosureOptions":
        values = {}
        for field in cls.__dataclass_fields__:
            if field == "enabled":
                continue
            if hasattr(config, field):
                values[field] = getattr(config, field)
        return cls(**values)


@dataclass
class LoopConstraint:
    source: int
    target: int
    source_frame: int
    target_frame: int
    similarity: float
    matches: int
    inliers: int
    inlier_ratio: float
    median_reprojection_error: float
    cycle_rotation_deg: float
    cycle_translation: float
    current_rotation_error_deg: float
    current_translation_error: float
    target_from_source: np.ndarray
    cluster_support: int = 1

    @property
    def score(self) -> float:
        return float(self.similarity * self.inlier_ratio * min(self.inliers / 80.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_from_source"] = self.target_from_source.tolist()
        result["score"] = self.score
        return result


def _rotation_degrees(matrix: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(matrix[:3, :3]).magnitude()))


def _retrieval_descriptors(buffer: GraphBuffer, batch_size: int = 32) -> np.ndarray:
    """Pool the already-computed DROID correspondence features without a new model."""

    descriptors = []
    for start in range(0, buffer.n_frames, batch_size):
        features = buffer.fmaps[start : start + batch_size, 0].float()
        features = F.normalize(features, dim=1)
        # A small spatial pyramid is substantially more discriminative than a
        # single global mean, while remaining almost free compared with SLAM.
        global_mean = features.mean(dim=(2, 3))
        global_rms = features.square().mean(dim=(2, 3)).sqrt()
        spatial = F.adaptive_avg_pool2d(features, (2, 2)).flatten(1)
        descriptors.append(
            F.normalize(torch.cat((global_mean, global_rms, spatial), dim=1), dim=1).cpu()
        )
    return torch.cat(descriptors).numpy().astype(np.float32, copy=False)


def _w2c_and_centers(buffer: GraphBuffer) -> tuple[np.ndarray, np.ndarray]:
    w2c = SE3(buffer.poses[: buffer.n_frames]).matrix().detach().cpu().numpy().astype(np.float64)
    c2w = np.linalg.inv(w2c)
    return w2c, c2w[:, :3, 3]


def _candidate_pairs(
    buffer: GraphBuffer,
    options: LoopClosureOptions,
) -> tuple[list[tuple[int, int, float, float]], dict[str, Any]]:
    descriptors = _retrieval_descriptors(buffer)
    similarity = descriptors @ descriptors.T
    timestamps = buffer.tstamp[: buffer.n_frames].detach().cpu().numpy().astype(np.int64)
    _, centers = _w2c_and_centers(buffer)
    candidates: list[tuple[int, int, float, float]] = []

    for target in range(buffer.n_frames):
        earlier = np.arange(target, dtype=np.int64)
        if len(earlier) == 0:
            continue
        temporal = timestamps[target] - timestamps[earlier] >= options.min_frame_gap
        if not np.any(temporal):
            continue
        distances = np.linalg.norm(centers[earlier] - centers[target], axis=1)
        sims = similarity[target, earlier]
        # Deliberately do not gate appearance retrieval with the current pose.
        # That would reproduce the stock backend's failure mode: once drift has
        # separated a revisit, the correct loop can no longer be proposed.
        appearance = sims >= options.similarity_threshold
        spatial = (distances <= options.spatial_search_radius) & (
            sims >= options.spatial_similarity_threshold
        )
        valid = temporal & (appearance | spatial)
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            continue
        order = valid_indices[np.argsort(sims[valid_indices])[::-1][: options.retrieval_top_k]]
        candidates.extend(
            (int(earlier[index]), target, float(sims[index]), float(distances[index]))
            for index in order
        )

    # Pair-space NMS avoids spending SIFT/PnP time on many copies of the same
    # revisit while retaining enough neighboring pairs for sequence support.
    selected: list[tuple[int, int, float, float]] = []
    for candidate in sorted(candidates, key=lambda item: item[2], reverse=True):
        source, target = candidate[:2]
        if any(
            abs(source - old_source) <= options.candidate_nms
            and abs(target - old_target) <= options.candidate_nms
            for old_source, old_target, _, _ in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= options.max_candidates:
            break
    return selected, {
        "keyframes": int(buffer.n_frames),
        "raw_retrieval_candidates": len(candidates),
        "candidates_after_nms": len(selected),
    }


class _FeatureCache:
    def __init__(self, buffer: GraphBuffer, options: LoopClosureOptions):
        self.buffer = buffer
        self.options = options
        self.features: dict[int, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
        self.disparities: dict[int, np.ndarray] = {}
        self.sift = cv2.SIFT_create(
            nfeatures=options.max_features,
            contrastThreshold=0.015,
            edgeThreshold=12,
        )

    def image_features(self, index: int) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        if index not in self.features:
            image = self.buffer.images[index, 0].detach().float().cpu().numpy()
            image = np.clip(np.moveaxis(image, 0, -1) * 255.0, 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            self.features[index] = self.sift.detectAndCompute(gray, None)
        return self.features[index]

    def disparity(self, index: int) -> np.ndarray:
        if index not in self.disparities:
            sensor = self.buffer.disps_sens[index, 0].detach().float().cpu().numpy()
            optimized = self.buffer.disps[index, 0].detach().float().cpu().numpy()
            self.disparities[index] = np.where(sensor > 1e-5, sensor, optimized)
        return self.disparities[index]


def _mutual_ratio_matches(
    desc_a: np.ndarray | None,
    desc_b: np.ndarray | None,
    ratio: float,
) -> list[cv2.DMatch]:
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def filtered(query: np.ndarray, train: np.ndarray) -> dict[int, cv2.DMatch]:
        result = {}
        for pair in matcher.knnMatch(query, train, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                result[pair[0].queryIdx] = pair[0]
        return result

    forward = filtered(desc_a, desc_b)
    reverse = filtered(desc_b, desc_a)
    return [
        match
        for query, match in forward.items()
        if reverse.get(match.trainIdx, None) is not None
        and reverse[match.trainIdx].trainIdx == query
    ]


def _sample_depth(disparity: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # ViPE stores disparity at original-image pixel centers 3, 11, 19, ...
    x = (pixels[:, 0] - 3.0) / 8.0
    y = (pixels[:, 1] - 3.0) / 8.0
    valid = (x >= 0) & (y >= 0) & (x < disparity.shape[1] - 1) & (y < disparity.shape[0] - 1)
    x0 = np.floor(np.clip(x, 0, disparity.shape[1] - 2)).astype(np.int64)
    y0 = np.floor(np.clip(y, 0, disparity.shape[0] - 2)).astype(np.int64)
    dx, dy = x - x0, y - y0
    sampled = (
        disparity[y0, x0] * (1 - dx) * (1 - dy)
        + disparity[y0, x0 + 1] * dx * (1 - dy)
        + disparity[y0 + 1, x0] * (1 - dx) * dy
        + disparity[y0 + 1, x0 + 1] * dx * dy
    )
    depth = np.divide(1.0, sampled, out=np.zeros_like(sampled), where=sampled > 1e-5)
    valid &= np.isfinite(depth) & (depth >= 0.15) & (depth <= 20.0)
    return depth, valid


def _pnp(
    source_pixels: np.ndarray,
    target_pixels: np.ndarray,
    disparity: np.ndarray,
    K: np.ndarray,
    options: LoopClosureOptions,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    depth, valid = _sample_depth(disparity, source_pixels)
    if int(valid.sum()) < options.min_matches:
        return None
    valid_indices = np.flatnonzero(valid)
    source_pixels = source_pixels[valid]
    target_pixels = target_pixels[valid]
    depth = depth[valid]
    xyz = np.column_stack(
        (
            (source_pixels[:, 0] - K[0, 2]) / K[0, 0] * depth,
            (source_pixels[:, 1] - K[1, 2]) / K[1, 1] * depth,
            depth,
        )
    ).astype(np.float32)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            xyz,
            target_pixels.astype(np.float32),
            K,
            None,
            iterationsCount=1000,
            reprojectionError=options.max_reprojection_error,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None
    if not ok or inliers is None or len(inliers) < options.min_inliers:
        return None
    inliers = inliers[:, 0]
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            xyz[inliers],
            target_pixels[inliers].astype(np.float32),
            K,
            None,
            rvec,
            tvec,
        )
    except cv2.error:
        return None
    projected, _ = cv2.projectPoints(xyz[inliers], rvec, tvec, K, None)
    errors = np.linalg.norm(projected[:, 0] - target_pixels[inliers], axis=1)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec[:, 0]
    return transform, valid_indices[inliers], float(np.median(errors))


def _verify_candidate(
    source: int,
    target: int,
    similarity: float,
    cache: _FeatureCache,
    K: np.ndarray,
    timestamps: np.ndarray,
    w2c: np.ndarray,
    options: LoopClosureOptions,
) -> tuple[LoopConstraint | None, str]:
    kp_source, desc_source = cache.image_features(source)
    kp_target, desc_target = cache.image_features(target)
    matches = _mutual_ratio_matches(desc_source, desc_target, options.ratio_test)
    if len(matches) < options.min_matches:
        return None, "too_few_mutual_matches"
    source_pixels = np.asarray([kp_source[m.queryIdx].pt for m in matches], dtype=np.float64)
    target_pixels = np.asarray([kp_target[m.trainIdx].pt for m in matches], dtype=np.float64)

    forward = _pnp(source_pixels, target_pixels, cache.disparity(source), K, options)
    reverse = _pnp(target_pixels, source_pixels, cache.disparity(target), K, options)
    if forward is None or reverse is None:
        return None, "pnp_failed"
    target_from_source, inliers, reprojection = forward
    source_from_target, reverse_inliers, _ = reverse
    inlier_count = len(np.intersect1d(inliers, reverse_inliers, assume_unique=True))
    inlier_ratio = inlier_count / max(len(matches), 1)
    if inlier_count < options.min_inliers or inlier_ratio < options.min_inlier_ratio:
        return None, "low_inlier_ratio"

    cycle = source_from_target @ target_from_source
    cycle_rotation = _rotation_degrees(cycle)
    cycle_translation = float(np.linalg.norm(cycle[:3, 3]))
    if cycle_rotation > options.max_cycle_rotation_deg or cycle_translation > options.max_cycle_translation:
        return None, "bidirectional_inconsistency"

    current = w2c[target] @ np.linalg.inv(w2c[source])
    correction = target_from_source @ np.linalg.inv(current)
    correction_rotation = _rotation_degrees(correction)
    correction_translation = float(np.linalg.norm(correction[:3, 3]))
    if (
        correction_rotation > options.max_pose_correction_rotation_deg
        or correction_translation > options.max_pose_correction_translation
    ):
        return None, "implausible_pose_correction"

    return (
        LoopConstraint(
            source=source,
            target=target,
            source_frame=int(timestamps[source]),
            target_frame=int(timestamps[target]),
            similarity=similarity,
            matches=len(matches),
            inliers=inlier_count,
            inlier_ratio=float(inlier_ratio),
            median_reprojection_error=reprojection,
            cycle_rotation_deg=cycle_rotation,
            cycle_translation=cycle_translation,
            current_rotation_error_deg=correction_rotation,
            current_translation_error=correction_translation,
            target_from_source=target_from_source,
        ),
        "accepted",
    )


def _sequence_filter(
    constraints: list[LoopConstraint], options: LoopClosureOptions
) -> list[LoopConstraint]:
    if options.min_cluster_support <= 1:
        return constraints
    retained = []
    for constraint in constraints:
        support = sum(
            abs(constraint.source - other.source) <= options.cluster_radius
            and abs(constraint.target - other.target) <= options.cluster_radius
            for other in constraints
        )
        constraint.cluster_support = support
        if support >= options.min_cluster_support:
            retained.append(constraint)
    return sorted(retained, key=lambda item: item.score, reverse=True)[: options.max_loop_edges]


def _matrix_from_vector(vector: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    matrix[:3, 3] = vector[3:]
    return matrix


def _vector_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]))


def _optimize_pose_graph_matrices(
    original: np.ndarray,
    constraints: list[LoopConstraint],
    options: LoopClosureOptions,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize W2C keyframe matrices while preserving local odometry."""

    n = len(original)
    if n < 2:
        return original.copy(), {
            "optimizer_success": True,
            "optimizer_status": 0,
            "optimizer_message": "Only one keyframe",
            "function_evaluations": 0,
            "cost_before": 0.0,
            "cost_after": 0.0,
            "median_camera_center_correction": 0.0,
            "max_camera_center_correction": 0.0,
            "median_local_odometry_change": 0.0,
            "max_local_odometry_change": 0.0,
        }
    initial = np.concatenate([_vector_from_matrix(original[index]) for index in range(1, n)])
    edges: list[tuple[int, int, np.ndarray, float, str]] = []
    for step, weight in ((1, 1.0), (2, 0.5), (4, 0.25)):
        for source in range(0, n - step):
            target = source + step
            measurement = original[target] @ np.linalg.inv(original[source])
            edges.append((source, target, measurement, weight, "odometry"))
    for constraint in constraints:
        confidence = min(constraint.inliers / 80.0, 1.0) * constraint.inlier_ratio
        edges.append(
            (
                constraint.source,
                constraint.target,
                constraint.target_from_source,
                8.0 + 12.0 * confidence,
                "loop",
            )
        )

    def unpack(vector: np.ndarray) -> list[np.ndarray]:
        return [original[0]] + [
            _matrix_from_vector(vector[6 * (index - 1) : 6 * index]) for index in range(1, n)
        ]

    def residual(vector: np.ndarray) -> np.ndarray:
        poses = unpack(vector)
        values = []
        for source, target, measurement, weight, _ in edges:
            predicted = poses[target] @ np.linalg.inv(poses[source])
            error = np.linalg.inv(measurement) @ predicted
            scale = np.sqrt(weight)
            values.extend((Rotation.from_matrix(error[:3, :3]).as_rotvec() / 0.05 * scale).tolist())
            values.extend((error[:3, 3] / 0.10 * scale).tolist())
        return np.asarray(values, dtype=np.float64)

    sparsity = lil_matrix((6 * len(edges), 6 * (n - 1)), dtype=np.int8)
    for edge_index, (source, target, _, _, _) in enumerate(edges):
        rows = slice(6 * edge_index, 6 * edge_index + 6)
        if source > 0:
            sparsity[rows, 6 * (source - 1) : 6 * source] = 1
        if target > 0:
            sparsity[rows, 6 * (target - 1) : 6 * target] = 1

    before = residual(initial)
    result = least_squares(
        residual,
        initial,
        jac_sparsity=sparsity.tocsr(),
        method="trf",
        # Loop edges have already passed strict geometric verification.  A
        # global robust loss can otherwise sacrifice one odometry edge and
        # create a pose discontinuity instead of distributing loop drift.
        loss="linear",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=options.pose_graph_max_nfev,
        verbose=0,
    )
    corrected = np.stack(unpack(result.x))
    after = residual(result.x)
    original_centers = np.linalg.inv(original)[:, :3, 3]
    corrected_centers = np.linalg.inv(corrected)[:, :3, 3]
    center_correction = np.linalg.norm(corrected_centers - original_centers, axis=1)
    local_changes = []
    for source in range(n - 1):
        old_relative = original[source + 1] @ np.linalg.inv(original[source])
        new_relative = corrected[source + 1] @ np.linalg.inv(corrected[source])
        delta = np.linalg.inv(old_relative) @ new_relative
        local_changes.append(
            np.linalg.norm(delta[:3, 3]) + 0.1 * np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
        )

    cost_before = float(0.5 * np.dot(before, before))
    cost_after = float(0.5 * np.dot(after, after))

    if (
        not result.success
        or not np.all(np.isfinite(corrected))
        or cost_after >= cost_before
        or float(np.max(center_correction)) > 5.0
        or float(np.max(local_changes, initial=0.0)) > 0.75
    ):
        raise RuntimeError("Loop pose-graph correction failed sanity checks")
    return corrected, {
        "applied": True,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "function_evaluations": int(result.nfev),
        "cost_before": cost_before,
        "cost_after": cost_after,
        "median_camera_center_correction": float(np.median(center_correction)),
        "max_camera_center_correction": float(np.max(center_correction)),
        "median_local_odometry_change": float(np.median(local_changes)),
        "max_local_odometry_change": float(np.max(local_changes, initial=0.0)),
    }


def _pose_graph_optimize(
    buffer: GraphBuffer,
    constraints: list[LoopConstraint],
    options: LoopClosureOptions,
) -> dict[str, Any]:
    original, _ = _w2c_and_centers(buffer)
    corrected, report = _optimize_pose_graph_matrices(original, constraints, options)
    corrected_se3 = se3_matrix_to_se3(corrected.astype(np.float32)).data.to(buffer.device)
    buffer.poses[: buffer.n_frames] = corrected_se3
    buffer.dirty[: buffer.n_frames] = True
    return report


@torch.no_grad()
def detect_and_correct_loops(
    buffer: GraphBuffer,
    config: Any,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    options = LoopClosureOptions.from_config(config)
    candidates, retrieval_report = _candidate_pairs(buffer, options)
    timestamps = buffer.tstamp[: buffer.n_frames].detach().cpu().numpy().astype(np.int64)
    w2c, _ = _w2c_and_centers(buffer)
    K = buffer.K[0].astype(np.float64)
    cache = _FeatureCache(buffer, options)
    verified = []
    rejections: Counter[str] = Counter()
    candidate_records = []
    for source, target, similarity, pose_distance in candidates:
        try:
            constraint, reason = _verify_candidate(
                source,
                target,
                similarity,
                cache,
                K,
                timestamps,
                w2c,
                options,
            )
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("Rejected loop candidate %d -> %d: %s", source, target, exc)
            constraint, reason = None, "verification_exception"
        rejections[reason] += 1
        record: dict[str, Any] = {
            "source_keyframe": source,
            "target_keyframe": target,
            "source_frame": int(timestamps[source]),
            "target_frame": int(timestamps[target]),
            "similarity": similarity,
            "current_camera_distance": pose_distance,
            "result": reason,
        }
        if constraint is not None:
            verified.append(constraint)
            record.update(constraint.to_dict())
        candidate_records.append(record)

    retained = _sequence_filter(verified, options)
    report: dict[str, Any] = {
        "enabled": True,
        "options": asdict(options),
        "retrieval": retrieval_report,
        "verification_counts": dict(rejections),
        "verified_before_sequence_filter": len(verified),
        "accepted_loop_edges": len(retained),
        "accepted": [constraint.to_dict() for constraint in retained],
        "candidates": candidate_records,
        "pose_graph": None,
    }
    if not retained:
        logger.warning("Loop closure found no geometrically verified long-range edge")
        return None, report

    try:
        report["pose_graph"] = _pose_graph_optimize(buffer, retained, options)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        report["pose_graph"] = {"applied": False, "error": str(exc)}
        logger.warning("Rejected loop pose-graph correction: %s", exc)
        return None, report
    edges = torch.tensor(
        [[constraint.source, constraint.target] for constraint in retained],
        dtype=torch.long,
        device=buffer.device,
    )
    logger.info(
        "Loop closure accepted %d/%d candidates; pose graph cost %.3f -> %.3f",
        len(retained),
        len(candidates),
        report["pose_graph"]["cost_before"],
        report["pose_graph"]["cost_after"],
    )
    return edges, report
