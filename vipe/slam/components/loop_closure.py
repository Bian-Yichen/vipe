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

LOOP_CLOSURE_VERSION = 2


@dataclass(frozen=True)
class LoopClosureOptions:
    # Require a genuinely long revisit.  Raw-frame and keyframe gaps are both
    # needed because a nearly stationary camera can produce very sparse
    # keyframes over hundreds of decoded frames.
    min_frame_gap: int = 900
    min_keyframe_gap: int = 20
    retrieval_top_k: int = 4
    similarity_threshold: float = 0.45
    max_candidates: int = 48
    candidate_nms: int = 4
    sequence_radius: int = 2
    min_sequence_support: int = 2
    droid_retrieval_weight: float = 0.35
    vlad_words: int = 24
    vlad_sample_descriptors: int = 12000
    vlad_kmeans_iters: int = 8
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
    max_loop_edges: int = 32
    switch_prior_weight: float = 25.0
    min_switch_weight: float = 0.25
    pose_graph_max_local_change: float = 0.35
    pose_graph_max_nfev: int = 60

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
    sequence_similarity: float
    sequence_direction: int
    matches: int
    inliers: int
    inlier_ratio: float
    essential_inliers: int
    essential_rotation_error_deg: float
    median_reprojection_error: float
    cycle_rotation_deg: float
    cycle_translation: float
    current_rotation_error_deg: float
    current_translation_error: float
    target_from_source: np.ndarray
    cluster_support: int = 1

    @property
    def score(self) -> float:
        retrieval = 0.35 * self.similarity + 0.65 * self.sequence_similarity
        return float(retrieval * self.inlier_ratio * min(self.inliers / 80.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_from_source"] = self.target_from_source.tolist()
        result["score"] = self.score
        return result


def _rotation_degrees(matrix: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(matrix[:3, :3]).magnitude()))


def _retrieval_descriptors(
    buffer: GraphBuffer,
    n_frames: int | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Pool the already-computed DROID correspondence features without a new model."""

    n_frames = int(buffer.n_frames if n_frames is None else n_frames)
    if n_frames <= 0:
        return np.empty((0, 0), dtype=np.float32)
    descriptors = []
    for start in range(0, n_frames, batch_size):
        # GraphBuffer is preallocated.  The last batch must stop at the number
        # of valid keyframes, otherwise a non-multiple-of-32 count also pools
        # unused capacity slots (for example 124 valid frames became 128).
        end = min(start + batch_size, n_frames)
        features = buffer.fmaps[start:end, 0].float()
        features = F.normalize(features, dim=1)
        # A small spatial pyramid is substantially more discriminative than a
        # single global mean, while remaining almost free compared with SLAM.
        global_mean = features.mean(dim=(2, 3))
        global_rms = features.square().mean(dim=(2, 3)).sqrt()
        spatial = F.adaptive_avg_pool2d(features, (2, 2)).flatten(1)
        descriptors.append(
            F.normalize(torch.cat((global_mean, global_rms, spatial), dim=1), dim=1).cpu()
        )
    pooled = torch.cat(descriptors)
    if len(pooled) != n_frames:
        raise RuntimeError(
            f"Expected {n_frames} valid DROID descriptors, pooled {len(pooled)}"
        )
    return pooled.numpy().astype(np.float32, copy=False)


def _w2c_and_centers(
    buffer: GraphBuffer, n_frames: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = int(buffer.n_frames if n_frames is None else n_frames)
    w2c = SE3(buffer.poses[:n_frames]).matrix().detach().cpu().numpy().astype(np.float64)
    c2w = np.linalg.inv(w2c)
    return w2c, c2w[:, :3, 3]


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
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            if descriptors is not None:
                # RootSIFT is markedly more robust to illumination changes in
                # room tours and remains compatible with Euclidean matching.
                descriptors = descriptors.astype(np.float32, copy=False)
                descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
                descriptors = np.sqrt(descriptors)
            self.features[index] = keypoints, descriptors
        return self.features[index]

    def disparity(self, index: int) -> np.ndarray:
        if index not in self.disparities:
            sensor = self.buffer.disps_sens[index, 0].detach().float().cpu().numpy()
            optimized = self.buffer.disps[index, 0].detach().float().cpu().numpy()
            self.disparities[index] = np.where(sensor > 1e-5, sensor, optimized)
        return self.disparities[index]


def _sift_vlad_descriptors(
    cache: _FeatureCache,
    n_frames: int,
    options: LoopClosureOptions,
) -> np.ndarray:
    """Build a video-specific RootSIFT-VLAD place descriptor.

    Learning the visual words from the current video is useful for repetitive
    interiors: wood, white walls and door frames are represented relative to
    the appearance distribution of this particular house.
    """

    descriptors_by_frame: list[np.ndarray | None] = []
    samples = []
    per_frame = max(32, options.vlad_sample_descriptors // max(n_frames, 1))
    for index in range(n_frames):
        _, descriptors = cache.image_features(index)
        descriptors_by_frame.append(descriptors)
        if descriptors is None or len(descriptors) == 0:
            continue
        if len(descriptors) > per_frame:
            selected = np.linspace(0, len(descriptors) - 1, per_frame, dtype=np.int64)
            descriptors = descriptors[selected]
        samples.append(descriptors)

    dimension = 128 * options.vlad_words
    if not samples:
        return np.zeros((n_frames, dimension), dtype=np.float32)
    training = np.concatenate(samples, axis=0).astype(np.float32, copy=False)
    if len(training) > options.vlad_sample_descriptors:
        selected = np.linspace(
            0,
            len(training) - 1,
            options.vlad_sample_descriptors,
            dtype=np.int64,
        )
        training = training[selected]
    words = min(options.vlad_words, len(training))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, options.vlad_kmeans_iters, 1e-4)
    _, _, codebook = cv2.kmeans(
        training,
        words,
        None,
        criteria,
        1,
        cv2.KMEANS_PP_CENTERS,
    )

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    output = np.zeros((n_frames, words * 128), dtype=np.float32)
    for index, descriptors in enumerate(descriptors_by_frame):
        if descriptors is None or len(descriptors) == 0:
            continue
        assignments = np.asarray(
            [match.trainIdx for match in matcher.match(descriptors, codebook)],
            dtype=np.int64,
        )
        residuals = np.zeros((words, 128), dtype=np.float32)
        np.add.at(residuals, assignments, descriptors - codebook[assignments])
        residuals /= np.linalg.norm(residuals, axis=1, keepdims=True) + 1e-7
        vector = residuals.reshape(-1)
        vector = np.sign(vector) * np.sqrt(np.abs(vector))
        output[index] = vector / (np.linalg.norm(vector) + 1e-7)
    return output


def _sequence_score(
    similarities: np.ndarray,
    source: int,
    target: int,
    radius: int,
) -> tuple[float, int]:
    forward = []
    reverse = []
    n_frames = len(similarities)
    for offset in range(-radius, radius + 1):
        source_index = source + offset
        forward_target = target + offset
        reverse_target = target - offset
        if 0 <= source_index < n_frames and 0 <= forward_target < n_frames:
            forward.append(float(similarities[source_index, forward_target]))
        if 0 <= source_index < n_frames and 0 <= reverse_target < n_frames:
            reverse.append(float(similarities[source_index, reverse_target]))
    forward_score = float(np.mean(forward)) if forward else -1.0
    reverse_score = float(np.mean(reverse)) if reverse else -1.0
    return (forward_score, 1) if forward_score >= reverse_score else (reverse_score, -1)


def _advanced_candidate_pairs(
    buffer: GraphBuffer,
    cache: _FeatureCache,
    options: LoopClosureOptions,
) -> tuple[list[tuple[int, int, float, float, int, float]], dict[str, Any]]:
    """Retrieve long sequence-level loop seeds, independent of current pose."""

    n_keyframes = int(buffer.n_frames)
    droid = _retrieval_descriptors(buffer, n_keyframes)
    vlad = _sift_vlad_descriptors(cache, n_keyframes, options)
    droid_similarity = droid @ droid.T
    vlad_similarity = vlad @ vlad.T
    similarities = (
        options.droid_retrieval_weight * droid_similarity
        + (1.0 - options.droid_retrieval_weight) * vlad_similarity
    )
    timestamps = buffer.tstamp[:n_keyframes].detach().cpu().numpy().astype(np.int64)
    _, centers = _w2c_and_centers(buffer, n_keyframes)
    candidates: list[tuple[int, int, float, float, int, float]] = []

    for target in range(n_keyframes):
        target_candidates = []
        for source in range(target):
            if target - source < options.min_keyframe_gap:
                continue
            if timestamps[target] - timestamps[source] < options.min_frame_gap:
                continue
            sequence_similarity, direction = _sequence_score(
                similarities,
                source,
                target,
                options.sequence_radius,
            )
            if sequence_similarity < options.similarity_threshold:
                continue
            pose_distance = float(np.linalg.norm(centers[source] - centers[target]))
            target_candidates.append(
                (
                    source,
                    target,
                    float(similarities[source, target]),
                    sequence_similarity,
                    direction,
                    pose_distance,
                )
            )
        target_candidates.sort(key=lambda item: item[3], reverse=True)
        candidates.extend(target_candidates[: options.retrieval_top_k])

    selected: list[tuple[int, int, float, float, int, float]] = []
    for candidate in sorted(candidates, key=lambda item: item[3], reverse=True):
        source, target = candidate[:2]
        if any(
            abs(source - old_source) <= options.candidate_nms
            and abs(target - old_target) <= options.candidate_nms
            for old_source, old_target, *_ in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= options.max_candidates:
            break
    droid_weight = options.droid_retrieval_weight
    return selected, {
        "version": LOOP_CLOSURE_VERSION,
        "keyframes": n_keyframes,
        "raw_long_sequence_candidates": len(candidates),
        "candidates_after_nms": len(selected),
        "retrieval_descriptor": (
            f"{droid_weight:.2f} DROID spatial pyramid + "
            f"{1.0 - droid_weight:.2f} video-specific RootSIFT-VLAD"
        ),
        "minimum_raw_frame_gap": options.min_frame_gap,
        "minimum_keyframe_gap": options.min_keyframe_gap,
    }


def _mutual_ratio_matches(
    desc_a: np.ndarray | None,
    desc_b: np.ndarray | None,
    ratio: float,
) -> list[cv2.DMatch]:
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []
    matcher = cv2.FlannBasedMatcher(
        {"algorithm": 1, "trees": 5},
        {"checks": 64},
    )

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
    sequence_similarity: float,
    sequence_direction: int,
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

    try:
        essential, essential_mask = cv2.findEssentialMat(
            source_pixels,
            target_pixels,
            K,
            method=getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
            prob=0.999,
            threshold=1.5,
        )
        if essential is None or essential_mask is None:
            return None, "essential_failed"
        essential = essential[:3]
        _, essential_rotation, _, recovered_mask = cv2.recoverPose(
            essential,
            source_pixels,
            target_pixels,
            K,
            mask=essential_mask,
        )
    except cv2.error:
        return None, "essential_failed"
    if recovered_mask is None:
        return None, "essential_failed"
    essential_inliers = int((recovered_mask > 0).sum())
    if essential_inliers < options.min_inliers:
        return None, "too_few_essential_inliers"

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

    essential_rotation_error = float(
        np.degrees(
            Rotation.from_matrix(
                essential_rotation @ target_from_source[:3, :3].T
            ).magnitude()
        )
    )
    if essential_rotation_error > options.max_cycle_rotation_deg:
        return None, "essential_pnp_rotation_inconsistency"

    cycle = source_from_target @ target_from_source
    cycle_rotation = _rotation_degrees(cycle)
    cycle_translation = float(np.linalg.norm(cycle[:3, 3]))
    if (
        cycle_rotation > options.max_cycle_rotation_deg
        or cycle_translation > options.max_cycle_translation
    ):
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
            sequence_similarity=sequence_similarity,
            sequence_direction=sequence_direction,
            matches=len(matches),
            inliers=inlier_count,
            inlier_ratio=float(inlier_ratio),
            essential_inliers=essential_inliers,
            essential_rotation_error_deg=essential_rotation_error,
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
            "applied": False,
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
    pose_size = 6 * (n - 1)
    initial_poses = np.concatenate(
        [_vector_from_matrix(original[index]) for index in range(1, n)]
    )
    odometry_edges: list[tuple[int, int, np.ndarray, float]] = []
    for step, weight in ((1, 1.0), (2, 0.5), (4, 0.25)):
        for source in range(0, n - step):
            target = source + step
            measurement = original[target] @ np.linalg.inv(original[source])
            odometry_edges.append((source, target, measurement, weight))
    loop_edges: list[tuple[int, int, np.ndarray, float]] = []
    for constraint in constraints:
        confidence = min(constraint.inliers / 80.0, 1.0) * constraint.inlier_ratio
        loop_edges.append(
            (
                constraint.source,
                constraint.target,
                constraint.target_from_source,
                8.0 + 12.0 * confidence,
            )
        )

    def unpack(vector: np.ndarray) -> list[np.ndarray]:
        return [original[0]] + [
            _matrix_from_vector(vector[6 * (index - 1) : 6 * index]) for index in range(1, n)
        ]

    def edge_error(
        poses: list[np.ndarray], source: int, target: int, measurement: np.ndarray
    ) -> np.ndarray:
        predicted = poses[target] @ np.linalg.inv(poses[source])
        error = np.linalg.inv(measurement) @ predicted
        return np.concatenate(
            (
                Rotation.from_matrix(error[:3, :3]).as_rotvec() / 0.05,
                error[:3, 3] / 0.10,
            )
        )

    # First solve jointly estimates one switch variable per loop constraint.
    # A bad loop can therefore turn itself off by paying a bounded prior cost,
    # instead of forcing the trajectory to tear in order to satisfy it.
    def switchable_residual(vector: np.ndarray) -> np.ndarray:
        poses = unpack(vector[:pose_size])
        switches = vector[pose_size:]
        values = []
        for source, target, measurement, weight in odometry_edges:
            values.extend((edge_error(poses, source, target, measurement) * np.sqrt(weight)).tolist())
        for loop_index, (source, target, measurement, weight) in enumerate(loop_edges):
            values.extend(
                (
                    edge_error(poses, source, target, measurement)
                    * np.sqrt(weight)
                    * switches[loop_index]
                ).tolist()
            )
        values.extend(
            (
                np.sqrt(options.switch_prior_weight) * (1.0 - switches)
            ).tolist()
        )
        return np.asarray(values, dtype=np.float64)

    switchable_rows = 6 * (len(odometry_edges) + len(loop_edges)) + len(loop_edges)
    switchable_sparsity = lil_matrix(
        (switchable_rows, pose_size + len(loop_edges)), dtype=np.int8
    )
    all_edges = odometry_edges + loop_edges
    for edge_index, (source, target, _, _) in enumerate(all_edges):
        rows = slice(6 * edge_index, 6 * edge_index + 6)
        if source > 0:
            switchable_sparsity[rows, 6 * (source - 1) : 6 * source] = 1
        if target > 0:
            switchable_sparsity[rows, 6 * (target - 1) : 6 * target] = 1
        if edge_index >= len(odometry_edges):
            loop_index = edge_index - len(odometry_edges)
            switchable_sparsity[rows, pose_size + loop_index] = 1
    prior_start = 6 * len(all_edges)
    for loop_index in range(len(loop_edges)):
        switchable_sparsity[prior_start + loop_index, pose_size + loop_index] = 1

    switchable_initial = np.concatenate((initial_poses, np.ones(len(loop_edges))))
    lower = np.concatenate((np.full(pose_size, -np.inf), np.zeros(len(loop_edges))))
    upper = np.concatenate((np.full(pose_size, np.inf), np.ones(len(loop_edges))))
    first_result = least_squares(
        switchable_residual,
        switchable_initial,
        jac_sparsity=switchable_sparsity.tocsr(),
        method="trf",
        bounds=(lower, upper),
        loss="linear",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=options.pose_graph_max_nfev,
        verbose=0,
    )
    switch_weights = np.clip(first_result.x[pose_size:], 0.0, 1.0)
    retained_indices = np.flatnonzero(switch_weights >= options.min_switch_weight)
    retained_loops = [loop_edges[index] for index in retained_indices]

    def fixed_residual(vector: np.ndarray) -> np.ndarray:
        poses = unpack(vector)
        values = []
        for source, target, measurement, weight in odometry_edges + retained_loops:
            values.extend(
                (edge_error(poses, source, target, measurement) * np.sqrt(weight)).tolist()
            )
        return np.asarray(values, dtype=np.float64)

    fixed_edges = odometry_edges + retained_loops
    fixed_sparsity = lil_matrix((6 * len(fixed_edges), pose_size), dtype=np.int8)
    for edge_index, (source, target, _, _) in enumerate(fixed_edges):
        rows = slice(6 * edge_index, 6 * edge_index + 6)
        if source > 0:
            fixed_sparsity[rows, 6 * (source - 1) : 6 * source] = 1
        if target > 0:
            fixed_sparsity[rows, 6 * (target - 1) : 6 * target] = 1

    before = fixed_residual(initial_poses)
    result = least_squares(
        fixed_residual,
        first_result.x[:pose_size],
        jac_sparsity=fixed_sparsity.tocsr(),
        method="trf",
        loss="linear",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=options.pose_graph_max_nfev,
        verbose=0,
    )
    corrected = np.stack(unpack(result.x))
    after = fixed_residual(result.x)
    original_centers = np.linalg.inv(original)[:, :3, 3]
    corrected_centers = np.linalg.inv(corrected)[:, :3, 3]
    center_correction = np.linalg.norm(corrected_centers - original_centers, axis=1)
    local_changes = []
    for source in range(n - 1):
        old_relative = original[source + 1] @ np.linalg.inv(original[source])
        new_relative = corrected[source + 1] @ np.linalg.inv(corrected[source])
        delta = np.linalg.inv(old_relative) @ new_relative
        local_changes.append(
            np.linalg.norm(delta[:3, 3])
            + 0.1 * np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
        )

    cost_before = float(0.5 * np.dot(before, before))
    cost_after = float(0.5 * np.dot(after, after))

    finite = bool(np.all(np.isfinite(corrected)))
    max_center_correction = float(np.max(center_correction))
    max_local_change = float(np.max(local_changes, initial=0.0))
    checks = {
        "finite": finite,
        "cost_decreased": cost_after < cost_before,
        "max_camera_center_correction_below_5m": max_center_correction <= 5.0,
        "max_local_odometry_change_within_limit": max_local_change
        <= options.pose_graph_max_local_change,
        "enough_robust_loop_edges": len(retained_indices)
        >= options.min_cluster_support,
    }
    applied = all(checks.values())
    report = {
        "applied": applied,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "function_evaluations": int(result.nfev),
        "cost_before": cost_before,
        "cost_after": cost_after,
        "median_camera_center_correction": float(np.median(center_correction)),
        "max_camera_center_correction": max_center_correction,
        "median_local_odometry_change": float(np.median(local_changes)),
        "max_local_odometry_change": max_local_change,
        "sanity_checks": checks,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
        "switchable_optimizer_success": bool(first_result.success),
        "switchable_optimizer_status": int(first_result.status),
        "switchable_optimizer_message": str(first_result.message),
        "switchable_function_evaluations": int(first_result.nfev),
        "robust_loop_switch_weights": switch_weights.tolist(),
        "robust_loop_edges_retained": int(len(retained_indices)),
    }
    # Hitting max_nfev is not itself a rejection: a finite, lower-cost and
    # locally smooth solution is safe to use and the final dense BA will refine
    # it.  This avoids discarding useful large graphs only because SciPy did not
    # satisfy its formal termination tolerance in time.
    return (corrected if applied else original.copy()), report


def _pose_graph_optimize(
    buffer: GraphBuffer,
    constraints: list[LoopConstraint],
    options: LoopClosureOptions,
) -> dict[str, Any]:
    original, _ = _w2c_and_centers(buffer)
    corrected, report = _optimize_pose_graph_matrices(original, constraints, options)
    if report["applied"]:
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
    timestamps = buffer.tstamp[: buffer.n_frames].detach().cpu().numpy().astype(np.int64)
    w2c, _ = _w2c_and_centers(buffer)
    K = buffer.K[0].astype(np.float64)
    cache = _FeatureCache(buffer, options)
    candidates, retrieval_report = _advanced_candidate_pairs(buffer, cache, options)
    verified_by_pair: dict[tuple[int, int], LoopConstraint] = {}
    pair_cache: dict[tuple[int, int], tuple[LoopConstraint | None, str]] = {}
    rejections: Counter[str] = Counter()
    candidate_records = []
    for source, target, similarity, sequence_similarity, direction, pose_distance in candidates:
        sequence_constraints = []
        pair_records = []
        for offset in range(-options.sequence_radius, options.sequence_radius + 1):
            pair_source = source + offset
            pair_target = target + direction * offset
            if not (0 <= pair_source < buffer.n_frames and 0 <= pair_target < buffer.n_frames):
                continue
            if pair_target <= pair_source or pair_target - pair_source < options.min_keyframe_gap:
                continue
            if timestamps[pair_target] - timestamps[pair_source] < options.min_frame_gap:
                continue
            pair = (pair_source, pair_target)
            if pair not in pair_cache:
                try:
                    pair_cache[pair] = _verify_candidate(
                        pair_source,
                        pair_target,
                        similarity,
                        sequence_similarity,
                        direction,
                        cache,
                        K,
                        timestamps,
                        w2c,
                        options,
                    )
                except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
                    logger.debug("Rejected loop pair %d -> %d: %s", pair_source, pair_target, exc)
                    pair_cache[pair] = None, "verification_exception"
            constraint, reason = pair_cache[pair]
            rejections[reason] += 1
            pair_record: dict[str, Any] = {
                "source_keyframe": pair_source,
                "target_keyframe": pair_target,
                "source_frame": int(timestamps[pair_source]),
                "target_frame": int(timestamps[pair_target]),
                "offset": offset,
                "result": reason,
            }
            if constraint is not None:
                sequence_constraints.append(constraint)
                pair_record.update(constraint.to_dict())
            pair_records.append(pair_record)

        support = len(sequence_constraints)
        seed_result = (
            "accepted_sequence"
            if support >= options.min_sequence_support
            else "insufficient_sequence_support"
        )
        if support >= options.min_sequence_support:
            for constraint in sequence_constraints:
                constraint.cluster_support = max(constraint.cluster_support, support)
                pair = (constraint.source, constraint.target)
                previous = verified_by_pair.get(pair)
                if previous is None or constraint.score > previous.score:
                    verified_by_pair[pair] = constraint
        else:
            rejections[seed_result] += 1

        record: dict[str, Any] = {
            "source_keyframe": source,
            "target_keyframe": target,
            "source_frame": int(timestamps[source]),
            "target_frame": int(timestamps[target]),
            "similarity": similarity,
            "sequence_similarity": sequence_similarity,
            "sequence_direction": direction,
            "current_camera_distance": pose_distance,
            "verified_sequence_support": support,
            "result": seed_result,
            "pairs": pair_records,
        }
        candidate_records.append(record)

    verified = list(verified_by_pair.values())
    retained = _sequence_filter(verified, options)
    report: dict[str, Any] = {
        "version": LOOP_CLOSURE_VERSION,
        "enabled": True,
        "options": asdict(options),
        "retrieval": retrieval_report,
        "verification_counts": dict(rejections),
        "verified_long_pairs_before_filter": len(verified),
        "accepted_loop_edges": len(retained),
        "accepted": [constraint.to_dict() for constraint in retained],
        "pose_graph_input_edges": [constraint.to_dict() for constraint in retained],
        "candidates": candidate_records,
        "pose_graph": None,
    }
    if not retained:
        logger.warning("Loop closure v2 found no sequence-supported long-range edge")
        return None, report

    report["pose_graph"] = _pose_graph_optimize(buffer, retained, options)
    switch_weights = report["pose_graph"]["robust_loop_switch_weights"]
    minimum_switch = options.min_switch_weight
    for record, switch_weight in zip(
        report["pose_graph_input_edges"], switch_weights, strict=True
    ):
        record["robust_switch_weight"] = float(switch_weight)
        record["retained_after_switchable_optimization"] = bool(
            switch_weight >= minimum_switch
        )
    if not report["pose_graph"]["applied"]:
        logger.warning(
            "Rejected loop pose-graph correction: %s",
            ", ".join(report["pose_graph"]["rejection_reasons"]),
        )
        return None, report
    robust_pairs = [
        (constraint, float(switch_weight))
        for constraint, switch_weight in zip(retained, switch_weights, strict=True)
        if switch_weight >= minimum_switch
    ]
    robust_retained = [constraint for constraint, _ in robust_pairs]
    report["pose_graph_downweighted_edges"] = len(retained) - len(robust_retained)
    report["accepted_loop_edges"] = len(robust_retained)
    report["accepted"] = []
    for constraint, switch_weight in robust_pairs:
        record = constraint.to_dict()
        record["robust_switch_weight"] = switch_weight
        report["accepted"].append(record)
    if not robust_retained:
        logger.warning("Loop pose graph downweighted every long-range edge")
        return None, report
    edges = torch.tensor(
        [[constraint.source, constraint.target] for constraint in robust_retained],
        dtype=torch.long,
        device=buffer.device,
    )
    logger.info(
        "Loop closure accepted %d/%d candidates; pose graph cost %.3f -> %.3f",
        len(robust_retained),
        len(candidates),
        report["pose_graph"]["cost_before"],
        report["pose_graph"]["cost_after"],
    )
    return edges, report


def snapshot_w2c(buffer: GraphBuffer) -> np.ndarray:
    return _w2c_and_centers(buffer)[0].copy()


def finalize_loop_report(
    buffer: GraphBuffer,
    report: dict[str, Any] | None,
    reference_w2c: np.ndarray | None,
) -> None:
    """Measure what remains after the final recurrent DROID bundle adjustment."""

    if report is None or reference_w2c is None:
        return
    final_w2c, _ = _w2c_and_centers(buffer)
    reference_centers = np.linalg.inv(reference_w2c)[:, :3, 3]
    final_centers = np.linalg.inv(final_w2c)[:, :3, 3]
    center_changes = np.linalg.norm(final_centers - reference_centers, axis=1)
    rotation_errors = []
    translation_errors = []
    for constraint in report.get("accepted", []):
        source = int(constraint["source"])
        target = int(constraint["target"])
        measurement = np.asarray(constraint["target_from_source"], dtype=np.float64)
        predicted = final_w2c[target] @ np.linalg.inv(final_w2c[source])
        error = np.linalg.inv(measurement) @ predicted
        rotation_error = _rotation_degrees(error)
        translation_error = float(np.linalg.norm(error[:3, 3]))
        constraint["final_ba_rotation_error_deg"] = rotation_error
        constraint["final_ba_translation_error"] = translation_error
        rotation_errors.append(rotation_error)
        translation_errors.append(translation_error)

    report["final_dense_ba"] = {
        "measured": True,
        "median_camera_center_change_from_preloop": float(np.median(center_changes)),
        "max_camera_center_change_from_preloop": float(np.max(center_changes, initial=0.0)),
        "median_accepted_loop_rotation_error_deg": float(np.median(rotation_errors))
        if rotation_errors
        else None,
        "max_accepted_loop_rotation_error_deg": float(np.max(rotation_errors, initial=0.0))
        if rotation_errors
        else None,
        "median_accepted_loop_translation_error": float(np.median(translation_errors))
        if translation_errors
        else None,
        "max_accepted_loop_translation_error": float(np.max(translation_errors, initial=0.0))
        if translation_errors
        else None,
    }
