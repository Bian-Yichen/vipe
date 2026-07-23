# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure NumPy/SciPy local-submap registration for viewpoint-reversed loops.

The image verifier in :mod:`loop_closure` is intentionally strict.  It cannot
verify a real revisit when the camera sees the same landing from the opposite
direction and the two RGB images have no common pixels.  This module instead
registers two short, locally consistent SLAM point-cloud windows.

Registration is conservative:

* a voxel cross-correlation supplies several translation hypotheses;
* reciprocal, trimmed ICP refines each hypothesis;
* the same transform must be recovered at two different submap radii;
* overlap, residual, colour, surface-normal and ambiguity checks all have to
  pass before a constraint is returned.

There is deliberately no dependency on Torch or OpenCV.  Keeping the geometric
core pure makes it possible to validate the exact algorithm on saved
``*_slam_map.pt`` artifacts without rerunning GPU inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SubmapRegistrationOptions:
    radius_small: int = 3
    radius_large: int = 5
    voxel_size: float = 0.08
    correlation_voxel_size: float = 0.10
    correlation_peaks: int = 5
    max_correlation_cells: int = 8_000_000
    max_translation: float = 3.0
    max_rotation_deg: float = 15.0
    icp_max_correspondence: float = 0.55
    evaluation_distance: float = 0.20
    min_symmetric_overlap: float = 0.20
    min_mutual_correspondences: int = 220
    max_rmse: float = 0.12
    min_normal_consistency: float = 0.62
    max_chromaticity_error: float = 0.14
    translation_consistency: float = 0.40
    rotation_consistency_deg: float = 4.0
    min_ambiguity_ratio: float = 1.02


@dataclass
class RegistrationMetrics:
    source_overlap: float
    target_overlap: float
    symmetric_overlap: float
    mutual_correspondences: int
    rmse: float
    median_distance: float
    normal_consistency: float
    chromaticity_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubmapRegistrationResult:
    accepted: bool
    reason: str
    target_world_to_source_world: np.ndarray | None
    metrics: RegistrationMetrics | None
    multiscale_support: int
    ambiguity_ratio: float
    correction_translation: float
    correction_rotation_deg: float
    source_points: int
    target_points: int
    hypotheses: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "target_world_to_source_world": (
                None
                if self.target_world_to_source_world is None
                else self.target_world_to_source_world.tolist()
            ),
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "multiscale_support": self.multiscale_support,
            "ambiguity_ratio": self.ambiguity_ratio,
            "correction_translation": self.correction_translation,
            "correction_rotation_deg": self.correction_rotation_deg,
            "source_points": self.source_points,
            "target_points": self.target_points,
            "hypotheses": self.hypotheses,
        }


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _rotation_degrees(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points.reshape(0, 3), colors.reshape(0, 3)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points = np.asarray(points[finite], dtype=np.float64)
    colors = np.asarray(colors[finite], dtype=np.float64)
    if len(points) == 0:
        return points.reshape(0, 3), colors.reshape(0, 3)
    voxels = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxels, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    output_points = np.zeros((len(count), 3), dtype=np.float64)
    output_colors = np.zeros((len(count), 3), dtype=np.float64)
    np.add.at(output_points, inverse, points)
    np.add.at(output_colors, inverse, colors)
    output_points /= count[:, None]
    output_colors /= count[:, None]
    return output_points, output_colors


def packed_submap(
    xyz: np.ndarray,
    rgb: np.ndarray,
    packinfo: np.ndarray,
    center: int,
    radius: int,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate all views in a keyframe window from ``SLAMMap`` packed arrays."""

    packinfo = np.asarray(packinfo, dtype=np.int64)
    if packinfo.ndim == 2:
        packinfo = packinfo[:, None, :]
    first = max(0, int(center) - int(radius))
    last = min(len(packinfo), int(center) + int(radius) + 1)
    point_parts = []
    color_parts = []
    for keyframe in range(first, last):
        for start, count in packinfo[keyframe]:
            start, count = int(start), int(count)
            if count <= 0:
                continue
            point_parts.append(xyz[start : start + count])
            color_parts.append(rgb[start : start + count])
    if not point_parts:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty.copy()
    return _voxel_downsample(
        np.concatenate(point_parts, axis=0),
        np.concatenate(color_parts, axis=0),
        voxel_size,
    )


def _bounded_points(points: np.ndarray) -> np.ndarray:
    """Drop extreme depth-filter outliers before allocating correlation grids."""

    if len(points) < 100:
        return points
    lower, upper = np.quantile(points, [0.002, 0.998], axis=0)
    keep = ((points >= lower) & (points <= upper)).all(axis=1)
    return points[keep]


def _correlation_hypotheses(
    source: np.ndarray,
    target: np.ndarray,
    options: SubmapRegistrationOptions,
) -> list[tuple[float, np.ndarray]]:
    """Return translations that map ``target`` into ``source`` coordinates."""

    source = _bounded_points(source)
    target = _bounded_points(target)
    voxel_size = float(options.correlation_voxel_size)
    for _ in range(4):
        source_min = np.floor(source.min(axis=0) / voxel_size).astype(np.int64)
        target_min = np.floor(target.min(axis=0) / voxel_size).astype(np.int64)
        source_index = np.floor(source / voxel_size).astype(np.int64) - source_min
        target_index = np.floor(target / voxel_size).astype(np.int64) - target_min
        source_shape = source_index.max(axis=0) + 1
        target_shape = target_index.max(axis=0) + 1
        correlation_shape = source_shape + target_shape - 1
        if int(np.prod(correlation_shape, dtype=np.int64)) <= options.max_correlation_cells:
            break
        voxel_size *= 1.35

    source_grid = np.zeros(tuple(source_shape), dtype=np.float32)
    target_grid = np.zeros(tuple(target_shape), dtype=np.float32)
    source_grid[tuple(source_index.T)] = 1.0
    target_grid[tuple(target_index.T)] = 1.0
    correlation = fftconvolve(
        source_grid,
        target_grid[::-1, ::-1, ::-1],
        mode="full",
    )
    local_max = maximum_filter(correlation, size=5, mode="constant")
    peak_indices = np.argwhere((correlation == local_max) & (correlation > 0.0))
    if len(peak_indices) == 0:
        return [(0.0, np.zeros(3, dtype=np.float64))]
    peak_values = correlation[tuple(peak_indices.T)]
    order = np.argsort(peak_values)[::-1]
    hypotheses: list[tuple[float, np.ndarray]] = []
    for index in order:
        lag = peak_indices[index] - (target_shape - 1)
        translation = (source_min - target_min + lag) * voxel_size
        if np.linalg.norm(translation) > options.max_translation:
            continue
        hypotheses.append((float(peak_values[index]), translation.astype(np.float64)))
        if len(hypotheses) >= options.correlation_peaks:
            break
    if not any(np.linalg.norm(translation) < 0.25 for _, translation in hypotheses):
        hypotheses.append((0.0, np.zeros(3, dtype=np.float64)))
    return hypotheses


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd(
        (source - source_center).T @ (target - target_center),
        full_matrices=False,
    )
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _reciprocal_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_tree = cKDTree(target)
    distance, target_index = target_tree.query(source, workers=1)
    reverse_index = cKDTree(source).query(target, workers=1)[1]
    source_index = np.arange(len(source), dtype=np.int64)
    keep = (distance < max_distance) & (reverse_index[target_index] == source_index)
    return source_index[keep], target_index[keep], distance[keep]


def _trimmed_icp(
    moving: np.ndarray,
    fixed: np.ndarray,
    initial: np.ndarray,
    options: SubmapRegistrationOptions,
) -> np.ndarray:
    matrix = initial.copy()
    thresholds = (
        options.icp_max_correspondence,
        max(0.32, 3.5 * options.voxel_size),
        max(0.20, 2.25 * options.voxel_size),
    )
    for threshold in thresholds:
        for _ in range(9):
            transformed = _transform(moving, matrix)
            moving_index, fixed_index, distance = _reciprocal_correspondences(
                transformed,
                fixed,
                threshold,
            )
            if len(moving_index) < 60:
                break
            retained = max(60, int(0.80 * len(moving_index)))
            order = np.argsort(distance)[:retained]
            rotation, translation = _rigid_fit(
                transformed[moving_index[order]],
                fixed[fixed_index[order]],
            )
            increment = np.eye(4, dtype=np.float64)
            increment[:3, :3] = rotation
            increment[:3, 3] = translation
            matrix = increment @ matrix
            if np.linalg.norm(translation) < 1e-5 and _rotation_degrees(increment) < 1e-3:
                break
    return matrix


def _normals_at_indices(
    points: np.ndarray,
    indices: np.ndarray,
    tree: cKDTree,
    neighbors: int = 12,
) -> np.ndarray:
    neighbor_index = tree.query(
        points[indices],
        k=min(neighbors, len(points)),
        workers=1,
    )[1]
    neighborhoods = points[neighbor_index]
    neighborhoods -= neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", neighborhoods, neighborhoods)
    _, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors[:, :, 0]


def _registration_metrics(
    moving: np.ndarray,
    moving_color: np.ndarray,
    fixed: np.ndarray,
    fixed_color: np.ndarray,
    matrix: np.ndarray,
    options: SubmapRegistrationOptions,
    *,
    detailed: bool = True,
) -> RegistrationMetrics:
    transformed = _transform(moving, matrix)
    fixed_tree = cKDTree(fixed)
    moving_tree = cKDTree(transformed)
    moving_distance, fixed_index = fixed_tree.query(transformed, workers=1)
    fixed_distance = moving_tree.query(fixed, workers=1)[0]
    moving_index = np.arange(len(moving), dtype=np.int64)
    reverse_index = moving_tree.query(fixed, workers=1)[1]
    mutual = (
        (moving_distance < options.evaluation_distance)
        & (reverse_index[fixed_index] == moving_index)
    )
    mutual_moving = moving_index[mutual]
    mutual_fixed = fixed_index[mutual]
    mutual_distance = moving_distance[mutual]
    if len(mutual_moving) == 0:
        return RegistrationMetrics(
            source_overlap=0.0,
            target_overlap=0.0,
            symmetric_overlap=0.0,
            mutual_correspondences=0,
            rmse=float("inf"),
            median_distance=float("inf"),
            normal_consistency=0.0,
            chromaticity_error=float("inf"),
        )

    if not detailed:
        normal_consistency = 1.0
    else:
        if len(mutual_moving) > 600:
            selected = np.linspace(0, len(mutual_moving) - 1, 600, dtype=np.int64)
            normal_moving_index = mutual_moving[selected]
            normal_fixed_index = mutual_fixed[selected]
        else:
            normal_moving_index = mutual_moving
            normal_fixed_index = mutual_fixed
        moving_normals = _normals_at_indices(
            transformed,
            normal_moving_index,
            moving_tree,
        )
        fixed_normals = _normals_at_indices(
            fixed,
            normal_fixed_index,
            fixed_tree,
        )
        normal_consistency = float(
            np.median(np.abs(np.sum(moving_normals * fixed_normals, axis=1)))
        )

    moving_rgb = moving_color[mutual_moving]
    fixed_rgb = fixed_color[mutual_fixed]
    moving_chromaticity = moving_rgb / (moving_rgb.sum(axis=1, keepdims=True) + 1e-6)
    fixed_chromaticity = fixed_rgb / (fixed_rgb.sum(axis=1, keepdims=True) + 1e-6)
    chromaticity_error = float(
        np.median(np.linalg.norm(moving_chromaticity - fixed_chromaticity, axis=1))
    )
    source_overlap = float(np.mean(moving_distance < options.evaluation_distance))
    target_overlap = float(np.mean(fixed_distance < options.evaluation_distance))
    return RegistrationMetrics(
        source_overlap=source_overlap,
        target_overlap=target_overlap,
        symmetric_overlap=min(source_overlap, target_overlap),
        mutual_correspondences=int(len(mutual_moving)),
        rmse=float(np.sqrt(np.mean(mutual_distance**2))),
        median_distance=float(np.median(moving_distance)),
        normal_consistency=normal_consistency,
        chromaticity_error=chromaticity_error,
    )


def _quality(metrics: RegistrationMetrics, normalized_correlation: float) -> float:
    normal = np.clip(metrics.normal_consistency, 0.0, 1.0)
    color = np.clip(1.0 - metrics.chromaticity_error / 0.20, 0.0, 1.0)
    residual = np.clip(1.0 - metrics.rmse / 0.20, 0.0, 1.0)
    return float(
        0.32 * metrics.symmetric_overlap
        + 0.12 * normal
        + 0.10 * color
        + 0.16 * residual
        + 0.30 * normalized_correlation
    )


def _relaxed_geometry_pass(
    metrics: RegistrationMetrics,
    matrix: np.ndarray,
    options: SubmapRegistrationOptions,
) -> bool:
    return bool(
        metrics.symmetric_overlap >= 0.70 * options.min_symmetric_overlap
        and metrics.mutual_correspondences >= max(
            80,
            int(0.50 * options.min_mutual_correspondences),
        )
        and metrics.rmse <= 1.35 * options.max_rmse
        and metrics.normal_consistency >= options.min_normal_consistency - 0.15
        and metrics.chromaticity_error <= 1.35 * options.max_chromaticity_error
        and np.linalg.norm(matrix[:3, 3]) <= options.max_translation
        and _rotation_degrees(matrix) <= options.max_rotation_deg
    )


def _consistent(
    first: np.ndarray,
    second: np.ndarray,
    options: SubmapRegistrationOptions,
) -> bool:
    return transforms_are_consistent(
        first,
        second,
        max_translation=options.translation_consistency,
        max_rotation_deg=options.rotation_consistency_deg,
    )


def transforms_are_consistent(
    first: np.ndarray,
    second: np.ndarray,
    *,
    max_translation: float,
    max_rotation_deg: float,
) -> bool:
    """Check whether two independently estimated world corrections agree."""

    delta = np.linalg.inv(first) @ second
    return bool(
        np.linalg.norm(delta[:3, 3]) <= max_translation
        and _rotation_degrees(delta) <= max_rotation_deg
    )


def independent_transform_support(
    candidates: list[tuple[int, int, np.ndarray]],
    *,
    index_radius: int,
    max_translation: float,
    max_rotation_deg: float,
) -> np.ndarray:
    """Count nearby, independently estimated transforms that agree."""

    support = np.zeros(len(candidates), dtype=np.int64)
    for index, (source, target, transform) in enumerate(candidates):
        support[index] = sum(
            abs(source - other_source) <= index_radius
            and abs(target - other_target) <= index_radius
            and transforms_are_consistent(
                transform,
                other_transform,
                max_translation=max_translation,
                max_rotation_deg=max_rotation_deg,
            )
            for other_source, other_target, other_transform in candidates
        )
    return support


def _mean_transform(matrices: list[np.ndarray]) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = Rotation.from_matrix(
        np.stack([matrix[:3, :3] for matrix in matrices])
    ).mean().as_matrix()
    output[:3, 3] = np.mean([matrix[:3, 3] for matrix in matrices], axis=0)
    return output


def register_packed_submaps(
    xyz: np.ndarray,
    rgb: np.ndarray,
    packinfo: np.ndarray,
    source: int,
    target: int,
    options: SubmapRegistrationOptions,
) -> SubmapRegistrationResult:
    """Register a later target submap into an earlier source submap."""

    radii = tuple(dict.fromkeys((options.radius_small, options.radius_large)))
    scale_candidates: list[list[dict[str, Any]]] = []
    scale_clouds = []
    hypotheses_report: list[dict[str, Any]] = []
    for radius in radii:
        source_points, source_color = packed_submap(
            xyz,
            rgb,
            packinfo,
            source,
            radius,
            options.voxel_size,
        )
        target_points, target_color = packed_submap(
            xyz,
            rgb,
            packinfo,
            target,
            radius,
            options.voxel_size,
        )
        scale_clouds.append(
            (source_points, source_color, target_points, target_color)
        )
        if min(len(source_points), len(target_points)) < 300:
            scale_candidates.append([])
            continue
        seeds = _correlation_hypotheses(source_points, target_points, options)
        peak_max = max((score for score, _ in seeds), default=1.0)
        candidates = []
        for rank, (correlation, translation) in enumerate(seeds):
            initial = np.eye(4, dtype=np.float64)
            initial[:3, 3] = translation
            matrix = _trimmed_icp(
                target_points,
                source_points,
                initial,
                options,
            )
            metrics = _registration_metrics(
                target_points,
                target_color,
                source_points,
                source_color,
                matrix,
                options,
                detailed=False,
            )
            normalized_correlation = float(correlation / max(peak_max, 1e-6))
            record = {
                "radius": int(radius),
                "rank": int(rank),
                "correlation": float(correlation),
                "normalized_correlation": normalized_correlation,
                "seed_translation": translation.tolist(),
                "transform": matrix.tolist(),
                "correction_translation": float(np.linalg.norm(matrix[:3, 3])),
                "correction_rotation_deg": _rotation_degrees(matrix),
                "metrics": metrics.to_dict(),
                "quality": _quality(metrics, normalized_correlation),
            }
            record["relaxed_geometry_pass"] = _relaxed_geometry_pass(
                metrics,
                matrix,
                options,
            )
            hypotheses_report.append(record)
            if record["relaxed_geometry_pass"]:
                candidates.append(
                    {
                        **record,
                        "_matrix": matrix,
                        "_metrics": metrics,
                    }
                )
        scale_candidates.append(candidates)

    if len(scale_candidates) < 2 or any(not candidates for candidates in scale_candidates):
        source_count = len(scale_clouds[-1][0]) if scale_clouds else 0
        target_count = len(scale_clouds[-1][2]) if scale_clouds else 0
        return SubmapRegistrationResult(
            accepted=False,
            reason="no_geometrically_valid_multiscale_hypothesis",
            target_world_to_source_world=None,
            metrics=None,
            multiscale_support=0,
            ambiguity_ratio=0.0,
            correction_translation=0.0,
            correction_rotation_deg=0.0,
            source_points=source_count,
            target_points=target_count,
            hypotheses=hypotheses_report,
        )

    clusters = []
    for first in scale_candidates[0]:
        matches = [first]
        for candidates in scale_candidates[1:]:
            compatible = [
                candidate
                for candidate in candidates
                if _consistent(first["_matrix"], candidate["_matrix"], options)
            ]
            if compatible:
                matches.append(max(compatible, key=lambda item: item["quality"]))
        if len(matches) < len(scale_candidates):
            continue
        score = float(np.mean([match["quality"] for match in matches]))
        clusters.append((score, matches))
    # Adjacent FFT cells often converge to the same ICP basin.  Treating those
    # duplicates as independent runner-up solutions makes an unambiguous loop
    # look ambiguous (for example translations 1.91 m and 1.98 m).  Collapse
    # consistent basins before computing the winner/runner-up ratio.
    unique_clusters: list[tuple[float, list[dict[str, Any]], np.ndarray]] = []
    for score, matches in sorted(clusters, key=lambda item: item[0], reverse=True):
        representative = _mean_transform(
            [candidate["_matrix"] for candidate in matches]
        )
        duplicate = next(
            (
                index
                for index, (_, _, old_representative) in enumerate(unique_clusters)
                if _consistent(representative, old_representative, options)
            ),
            None,
        )
        if duplicate is None:
            unique_clusters.append((score, matches, representative))
        elif score > unique_clusters[duplicate][0]:
            unique_clusters[duplicate] = (score, matches, representative)
    clusters = [
        (score, matches)
        for score, matches, _ in sorted(
            unique_clusters,
            key=lambda item: item[0],
            reverse=True,
        )
    ]
    if not clusters:
        source_count = len(scale_clouds[-1][0])
        target_count = len(scale_clouds[-1][2])
        return SubmapRegistrationResult(
            accepted=False,
            reason="submap_transform_not_repeatable_across_scales",
            target_world_to_source_world=None,
            metrics=None,
            multiscale_support=1,
            ambiguity_ratio=0.0,
            correction_translation=0.0,
            correction_rotation_deg=0.0,
            source_points=source_count,
            target_points=target_count,
            hypotheses=hypotheses_report,
        )

    winner_score, winner = clusters[0]
    runner_up_score = clusters[1][0] if len(clusters) > 1 else 0.0
    ambiguity_ratio = (
        999.0
        if runner_up_score <= 1e-9
        else float(winner_score / runner_up_score)
    )
    initial = _mean_transform([candidate["_matrix"] for candidate in winner])
    source_points, source_color, target_points, target_color = scale_clouds[-1]
    matrix = _trimmed_icp(
        target_points,
        source_points,
        initial,
        options,
    )
    metrics = _registration_metrics(
        target_points,
        target_color,
        source_points,
        source_color,
        matrix,
        options,
    )
    checks = {
        "symmetric_overlap": bool(
            metrics.symmetric_overlap >= options.min_symmetric_overlap
        ),
        "mutual_correspondences": bool(
            metrics.mutual_correspondences
            >= options.min_mutual_correspondences
        ),
        "rmse": bool(metrics.rmse <= options.max_rmse),
        "normal_consistency": bool(
            metrics.normal_consistency >= options.min_normal_consistency
        ),
        "chromaticity": bool(
            metrics.chromaticity_error <= options.max_chromaticity_error
        ),
        "translation": bool(
            np.linalg.norm(matrix[:3, 3]) <= options.max_translation
        ),
        "rotation": bool(
            _rotation_degrees(matrix) <= options.max_rotation_deg
        ),
        "ambiguity": bool(
            ambiguity_ratio >= options.min_ambiguity_ratio
        ),
    }
    accepted = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    hypotheses_report.append(
        {
            "selected_multiscale_score": winner_score,
            "runner_up_multiscale_score": runner_up_score,
            "ambiguity_ratio": ambiguity_ratio,
            "final_transform": matrix.tolist(),
            "final_metrics": metrics.to_dict(),
            "checks": checks,
        }
    )
    return SubmapRegistrationResult(
        accepted=accepted,
        reason="accepted" if accepted else "failed_" + "_".join(failed),
        target_world_to_source_world=matrix if accepted else None,
        metrics=metrics,
        multiscale_support=len(winner),
        ambiguity_ratio=ambiguity_ratio,
        correction_translation=float(np.linalg.norm(matrix[:3, 3])),
        correction_rotation_deg=_rotation_degrees(matrix),
        source_points=len(source_points),
        target_points=len(target_points),
        hypotheses=hypotheses_report,
    )
