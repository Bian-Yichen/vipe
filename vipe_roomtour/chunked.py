# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlapping chunk inference and robust Sim(3) stitching for long tours."""

from __future__ import annotations

import csv
import json
import logging
import os
import queue
import subprocess
import tempfile
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import ArtifactSet, Calibration, export_calibration, load_calibration, write_ply
from .depth_options import DenseDepthOptions
from .fusion import VoxelAccumulator
from .geometry import estimate_floor_y, estimate_level_frame, intrinsics_matrix, pose_tilt_degrees
from .map_builder import MapOptions, build_map, probe_video
from .output_paths import resolve_output_path
from .runner import _run_vipe
from .topdown import render_topdown

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkSpec:
    index: int
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def name(self) -> str:
        return f"chunk_{self.index:04d}_f{self.start_frame:06d}_f{self.end_frame - 1:06d}"


@dataclass(frozen=True)
class SimilarityTransform:
    """Local-to-global similarity: ``p_global = scale * R @ p_local + t``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    @staticmethod
    def identity() -> "SimilarityTransform":
        return SimilarityTransform(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation

    def transform_poses(self, c2w: np.ndarray) -> np.ndarray:
        c2w = np.asarray(c2w, dtype=np.float64)
        output = c2w.copy()
        output[:, :3, :3] = np.einsum("ij,njk->nik", self.rotation, c2w[:, :3, :3])
        output[:, :3, 3] = self.transform_points(c2w[:, :3, 3])
        output[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "matrix_without_scale": np.block(
                [[self.rotation, self.translation[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]
            ).tolist(),
            "note": "Points and pose translations use scale*R*x+t; pose rotations use R*R_local.",
        }


@dataclass(frozen=True)
class StitchThresholds:
    min_overlap_frames: int = 30
    min_baseline: float = 0.25
    max_position_rmse: float = 0.75
    max_rotation_median_deg: float = 10.0
    min_scale: float = 0.5
    max_scale: float = 2.0


@dataclass
class ChunkResult:
    spec: ChunkSpec
    artifact: ArtifactSet
    map_dir: Path
    calibration: Calibration
    transform: SimilarityTransform


def plan_chunks(total_frames: int, chunk_frames: int, overlap_frames: int) -> list[ChunkSpec]:
    if total_frames < 3:
        raise ValueError("At least three video frames are required")
    if chunk_frames < 100:
        raise ValueError("chunk_frames must be at least 100")
    if overlap_frames < 30:
        raise ValueError("overlap_frames must be at least 30 for robust alignment")
    if overlap_frames * 2 >= chunk_frames:
        raise ValueError("overlap_frames must be less than half of chunk_frames")
    if total_frames <= chunk_frames:
        return [ChunkSpec(0, 0, total_frames)]

    stride = chunk_frames - overlap_frames
    chunks: list[ChunkSpec] = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        chunks.append(ChunkSpec(len(chunks), start, end))
        if end == total_frames:
            break
        start += stride
    return chunks


def count_video_frames(path: Path) -> int:
    """Count decoded frames once; container frame counts are often only declarations."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe was not found; install FFmpeg first") from exc
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {path}")
    stream = streams[0]
    value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if value in (None, "N/A") or int(value) <= 0:
        raise RuntimeError(f"Could not count decoded frames in {path}")
    return int(value)


def _rotation_angle_degrees(rotation_matrices: np.ndarray) -> np.ndarray:
    return np.degrees(Rotation.from_matrix(rotation_matrices).magnitude())


def _robust_rotation(candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations = Rotation.from_matrix(candidates)
    estimate = rotations.mean()
    residual = (estimate.inv() * rotations).magnitude()
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    threshold = min(np.deg2rad(20.0), max(np.deg2rad(2.0), median + 3.0 * 1.4826 * mad))
    inliers = residual <= threshold
    if int(inliers.sum()) < 3:
        inliers = np.ones(len(candidates), dtype=bool)
    return Rotation.from_matrix(candidates[inliers]).mean().as_matrix(), inliers


def estimate_pose_similarity(
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
    *,
    thresholds: StitchThresholds,
) -> tuple[SimilarityTransform, dict[str, Any]]:
    """Align matching source poses to target poses using robust orientation + scale."""

    source_c2w = np.asarray(source_c2w, dtype=np.float64)
    target_c2w = np.asarray(target_c2w, dtype=np.float64)
    if source_c2w.shape != target_c2w.shape or source_c2w.ndim != 3 or source_c2w.shape[1:] != (4, 4):
        raise ValueError("source_c2w and target_c2w must have matching Nx4x4 shapes")
    if len(source_c2w) < thresholds.min_overlap_frames:
        raise ValueError(
            f"Only {len(source_c2w)} shared poses; need at least {thresholds.min_overlap_frames}"
        )

    source_rot = source_c2w[:, :3, :3]
    target_rot = target_c2w[:, :3, :3]
    rotation_candidates = np.einsum("nij,nkj->nik", target_rot, source_rot)
    align_rotation, rotation_inliers = _robust_rotation(rotation_candidates)

    source_centers = source_c2w[:, :3, 3] @ align_rotation.T
    target_centers = target_c2w[:, :3, 3]
    target_median = np.median(target_centers, axis=0)
    baseline = float(2.0 * np.percentile(np.linalg.norm(target_centers - target_median, axis=1), 95))
    if baseline < thresholds.min_baseline:
        raise ValueError(
            f"Overlap camera baseline is only {baseline:.3f}; need {thresholds.min_baseline:.3f}. "
            "Increase overlap or place chunk boundaries in a moving section."
        )

    weights = rotation_inliers.astype(np.float64)
    scale = 1.0
    translation = np.zeros(3, dtype=np.float64)
    for _ in range(10):
        weight_sum = float(weights.sum())
        if weight_sum < 3:
            raise ValueError("Too few inlier poses remain for Sim(3) fitting")
        source_mean = np.sum(source_centers * weights[:, None], axis=0) / weight_sum
        target_mean = np.sum(target_centers * weights[:, None], axis=0) / weight_sum
        source_delta = source_centers - source_mean
        target_delta = target_centers - target_mean
        denominator = float(np.sum(weights[:, None] * source_delta * source_delta))
        if denominator < 1e-10:
            raise ValueError("Overlap motion is degenerate for scale estimation")
        scale = float(np.sum(weights[:, None] * source_delta * target_delta) / denominator)
        translation = target_mean - scale * source_mean
        residual = np.linalg.norm(scale * source_centers + translation - target_centers, axis=1)
        residual_median = float(np.median(residual[rotation_inliers]))
        mad = float(np.median(np.abs(residual[rotation_inliers] - residual_median)))
        huber_delta = max(0.02, residual_median + 2.5 * 1.4826 * mad)
        robust_weights = np.minimum(1.0, huber_delta / np.maximum(residual, 1e-9))
        weights = robust_weights * rotation_inliers

    transform = SimilarityTransform(scale, align_rotation, translation)
    mapped = transform.transform_poses(source_c2w)
    position_residual = np.linalg.norm(mapped[:, :3, 3] - target_c2w[:, :3, 3], axis=1)
    rotation_residual = _rotation_angle_degrees(
        np.einsum("nij,nkj->nik", target_rot, mapped[:, :3, :3])
    )
    position_inliers = rotation_inliers & (weights >= 0.5)
    if int(position_inliers.sum()) < 3:
        position_inliers = rotation_inliers
    position_rmse = float(np.sqrt(np.mean(position_residual[position_inliers] ** 2)))
    rotation_median = float(np.median(rotation_residual[rotation_inliers]))
    metrics = {
        "shared_frames": int(len(source_c2w)),
        "rotation_inliers": int(rotation_inliers.sum()),
        "position_inliers": int(position_inliers.sum()),
        "overlap_baseline": baseline,
        "scale": scale,
        "position_rmse": position_rmse,
        "position_median": float(np.median(position_residual)),
        "position_max": float(np.max(position_residual)),
        "rotation_median_deg": rotation_median,
        "rotation_max_deg": float(np.max(rotation_residual)),
    }
    if not thresholds.min_scale <= scale <= thresholds.max_scale:
        raise ValueError(
            f"Estimated chunk scale {scale:.4f} is outside "
            f"[{thresholds.min_scale}, {thresholds.max_scale}]"
        )
    if position_rmse > thresholds.max_position_rmse:
        raise ValueError(
            f"Overlap position RMSE {position_rmse:.3f} exceeds {thresholds.max_position_rmse:.3f}"
        )
    if rotation_median > thresholds.max_rotation_median_deg:
        raise ValueError(
            f"Overlap median rotation error {rotation_median:.3f} deg exceeds "
            f"{thresholds.max_rotation_median_deg:.3f} deg"
        )
    return transform, metrics


def _global_indices(spec: ChunkSpec, calibration: Calibration) -> np.ndarray:
    return calibration.indices.astype(np.int64) + spec.start_frame


def _overlap_pose_arrays(previous: ChunkResult, current: ChunkResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    previous_indices = _global_indices(previous.spec, previous.calibration)
    current_indices = _global_indices(current.spec, current.calibration)
    shared, previous_rows, current_rows = np.intersect1d(
        previous_indices,
        current_indices,
        assume_unique=True,
        return_indices=True,
    )
    previous_global = previous.transform.transform_poses(previous.calibration.c2w[previous_rows])
    current_local = current.calibration.c2w[current_rows]
    return shared, current_local, previous_global


def _align_adjacent_chunks(
    previous: ChunkResult,
    current: ChunkResult,
    thresholds: StitchThresholds,
) -> dict[str, Any]:
    shared, current_local, previous_global = _overlap_pose_arrays(previous, current)
    expected_overlap = previous.spec.end_frame - current.spec.start_frame
    if len(shared) != expected_overlap:
        raise RuntimeError(
            f"{previous.spec.name} and {current.spec.name} should share "
            f"{expected_overlap} frames, but their artifacts share {len(shared)}"
        )
    transform, metrics = estimate_pose_similarity(current_local, previous_global, thresholds=thresholds)
    current.transform = transform
    metrics.update(
        {
            "previous_chunk": previous.spec.name,
            "current_chunk": current.spec.name,
            "first_shared_frame": int(shared[0]),
            "last_shared_frame": int(shared[-1]),
        }
    )
    logger.info(
        "Aligned %s: scale=%.5f, position RMSE=%.4f, rotation median=%.3f deg",
        current.spec.name,
        transform.scale,
        metrics["position_rmse"],
        metrics["rotation_median_deg"],
    )
    return metrics


def _chunk_weight(spec: ChunkSpec, frame_index: int, chunk_count: int, overlap_frames: int) -> float:
    weight = 1.0
    denominator = max(overlap_frames - 1, 1)
    if spec.index > 0 and frame_index < spec.start_frame + overlap_frames:
        weight = min(weight, (frame_index - spec.start_frame) / denominator)
    if spec.index < chunk_count - 1 and frame_index >= spec.end_frame - overlap_frames:
        weight = min(weight, (spec.end_frame - 1 - frame_index) / denominator)
    return float(np.clip(weight, 0.0, 1.0))


def merge_chunk_calibrations(
    chunks: list[ChunkResult],
    *,
    total_frames: int,
    overlap_frames: int,
) -> tuple[Calibration, np.ndarray, np.ndarray]:
    """Blend duplicate overlap poses and intrinsics into one continuous result."""

    contributions: dict[int, list[tuple[np.ndarray, np.ndarray, float, float, int]]] = {}
    for chunk in chunks:
        poses = chunk.transform.transform_poses(chunk.calibration.c2w)
        for row, frame_index in enumerate(_global_indices(chunk.spec, chunk.calibration)):
            weight = _chunk_weight(chunk.spec, int(frame_index), len(chunks), overlap_frames)
            contributions.setdefault(int(frame_index), []).append(
                (
                    poses[row],
                    chunk.calibration.intrinsics[row],
                    weight,
                    chunk.transform.scale,
                    chunk.spec.index,
                )
            )

    expected = np.arange(total_frames, dtype=np.int64)
    if sorted(contributions) != expected.tolist():
        missing = sorted(set(expected.tolist()) - set(contributions))
        raise RuntimeError(f"Chunk calibration does not cover all source frames; first missing={missing[:5]}")

    merged_poses: list[np.ndarray] = []
    merged_intrinsics: list[np.ndarray] = []
    depth_scales: list[float] = []
    primary_chunks: list[int] = []
    for frame_index in expected:
        entries = contributions[int(frame_index)]
        weights = np.asarray([entry[2] for entry in entries], dtype=np.float64)
        if float(weights.sum()) <= 1e-12:
            weights[:] = 1.0
        weights /= weights.sum()
        rotations = Rotation.from_matrix(np.stack([entry[0][:3, :3] for entry in entries]))
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotations.mean(weights=weights).as_matrix()
        pose[:3, 3] = np.sum(
            np.stack([entry[0][:3, 3] for entry in entries]) * weights[:, None],
            axis=0,
        )
        merged_poses.append(pose)
        merged_intrinsics.append(
            np.sum(np.stack([entry[1] for entry in entries]) * weights[:, None], axis=0)
        )
        depth_scales.append(float(np.dot(weights, [entry[3] for entry in entries])))
        primary_chunks.append(int(entries[int(np.argmax(weights))][4]))

    intrinsics = np.stack(merged_intrinsics)
    calibration = Calibration(
        indices=expected,
        c2w=np.stack(merged_poses),
        w2c=np.linalg.inv(np.stack(merged_poses)),
        intrinsics=intrinsics,
        K=np.stack([intrinsics_matrix(row) for row in intrinsics]),
        camera_types=tuple("PINHOLE" for _ in expected),
    )
    return calibration, np.asarray(depth_scales), np.asarray(primary_chunks, dtype=np.int32)


_PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}


def read_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the binary vertex PLY emitted by :func:`write_ply`."""

    properties: list[tuple[str, str]] = []
    vertex_count: int | None = None
    in_vertex = False
    with Path(path).open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"Truncated PLY header: {path}")
            line = raw_line.decode("ascii").strip()
            if line == "format binary_little_endian 1.0":
                continue
            if line.startswith("format "):
                raise ValueError(f"Only binary little-endian PLY is supported: {path}")
            if line.startswith("element "):
                _, element_name, count = line.split()
                in_vertex = element_name == "vertex"
                if in_vertex:
                    vertex_count = int(count)
            elif line.startswith("property ") and in_vertex:
                tokens = line.split()
                if len(tokens) != 3 or tokens[1] not in _PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property '{line}' in {path}")
                properties.append((tokens[2], _PLY_TYPES[tokens[1]]))
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        records = np.fromfile(handle, dtype=np.dtype(properties), count=vertex_count)
    if len(records) != vertex_count:
        raise ValueError(f"Expected {vertex_count} vertices, read {len(records)} from {path}")
    points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float32)
    colors = np.column_stack((records["red"], records["green"], records["blue"])).astype(np.uint8)
    observations = (
        np.asarray(records["observations"], dtype=np.int32)
        if "observations" in records.dtype.names
        else np.ones(vertex_count, dtype=np.int32)
    )
    return points, colors, observations


def _chunk_map_is_current(map_dir: Path, options: MapOptions, artifact: ArtifactSet) -> bool:
    metadata_path = map_dir / "map_metadata.json"
    required = (map_dir / "global_rgb_map_world.ply", map_dir / "calibration.npz", metadata_path)
    if not all(path.exists() for path in required):
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        depth_metadata = (
            json.loads(artifact.depth_metadata.read_text()) if artifact.depth_metadata.exists() else None
        )
        return metadata.get("map_options") == asdict(options) and metadata.get("dense_depth") == depth_metadata
    except (OSError, ValueError, TypeError):
        return False


def _write_global_trajectory(path: Path, calibration: Calibration, fps: float, level_xyz: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "timestamp_seconds",
                "world_x",
                "world_y",
                "world_z",
                "level_x",
                "level_y_down",
                "level_z",
            ]
        )
        for frame_index, pose, level in zip(calibration.indices, calibration.c2w, level_xyz, strict=True):
            writer.writerow([int(frame_index), float(frame_index / fps), *pose[:3, 3].tolist(), *level.tolist()])


def _write_frame_assignment(
    path: Path,
    indices: np.ndarray,
    primary_chunks: np.ndarray,
    blended_depth_scales: np.ndarray,
    chunk_scales: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "primary_chunk",
                "primary_chunk_depth_scale_to_global",
                "overlap_blended_depth_scale_to_global",
            ]
        )
        for frame_index, chunk_index, blended_scale in zip(
            indices, primary_chunks, blended_depth_scales, strict=True
        ):
            writer.writerow(
                [
                    int(frame_index),
                    int(chunk_index),
                    float(chunk_scales[int(chunk_index)]),
                    float(blended_scale),
                ]
            )


def stitch_chunk_results(
    chunks: list[ChunkResult],
    output_dir: Path,
    *,
    total_frames: int,
    fps: float,
    image_wh: tuple[int, int],
    overlap_frames: int,
    map_options: MapOptions,
    thresholds: StitchThresholds,
) -> dict[str, Any]:
    """Align all chunks, merge trajectories/clouds, and render global QA."""

    if not chunks:
        raise ValueError("No chunks to stitch")
    stitch_metrics: list[dict[str, Any]] = []
    chunks[0].transform = SimilarityTransform.identity()
    for previous, current in zip(chunks[:-1], chunks[1:], strict=True):
        stitch_metrics.append(_align_adjacent_chunks(previous, current, thresholds))

    calibration, depth_scales, primary_chunks = merge_chunk_calibrations(
        chunks,
        total_frames=total_frames,
        overlap_frames=overlap_frames,
    )
    for metrics in stitch_metrics:
        boundary = (int(metrics["first_shared_frame"]) + int(metrics["last_shared_frame"])) // 2
        if boundary > 0:
            metrics["blended_boundary_frame"] = boundary
            metrics["blended_boundary_translation_step"] = float(
                np.linalg.norm(calibration.c2w[boundary, :3, 3] - calibration.c2w[boundary - 1, :3, 3])
            )
            relative_rotation = (
                calibration.c2w[boundary, :3, :3]
                @ calibration.c2w[boundary - 1, :3, :3].T
            )
            metrics["blended_boundary_rotation_step_deg"] = float(
                _rotation_angle_degrees(relative_rotation[None])[0]
            )
    output_dir = resolve_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_calibration(calibration, output_dir, fps=fps, image_wh=image_wh)
    chunk_scales = np.asarray([chunk.transform.scale for chunk in chunks], dtype=np.float64)
    _write_frame_assignment(
        output_dir / "per_frame_chunk_assignment.csv",
        calibration.indices,
        primary_chunks,
        depth_scales,
        chunk_scales,
    )

    with tempfile.TemporaryDirectory(prefix="stitched_voxels_", dir=output_dir) as temp_dir:
        accumulator = VoxelAccumulator(map_options.voxel_size, Path(temp_dir))
        for chunk in chunks:
            points, colors, observations = read_binary_ply(chunk.map_dir / "global_rgb_map_world.ply")
            transformed = chunk.transform.transform_points(points)
            accumulator.add_cloud(transformed, colors, observations=observations)
        cloud = accumulator.finalize(map_options.min_voxel_observations)
        if len(cloud.points) == 0 and map_options.min_voxel_observations > 1:
            logger.warning("No stitched voxel met min observations; falling back to 1")
            cloud = accumulator.finalize(1)
    if len(cloud.points) == 0:
        raise RuntimeError("Stitched chunk point cloud is empty")
    write_ply(output_dir / "global_rgb_map_world.ply", cloud.points, cloud.colors, cloud.observations)

    level = estimate_level_frame(calibration.c2w)
    points_level = level.transform_points(cloud.points).astype(np.float32)
    trajectory_world = calibration.c2w[:, :3, 3]
    trajectory_level = level.transform_points(trajectory_world)
    if map_options.floor_y is not None:
        floor_y = float(map_options.floor_y)
        floor_estimation = "manual"
    else:
        try:
            floor_y = estimate_floor_y(points_level, trajectory_level)
            floor_estimation = "histogram"
        except ValueError as exc:
            logger.warning("Floor detection failed (%s); using fallback camera height", exc)
            floor_y = float(np.median(trajectory_level[:, 1]) + map_options.fallback_camera_height)
            floor_estimation = "fallback_camera_height"
    write_ply(output_dir / "global_rgb_map_leveled.ply", points_level, cloud.colors, cloud.observations)
    raster = render_topdown(
        output_dir,
        points_level,
        cloud.colors,
        trajectory_level,
        floor_y=floor_y,
        min_height=map_options.topdown_min_height,
        max_height=map_options.topdown_max_height,
        resolution=map_options.topdown_resolution,
        max_size=map_options.max_raster_size,
    )
    _write_global_trajectory(output_dir / "trajectory.csv", calibration, fps, trajectory_level)

    transforms = []
    for chunk in chunks:
        item = {
            "chunk": asdict(chunk.spec),
            "name": chunk.spec.name,
            "artifact_root": str(chunk.artifact.root),
            "map_dir": str(chunk.map_dir),
            "local_to_global": chunk.transform.to_dict(),
            "depth_scale_to_global": float(chunk.transform.scale),
        }
        transforms.append(item)
    (output_dir / "chunk_transforms.json").write_text(json.dumps(transforms, indent=2) + "\n")
    (output_dir / "stitch_metrics.json").write_text(json.dumps(stitch_metrics, indent=2) + "\n")

    step_distances = np.linalg.norm(np.diff(trajectory_world, axis=0), axis=1)
    camera_heights = floor_y - trajectory_level[:, 1]
    tilt = pose_tilt_degrees(calibration.c2w, level.down)
    metrics: dict[str, Any] = {
        "total_frames": int(total_frames),
        "chunk_count": len(chunks),
        "overlap_frames": overlap_frames,
        "path_length_global_units": float(step_distances.sum()),
        "max_frame_translation_global_units": float(step_distances.max(initial=0.0)),
        "fused_voxels": int(len(cloud.points)),
        "camera_height_median_global_units": float(np.median(camera_heights)),
        "camera_height_std_global_units": float(np.std(camera_heights)),
        "camera_tilt_median_deg": float(np.median(tilt)),
        "floor_y_down_global_units": floor_y,
        "floor_estimation": floor_estimation,
        "topdown_meters_per_pixel_in_chunk0_scale": raster.meters_per_pixel,
        "stitches": stitch_metrics,
        "scale_anchor": "chunk 0; monocular metric scale is not guaranteed to be physically exact",
    }
    (output_dir / "quality_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    metadata = {
        "map_options": asdict(map_options),
        "stitch_thresholds": asdict(thresholds),
        "world_to_level": level.world_to_level.tolist(),
        "coordinate_convention": "OpenCV c2w; +x right, +y down, +z forward",
    }
    (output_dir / "map_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = [
        "# Chunked room-tour stitch report",
        "",
        "Open `topdown_with_trajectory.png` first. Inspect every boundary listed below for doubled walls.",
        "",
        "![Top-down map](topdown_with_trajectory.png)",
        "",
        f"- Chunks: {len(chunks)}",
        f"- Frames: {total_frames}",
        f"- Global path length (chunk-0 scale): {metrics['path_length_global_units']:.3f}",
        "",
        "## Adjacent-chunk alignment",
        "",
    ]
    for item in stitch_metrics:
        report.append(
            f"- `{item['previous_chunk']}` → `{item['current_chunk']}`: "
            f"scale={item['scale']:.5f}, position RMSE={item['position_rmse']:.4f}, "
            f"rotation median={item['rotation_median_deg']:.3f}°"
        )
    (output_dir / "quality_report.md").write_text("\n".join(report) + "\n")
    return metrics


def stitch_chunk_pose_results(
    chunks: list[ChunkResult],
    output_dir: Path,
    *,
    total_frames: int,
    fps: float,
    image_wh: tuple[int, int],
    overlap_frames: int,
    thresholds: StitchThresholds,
) -> dict[str, Any]:
    """Export a globally continuous all-frame trajectory without running dense DAv3."""

    if not chunks:
        raise ValueError("No chunks to stitch")
    chunks[0].transform = SimilarityTransform.identity()
    stitch_metrics = [
        _align_adjacent_chunks(previous, current, thresholds)
        for previous, current in zip(chunks[:-1], chunks[1:], strict=True)
    ]
    calibration, depth_scales, primary_chunks = merge_chunk_calibrations(
        chunks,
        total_frames=total_frames,
        overlap_frames=overlap_frames,
    )
    output_dir = resolve_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_calibration(calibration, output_dir, fps=fps, image_wh=image_wh)
    chunk_scales = np.asarray([chunk.transform.scale for chunk in chunks], dtype=np.float64)
    _write_frame_assignment(
        output_dir / "per_frame_chunk_assignment.csv",
        calibration.indices,
        primary_chunks,
        depth_scales,
        chunk_scales,
    )
    level = estimate_level_frame(calibration.c2w)
    trajectory_world = calibration.c2w[:, :3, 3]
    trajectory_level = level.transform_points(trajectory_world)
    _write_global_trajectory(output_dir / "trajectory.csv", calibration, fps, trajectory_level)
    transforms = [
        {
            "chunk": asdict(chunk.spec),
            "name": chunk.spec.name,
            "artifact_root": str(chunk.artifact.root),
            "local_to_global": chunk.transform.to_dict(),
            "depth_scale_to_global": float(chunk.transform.scale),
        }
        for chunk in chunks
    ]
    (output_dir / "chunk_transforms.json").write_text(json.dumps(transforms, indent=2) + "\n")
    (output_dir / "stitch_metrics.json").write_text(json.dumps(stitch_metrics, indent=2) + "\n")
    step_distances = np.linalg.norm(np.diff(trajectory_world, axis=0), axis=1)
    metrics = {
        "mode": "pose_only",
        "total_frames": int(total_frames),
        "chunk_count": len(chunks),
        "overlap_frames": int(overlap_frames),
        "path_length_global_units": float(step_distances.sum()),
        "max_frame_translation_global_units": float(step_distances.max(initial=0.0)),
        "stitches": stitch_metrics,
        "scale_anchor": "chunk 0; monocular metric scale is not guaranteed to be physically exact",
    }
    (output_dir / "pose_quality_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def _cuda_devices(depth_workers: int) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [item.strip() for item in visible.split(",") if item.strip()] if visible else []
    if not devices:
        devices = [str(index) for index in range(depth_workers)]
    if depth_workers > len(devices):
        raise ValueError(
            f"Requested {depth_workers} depth workers but only {len(devices)} CUDA devices are visible: {devices}"
        )
    return devices[:depth_workers]


def _run_chunk_depth_jobs(
    jobs: list[ChunkResult],
    *,
    input_video: Path,
    pipeline: str,
    artifact_root: Path,
    depth_options: DenseDepthOptions,
    depth_workers: int,
    loop_closure: bool,
    loop_experiment: str,
) -> None:
    if not jobs:
        return
    devices: queue.Queue[str] = queue.Queue()
    for device in _cuda_devices(depth_workers):
        devices.put(device)

    def run_one(chunk: ChunkResult) -> None:
        device = devices.get()
        try:
            _run_vipe(
                input_video,
                artifact_root,
                pipeline,
                False,
                start_frame=chunk.spec.start_frame,
                end_frame=chunk.spec.end_frame,
                artifact_name=chunk.spec.name,
                mode="depth",
                depth_options=depth_options,
                cuda_visible_device=device,
                loop_closure=loop_closure,
                loop_experiment=loop_experiment,
            )
        finally:
            devices.put(device)

    logger.info("Running dense DAv3 for %d chunks on %d GPU worker(s)", len(jobs), depth_workers)
    if depth_workers == 1:
        for chunk in jobs:
            run_one(chunk)
        return
    with ThreadPoolExecutor(max_workers=depth_workers) as executor:
        futures = [executor.submit(run_one, chunk) for chunk in jobs]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        first_error = next(
            (future.exception() for future in done if future.exception() is not None),
            None,
        )
        if first_error is not None:
            for future in pending:
                future.cancel()
            raise first_error
        for future in pending:
            future.result()


def run_chunked_roomtour(
    input_video: Path,
    output_root: Path,
    *,
    pipeline: str,
    chunk_frames: int,
    overlap_frames: int,
    map_options: MapOptions,
    thresholds: StitchThresholds,
    skip_inference: bool = False,
    skip_chunk_maps: bool = False,
    inference_mode: str = "full",
    depth_options: DenseDepthOptions | None = None,
    depth_workers: int = 1,
    loop_closure: bool = False,
    loop_experiment: str = "normal",
) -> dict[str, Any]:
    input_video = Path(input_video).resolve()
    output_root = resolve_output_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    video_info = probe_video(input_video)
    total_frames = count_video_frames(input_video)
    specs = plan_chunks(total_frames, chunk_frames, overlap_frames)
    artifact_root = output_root / "vipe_artifacts"
    chunk_root = output_root / "chunks"
    artifact_root.mkdir(parents=True, exist_ok=True)
    chunk_root.mkdir(parents=True, exist_ok=True)
    if inference_mode not in {"full", "pose", "depth"}:
        raise ValueError(f"Unsupported inference mode: {inference_mode}")
    if depth_workers < 1:
        raise ValueError("depth_workers must be >= 1")
    depth_options = depth_options or DenseDepthOptions.from_preset("quality")

    logger.info(
        "Chunking %d frames into %d chunks (max=%d, overlap=%d)",
        total_frames,
        len(specs),
        chunk_frames,
        overlap_frames,
    )
    results: list[ChunkResult] = []
    manifest: dict[str, Any] = {
        "input_video": str(input_video),
        "total_frames": total_frames,
        "fps": video_info["fps"],
        "pipeline": pipeline,
        "inference_mode": inference_mode,
        "dense_depth": depth_options.to_dict(),
        "depth_workers": depth_workers,
        "loop_closure": bool(loop_closure),
        "loop_experiment": loop_experiment,
        "chunk_frames": chunk_frames,
        "overlap_frames": overlap_frames,
        "map_options": asdict(map_options),
        "stitch_thresholds": asdict(thresholds),
        "chunks": [],
    }
    # Phase 1: finish and validate all-frame SLAM calibration for every chunk.
    # Dense DAv3 is not loaded in this phase.
    for spec in specs:
        artifact = ArtifactSet(artifact_root, spec.name)
        calibration_complete = artifact.slam_matches(
            loop_closure,
            loop_experiment,
        )
        if not calibration_complete:
            if skip_inference or inference_mode == "depth":
                raise FileNotFoundError(
                    "Missing SLAM checkpoint matching "
                    f"loop_closure={loop_closure}, "
                    f"loop_experiment={loop_experiment} for {spec.name}"
                )
            _run_vipe(
                input_video,
                artifact_root,
                pipeline,
                False,
                start_frame=spec.start_frame,
                end_frame=spec.end_frame,
                artifact_name=spec.name,
                mode="pose",
                loop_closure=loop_closure,
                loop_experiment=loop_experiment,
            )
        else:
            logger.info("Complete SLAM checkpoint already exists for %s", spec.name)
        artifact.validate_calibration()

        map_dir = chunk_root / spec.name
        calibration = load_calibration(artifact)
        if len(calibration.indices) != spec.frame_count:
            raise RuntimeError(
                f"{spec.name} expected {spec.frame_count} calibrated frames, got {len(calibration.indices)}"
            )
        result = ChunkResult(spec, artifact, map_dir, calibration, SimilarityTransform.identity())
        if results:
            # Reject a bad pose boundary before spending hours on dense depth.
            _align_adjacent_chunks(results[-1], result, thresholds)
        results.append(result)
        manifest["chunks"].append(
            {
                **asdict(spec),
                "name": spec.name,
                "artifact_root": str(artifact_root),
                "map_dir": str(map_dir),
            }
        )
        (output_root / "chunk_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if inference_mode == "pose":
        metrics = stitch_chunk_pose_results(
            results,
            output_root / "global",
            total_frames=total_frames,
            fps=float(video_info["fps"]),
            image_wh=(int(video_info["width"]), int(video_info["height"])),
            overlap_frames=overlap_frames,
            thresholds=thresholds,
        )
        manifest["global_output"] = str(output_root / "global")
        manifest["metrics"] = metrics
        (output_root / "chunk_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    # Phase 2: DAv3 is independent across chunks after pose alignment, so run
    # one subprocess per available GPU. Existing matching depth is reused.
    depth_jobs = [
        chunk
        for chunk in results
        if not chunk.artifact.depth_matches(
            depth_options,
            loop_closure=loop_closure,
            loop_experiment=loop_experiment,
        )
    ]
    if depth_jobs and skip_inference:
        missing = ", ".join(chunk.spec.name for chunk in depth_jobs)
        raise FileNotFoundError(f"Missing or mismatched dense-depth artifacts: {missing}")
    _run_chunk_depth_jobs(
        depth_jobs,
        input_video=input_video,
        pipeline=pipeline,
        artifact_root=artifact_root,
        depth_options=depth_options,
        depth_workers=depth_workers,
        loop_closure=loop_closure,
        loop_experiment=loop_experiment,
    )

    # Phase 3: build per-chunk RGB-D maps only after all requested depth is
    # complete. This keeps GPU depth workers independent from CPU map fusion.
    for result in results:
        result.artifact.validate()
        if not _chunk_map_is_current(result.map_dir, map_options, result.artifact):
            if skip_chunk_maps:
                raise FileNotFoundError(f"Missing or stale chunk map: {result.map_dir}")
            build_map(
                result.artifact,
                result.map_dir,
                map_options,
                source_time_offset=result.spec.start_frame / float(video_info["fps"]),
            )
        else:
            logger.info("Current chunk map already exists for %s", result.spec.name)

    metrics = stitch_chunk_results(
        results,
        output_root / "global",
        total_frames=total_frames,
        fps=float(video_info["fps"]),
        image_wh=(int(video_info["width"]), int(video_info["height"])),
        overlap_frames=overlap_frames,
        map_options=map_options,
        thresholds=thresholds,
    )
    manifest["global_output"] = str(output_root / "global")
    manifest["metrics"] = metrics
    (output_root / "chunk_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
