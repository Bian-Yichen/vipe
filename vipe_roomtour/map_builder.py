# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a leveled global RGB-D map from native ViPE artifacts."""

from __future__ import annotations

import csv
import json
import logging
import pickle
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactSet, export_calibration, load_calibration, write_ply
from .fusion import VoxelAccumulator
from .geometry import (
    backproject_pinhole,
    estimate_floor_y,
    estimate_level_frame,
    pose_tilt_degrees,
    scale_intrinsics,
)
from .topdown import render_topdown

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MapOptions:
    frame_step: int = 5
    pixel_stride: int = 4
    voxel_size: float = 0.03
    min_voxel_observations: int = 2
    min_depth: float = 0.15
    max_depth: float = 15.0
    depth_edge_threshold: float = 0.08
    topdown_resolution: float = 0.025
    topdown_min_height: float = -0.10
    topdown_max_height: float = 2.20
    max_raster_size: int = 4096
    floor_y: float | None = None
    fallback_camera_height: float = 1.6

    def __post_init__(self) -> None:
        if self.frame_step < 1 or self.pixel_stride < 1:
            raise ValueError("frame_step and pixel_stride must be >= 1")
        if self.voxel_size <= 0 or self.topdown_resolution <= 0:
            raise ValueError("voxel and top-down resolutions must be positive")
        if self.max_depth <= self.min_depth:
            raise ValueError("max_depth must be greater than min_depth")
        if self.topdown_max_height <= self.topdown_min_height:
            raise ValueError("topdown_max_height must be greater than topdown_min_height")


def probe_video(path: Path) -> dict[str, Any]:
    """Return width, height, fps, duration and frame count using ffprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe was not found; install FFmpeg first") from exc
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream.get("avg_frame_rate", "0/1").split("/")
    fps = float(numerator) / max(float(denominator), 1.0)
    if fps <= 0:
        raise RuntimeError(f"Could not determine a positive FPS for {path}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration_seconds": float(stream.get("duration") or 0.0),
        "declared_frames": int(stream["nb_frames"]) if stream.get("nb_frames") not in (None, "N/A") else None,
    }


def _scalar(value: Any) -> float | None:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.size == 1 and np.isfinite(array.reshape(-1)[0]):
            return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        pass
    return None


def _read_ba_residual(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            info = pickle.load(handle)  # noqa: S301 - file is a locally generated VIPE artifact.
        value = info.get("ba_residual") if isinstance(info, dict) else getattr(info, "ba_residual", None)
        return _scalar(value)
    except Exception as exc:  # Metadata is optional and has changed between VIPE releases.
        logger.warning("Could not read optional VIPE info %s: %s", path, exc)
        return None


def _write_trajectory(path: Path, indices: np.ndarray, times: np.ndarray, c2w: np.ndarray, level_xyz: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_seconds", "world_x", "world_y", "world_z", "level_x", "level_y_down", "level_z"])
        for frame, timestamp, pose, level in zip(indices, times, c2w, level_xyz, strict=True):
            writer.writerow([int(frame), float(timestamp), *pose[:3, 3].tolist(), *level.tolist()])


def _write_report(path: Path, artifact: ArtifactSet, metrics: dict[str, Any]) -> None:
    lines = [
        f"# Room-tour map report: `{artifact.name}`",
        "",
        "Open `topdown_with_trajectory.png` first. A good single-floor result normally has coherent walls/furniture, "
        "few duplicated room outlines, a smooth trajectory, and nearly constant camera height.",
        "",
        "![Top-down map](topdown_with_trajectory.png)",
        "",
        "## Quick metrics",
        "",
        f"- Calibrated frames: {metrics['calibrated_frames']}",
        f"- Fused RGB voxels: {metrics['fused_voxels']}",
        f"- Path length: {metrics['path_length_m']:.3f} m",
        f"- Camera height median/std: {metrics['camera_height_median_m']:.3f} / {metrics['camera_height_std_m']:.3f} m",
        f"- Camera tilt median/max: {metrics['camera_tilt_median_deg']:.3f} / {metrics['camera_tilt_max_deg']:.3f} deg",
        f"- Multi-frame voxel ratio: {metrics['multi_frame_voxel_ratio']:.3f}",
        "",
        "## Coordinate convention",
        "",
        "`calibration.npz` contains original-resolution `[fx, fy, cx, cy]`, `K`, OpenCV `c2w`, and `w2c` for every calibrated frame. "
        "Camera axes are +x right, +y down, +z forward.",
    ]
    path.write_text("\n".join(lines) + "\n")


def build_map(
    artifact: ArtifactSet,
    output_dir: Path,
    options: MapOptions,
    *,
    source_time_offset: float = 0.0,
) -> dict[str, Any]:
    """Build all calibration and global-map deliverables for one artifact set."""

    from vipe.utils.io import read_depth_artifacts, read_rgb_artifacts

    artifact.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video = probe_video(artifact.rgb)
    calibration = load_calibration(artifact)
    fps = float(video["fps"])
    export_calibration(
        calibration,
        output_dir,
        fps=fps,
        source_time_offset=source_time_offset,
        image_wh=(video["width"], video["height"]),
    )

    pose_row = {int(index): row for row, index in enumerate(calibration.indices)}
    processed_frames = 0
    input_depth_wh: tuple[int, int] | None = None
    with tempfile.TemporaryDirectory(prefix="voxel_chunks_", dir=output_dir) as temp_dir:
        accumulator = VoxelAccumulator(options.voxel_size, Path(temp_dir))
        rgb_iterator = iter(read_rgb_artifacts(artifact.rgb))
        rgb_item = next(rgb_iterator, None)
        for depth_index, depth_tensor in read_depth_artifacts(artifact.depth):
            while rgb_item is not None and rgb_item[0] < depth_index:
                rgb_item = next(rgb_iterator, None)
            if rgb_item is None:
                break
            if rgb_item[0] != depth_index or depth_index not in pose_row:
                continue
            if depth_index % options.frame_step != 0:
                continue
            row = pose_row[int(depth_index)]
            rgb = np.asarray(rgb_item[1].cpu().numpy())
            depth = np.asarray(depth_tensor.cpu().numpy(), dtype=np.float32)
            rgb_h, rgb_w = rgb.shape[:2]
            depth_h, depth_w = depth.shape
            input_depth_wh = (depth_w, depth_h)
            intrinsics_depth = scale_intrinsics(
                calibration.intrinsics[row], (rgb_w, rgb_h), (depth_w, depth_h)
            )
            points, u, v = backproject_pinhole(
                depth,
                intrinsics_depth,
                calibration.c2w[row],
                stride=options.pixel_stride,
                min_depth=options.min_depth,
                max_depth=options.max_depth,
                edge_threshold=options.depth_edge_threshold,
            )
            if len(points) == 0:
                continue
            rgb_u = np.clip(
                np.rint((u + 0.5) * rgb_w / depth_w - 0.5).astype(np.int64), 0, rgb_w - 1
            )
            rgb_v = np.clip(
                np.rint((v + 0.5) * rgb_h / depth_h - 0.5).astype(np.int64), 0, rgb_h - 1
            )
            colors = np.clip(np.rint(rgb[rgb_v, rgb_u] * 255.0), 0, 255).astype(np.uint8)
            accumulator.add_frame(points, colors)
            processed_frames += 1

        cloud = accumulator.finalize(options.min_voxel_observations)
        used_min_observations = options.min_voxel_observations
        if len(cloud.points) == 0 and options.min_voxel_observations > 1:
            logger.warning("No voxel met min observations=%d; falling back to 1", options.min_voxel_observations)
            cloud = accumulator.finalize(1)
            used_min_observations = 1

    if len(cloud.points) == 0:
        raise RuntimeError("RGB-D fusion produced no points; inspect depth range and artifact alignment")
    write_ply(output_dir / "global_rgb_map_world.ply", cloud.points, cloud.colors, cloud.observations)

    level = estimate_level_frame(calibration.c2w)
    points_level = level.transform_points(cloud.points).astype(np.float32)
    trajectory_world = calibration.c2w[:, :3, 3]
    trajectory_level = level.transform_points(trajectory_world)
    floor_estimation = "histogram"
    if options.floor_y is not None:
        floor_y = float(options.floor_y)
        floor_estimation = "manual"
    else:
        try:
            floor_y = estimate_floor_y(points_level, trajectory_level)
        except ValueError as exc:
            logger.warning("Floor detection failed (%s); using fallback camera height %.2f m", exc, options.fallback_camera_height)
            floor_y = float(np.median(trajectory_level[:, 1]) + options.fallback_camera_height)
            floor_estimation = "fallback_camera_height"
    write_ply(output_dir / "global_rgb_map_leveled.ply", points_level, cloud.colors, cloud.observations)
    raster = render_topdown(
        output_dir,
        points_level,
        cloud.colors,
        trajectory_level,
        floor_y=floor_y,
        min_height=options.topdown_min_height,
        max_height=options.topdown_max_height,
        resolution=options.topdown_resolution,
        max_size=options.max_raster_size,
    )

    timestamps = calibration.indices / fps
    _write_trajectory(output_dir / "trajectory.csv", calibration.indices, timestamps, calibration.c2w, trajectory_level)
    camera_heights = floor_y - trajectory_level[:, 1]
    tilt = pose_tilt_degrees(calibration.c2w, level.down)
    step_distances = np.linalg.norm(np.diff(trajectory_world, axis=0), axis=1)
    metrics: dict[str, Any] = {
        "artifact_name": artifact.name,
        "calibrated_frames": int(len(calibration.indices)),
        "processed_rgbd_frames": int(processed_frames),
        "fused_voxels": int(len(cloud.points)),
        "voxel_size_m": options.voxel_size,
        "min_voxel_observations_used": used_min_observations,
        "multi_frame_voxel_ratio": float(np.mean(cloud.observations >= 2)),
        "median_voxel_frame_observations": float(np.median(cloud.observations)),
        "path_length_m": float(step_distances.sum()),
        "max_frame_translation_m": float(step_distances.max(initial=0.0)),
        "camera_height_median_m": float(np.median(camera_heights)),
        "camera_height_std_m": float(np.std(camera_heights)),
        "camera_height_range_m": float(np.ptp(camera_heights)),
        "camera_tilt_median_deg": float(np.median(tilt)),
        "camera_tilt_max_deg": float(np.max(tilt)),
        "floor_y_down_m": floor_y,
        "floor_estimation": floor_estimation,
        "ba_residual": _read_ba_residual(artifact.info),
        "rgb_width": video["width"],
        "rgb_height": video["height"],
        "depth_width": input_depth_wh[0] if input_depth_wh else None,
        "depth_height": input_depth_wh[1] if input_depth_wh else None,
        "fps": fps,
        "topdown_meters_per_pixel": raster.meters_per_pixel,
        "topdown_width": raster.width,
        "topdown_height": raster.height,
    }
    (output_dir / "quality_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    metadata = {
        "source_artifacts": str(artifact.root),
        "artifact_name": artifact.name,
        "map_options": asdict(options),
        "world_to_level": level.world_to_level.tolist(),
        "level_axes_in_world": {
            "right": level.right.tolist(),
            "down": level.down.tolist(),
            "forward": level.forward.tolist(),
        },
        "floor_y_down_m": floor_y,
    }
    (output_dir / "map_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _write_report(output_dir / "quality_report.md", artifact, metrics)
    return metrics
