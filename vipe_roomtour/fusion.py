# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunked voxel fusion for long room-tour videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FusedCloud:
    points: np.ndarray
    colors: np.ndarray
    observations: np.ndarray
    samples: np.ndarray


def _reduce_records(
    coords: np.ndarray,
    point_sums: np.ndarray,
    color_sums: np.ndarray,
    sample_counts: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse = np.unique(coords, axis=0, return_inverse=True)
    n = len(unique)
    reduced_points = np.zeros((n, 3), dtype=np.float64)
    reduced_colors = np.zeros((n, 3), dtype=np.float64)
    reduced_samples = np.zeros(n, dtype=np.int64)
    reduced_observations = np.zeros(n, dtype=np.int64)
    np.add.at(reduced_points, inverse, point_sums)
    np.add.at(reduced_colors, inverse, color_sums)
    np.add.at(reduced_samples, inverse, sample_counts)
    np.add.at(reduced_observations, inverse, observations)
    return unique, reduced_points, reduced_colors, reduced_samples, reduced_observations


class VoxelAccumulator:
    """Fuse points without retaining every RGB-D sample in memory.

    A frame first contributes at most once to a voxel.  Consequently the
    ``observations`` output means number of different frames, which is more
    useful for rejecting one-frame depth artifacts than raw pixel count.
    """

    def __init__(self, voxel_size: float, work_dir: Path, max_records: int = 1_000_000):
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.voxel_size = float(voxel_size)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_records = int(max_records)
        self._records = 0
        self._coords: list[np.ndarray] = []
        self._points: list[np.ndarray] = []
        self._colors: list[np.ndarray] = []
        self._samples: list[np.ndarray] = []
        self._observations: list[np.ndarray] = []
        self._chunks: list[Path] = []

    def add_frame(self, points: np.ndarray, colors: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64)
        colors = np.asarray(colors, dtype=np.float64)
        if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points and colors must both be Nx3")
        valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
        points = points[valid]
        colors = colors[valid]
        if len(points) == 0:
            return
        coords = np.floor(points / self.voxel_size).astype(np.int64)
        ones = np.ones(len(points), dtype=np.int64)
        unique, point_sums, color_sums, samples, _ = _reduce_records(
            coords, points, colors, ones, ones
        )
        self._coords.append(unique)
        self._points.append(point_sums)
        self._colors.append(color_sums)
        self._samples.append(samples)
        self._observations.append(np.ones(len(unique), dtype=np.int64))
        self._records += len(unique)
        if self._records >= self.max_records:
            self._flush()

    def add_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        observations: np.ndarray | None = None,
        samples: np.ndarray | None = None,
    ) -> None:
        """Merge an already-aggregated voxel cloud without expanding samples.

        This is used by chunk stitching: each input PLY is already a fused
        per-chunk cloud, so its averaged point/color must be weighted by its
        support before the transformed chunks are voxelized again globally.
        """

        points = np.asarray(points, dtype=np.float64)
        colors = np.asarray(colors, dtype=np.float64)
        if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points and colors must both be Nx3")
        if observations is None:
            observations = np.ones(len(points), dtype=np.int64)
        if samples is None:
            samples = observations
        observations = np.asarray(observations, dtype=np.int64)
        samples = np.asarray(samples, dtype=np.int64)
        if observations.shape != (len(points),) or samples.shape != (len(points),):
            raise ValueError("observations and samples must have one value per point")

        valid = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(colors).all(axis=1)
            & (observations > 0)
            & (samples > 0)
        )
        points = points[valid]
        colors = colors[valid]
        observations = observations[valid]
        samples = samples[valid]
        if len(points) == 0:
            return

        coords = np.floor(points / self.voxel_size).astype(np.int64)
        reduced = _reduce_records(
            coords,
            points * samples[:, None],
            colors * samples[:, None],
            samples,
            observations,
        )
        self._coords.append(reduced[0])
        self._points.append(reduced[1])
        self._colors.append(reduced[2])
        self._samples.append(reduced[3])
        self._observations.append(reduced[4])
        self._records += len(reduced[0])
        if self._records >= self.max_records:
            self._flush()

    def _flush(self) -> None:
        if self._records == 0:
            return
        reduced = _reduce_records(
            np.concatenate(self._coords),
            np.concatenate(self._points),
            np.concatenate(self._colors),
            np.concatenate(self._samples),
            np.concatenate(self._observations),
        )
        path = self.work_dir / f"voxel_chunk_{len(self._chunks):05d}.npz"
        np.savez_compressed(
            path,
            coords=reduced[0],
            points=reduced[1],
            colors=reduced[2],
            samples=reduced[3],
            observations=reduced[4],
        )
        self._chunks.append(path)
        self._coords.clear()
        self._points.clear()
        self._colors.clear()
        self._samples.clear()
        self._observations.clear()
        self._records = 0

    def finalize(self, min_observations: int = 2) -> FusedCloud:
        self._flush()
        if not self._chunks:
            return FusedCloud(
                np.empty((0, 3), np.float32),
                np.empty((0, 3), np.uint8),
                np.empty(0, np.int32),
                np.empty(0, np.int32),
            )
        arrays = [np.load(path) for path in self._chunks]
        reduced = _reduce_records(
            np.concatenate([a["coords"] for a in arrays]),
            np.concatenate([a["points"] for a in arrays]),
            np.concatenate([a["colors"] for a in arrays]),
            np.concatenate([a["samples"] for a in arrays]),
            np.concatenate([a["observations"] for a in arrays]),
        )
        _, point_sums, color_sums, samples, observations = reduced
        keep = observations >= min_observations
        denom = np.maximum(samples[keep, None], 1)
        points = (point_sums[keep] / denom).astype(np.float32)
        colors = np.clip(np.rint(color_sums[keep] / denom), 0, 255).astype(np.uint8)
        return FusedCloud(
            points,
            colors,
            observations[keep].astype(np.int32),
            samples[keep].astype(np.int32),
        )
