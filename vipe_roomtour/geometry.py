# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry helpers for ViPE camera-to-world poses and RGB-D backprojection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LevelFrame:
    """A gravity-aligned coordinate frame.

    Coordinates are x=right, y=down and z=forward.  This deliberately keeps
    the OpenCV y-down convention used by ViPE; height above the floor is
    therefore ``floor_y - y``.
    """

    world_to_level: np.ndarray
    right: np.ndarray
    down: np.ndarray
    forward: np.ndarray
    origin: np.ndarray

    def transform_points(self, points_world: np.ndarray) -> np.ndarray:
        points_world = np.asarray(points_world, dtype=np.float64)
        return points_world @ self.world_to_level[:3, :3].T + self.world_to_level[:3, 3]


def intrinsics_matrix(intrinsics: np.ndarray) -> np.ndarray:
    """Convert ``[fx, fy, cx, cy]`` to a 3x3 K matrix."""

    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_intrinsics(intrinsics: np.ndarray, source_wh: tuple[int, int], target_wh: tuple[int, int]) -> np.ndarray:
    """Scale pixel intrinsics between image grids, independently in x/y."""

    source_w, source_h = source_wh
    target_w, target_h = target_wh
    if source_w <= 0 or source_h <= 0 or target_w <= 0 or target_h <= 0:
        raise ValueError("Image dimensions must be positive")
    scaled = np.asarray(intrinsics, dtype=np.float64).copy()
    scaled[[0, 2]] *= target_w / source_w
    scaled[[1, 3]] *= target_h / source_h
    return scaled


def backproject_pinhole(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    *,
    stride: int = 1,
    min_depth: float = 0.1,
    max_depth: float = 20.0,
    edge_threshold: float | None = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backproject a depth image into world coordinates.

    Returns world points plus the integer ``u`` and ``v`` samples used.  Depth
    is interpreted as OpenCV z-depth, which matches ViPE's artifact exporter.
    """

    from scipy.ndimage import maximum_filter, minimum_filter

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth, got {depth.shape}")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    if edge_threshold is not None and edge_threshold > 0:
        low_input = np.where(valid, depth, np.inf)
        high_input = np.where(valid, depth, -np.inf)
        local_min = minimum_filter(low_input, size=3, mode="nearest")
        local_max = maximum_filter(high_input, size=3, mode="nearest")
        spread = (local_max - local_min) / np.maximum(depth, 1e-6)
        valid &= np.isfinite(spread) & (spread <= edge_threshold)

    vv, uu = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    zz = depth[::stride, ::stride]
    keep = valid[::stride, ::stride]
    u = uu[keep].astype(np.int32)
    v = vv[keep].astype(np.int32)
    z = zz[keep].astype(np.float64)
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64)
    camera = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))

    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.shape != (4, 4):
        raise ValueError(f"Expected 4x4 c2w, got {c2w.shape}")
    world = camera @ c2w[:3, :3].T + c2w[:3, 3]
    return world.astype(np.float32), u, v


def estimate_level_frame(c2w: np.ndarray) -> LevelFrame:
    """Estimate gravity direction from the consensus of camera y-down axes."""

    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or len(c2w) == 0:
        raise ValueError("c2w must be a non-empty Nx4x4 array")

    down_axes = c2w[:, :3, 1].copy()
    reference = down_axes[0]
    down_axes[np.einsum("ij,j->i", down_axes, reference) < 0] *= -1
    scatter = down_axes.T @ down_axes
    _, vectors = np.linalg.eigh(scatter)
    down = vectors[:, -1]
    if np.dot(down, down_axes.mean(axis=0)) < 0:
        down *= -1
    down /= np.linalg.norm(down)

    forward_candidates = c2w[:, :3, 2]
    forward = forward_candidates.mean(axis=0)
    forward -= down * np.dot(forward, down)
    if np.linalg.norm(forward) < 1e-6:
        forward = c2w[0, :3, 2] - down * np.dot(c2w[0, :3, 2], down)
    if np.linalg.norm(forward) < 1e-6:
        seed = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(seed, down)) > 0.9:
            seed = np.array([1.0, 0.0, 0.0])
        forward = seed - down * np.dot(seed, down)
    forward /= np.linalg.norm(forward)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    forward = np.cross(right, down)
    forward /= np.linalg.norm(forward)

    origin = c2w[0, :3, 3].copy()
    rotation = np.stack((right, down, forward), axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ origin
    return LevelFrame(transform, right, down, forward, origin)


def estimate_floor_y(
    points_level: np.ndarray,
    trajectory_level: np.ndarray,
    *,
    min_camera_height: float = 0.5,
    max_camera_height: float = 2.6,
    bin_size: float = 0.025,
) -> float:
    """Estimate the floor's y-down coordinate from a smoothed height histogram."""

    from scipy.ndimage import gaussian_filter1d

    points_level = np.asarray(points_level)
    camera_y = float(np.median(np.asarray(trajectory_level)[:, 1]))
    low = camera_y + min_camera_height
    high = camera_y + max_camera_height
    y = points_level[:, 1]
    y = y[np.isfinite(y) & (y >= low) & (y <= high)]
    if len(y) < 100:
        raise ValueError("Not enough plausible floor points; adjust floor-height limits")
    bins = max(16, int(np.ceil((high - low) / bin_size)))
    hist, edges = np.histogram(y, bins=bins, range=(low, high))
    smooth = gaussian_filter1d(hist.astype(np.float64), sigma=max(1.0, 0.05 / bin_size))
    idx = int(np.argmax(smooth))
    return float((edges[idx] + edges[idx + 1]) * 0.5)


def pose_tilt_degrees(c2w: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Angle between every camera's y axis and the estimated down direction."""

    axes = np.asarray(c2w)[:, :3, 1]
    dots = np.abs(axes @ np.asarray(down)) / np.maximum(np.linalg.norm(axes, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
