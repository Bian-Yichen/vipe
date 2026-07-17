# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from vipe_roomtour.geometry import (
    backproject_pinhole,
    estimate_floor_y,
    estimate_level_frame,
    scale_intrinsics,
)


def test_scale_intrinsics_handles_nonuniform_resize():
    intr = scale_intrinsics(np.array([1000.0, 900.0, 500.0, 400.0]), (1000, 800), (500, 200))
    np.testing.assert_allclose(intr, [500.0, 225.0, 250.0, 100.0])


def test_backproject_identity_pose():
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    points, u, v = backproject_pinhole(
        depth,
        np.array([2.0, 2.0, 1.0, 1.0]),
        np.eye(4),
        edge_threshold=None,
    )
    center = np.flatnonzero((u == 1) & (v == 1))[0]
    np.testing.assert_allclose(points[center], [0.0, 0.0, 2.0])
    corner = np.flatnonzero((u == 0) & (v == 0))[0]
    np.testing.assert_allclose(points[corner], [-1.0, -1.0, 2.0])


def test_level_frame_and_floor_histogram():
    poses = np.repeat(np.eye(4)[None], 20, axis=0)
    poses[:, 0, 3] = np.linspace(0, 2, len(poses))
    level = estimate_level_frame(poses)
    trajectory = level.transform_points(poses[:, :3, 3])
    rng = np.random.default_rng(4)
    floor = np.column_stack(
        (
            rng.uniform(-2, 2, 5000),
            rng.normal(1.55, 0.01, 5000),
            rng.uniform(-2, 2, 5000),
        )
    )
    floor_y = estimate_floor_y(floor, trajectory)
    assert abs(floor_y - 1.55) < 0.05
