# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np

from vipe_roomtour.artifacts import write_ply
from vipe_roomtour.fusion import VoxelAccumulator
from vipe_roomtour.topdown import render_topdown


def test_voxel_observations_count_distinct_frames(tmp_path: Path):
    accumulator = VoxelAccumulator(0.1, tmp_path / "chunks", max_records=2)
    accumulator.add_frame(np.array([[0.01, 0.01, 0.01], [0.02, 0.01, 0.01]]), np.array([[100, 0, 0], [200, 0, 0]]))
    accumulator.add_frame(np.array([[0.03, 0.01, 0.01]]), np.array([[150, 0, 0]]))
    cloud = accumulator.finalize(min_observations=2)
    assert len(cloud.points) == 1
    assert cloud.observations.tolist() == [2]
    np.testing.assert_allclose(cloud.colors[0], [150, 0, 0], atol=1)


def test_aggregated_cloud_support_is_preserved(tmp_path: Path):
    accumulator = VoxelAccumulator(0.1, tmp_path / "aggregate")
    accumulator.add_cloud(
        np.array([[0.01, 0.01, 0.01], [0.02, 0.01, 0.01]]),
        np.array([[100, 0, 0], [200, 0, 0]]),
        observations=np.array([3, 5]),
        samples=np.array([6, 10]),
    )
    cloud = accumulator.finalize(min_observations=2)
    assert cloud.observations.tolist() == [8]
    assert cloud.samples.tolist() == [16]
    np.testing.assert_allclose(cloud.points[0], [0.01625, 0.01, 0.01], atol=1e-6)
    np.testing.assert_allclose(cloud.colors[0], [163, 0, 0], atol=1)


def test_topdown_and_ply_are_written(tmp_path: Path):
    x, z = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    points = np.column_stack((x.ravel(), np.full(x.size, 1.6), z.ravel()))
    colors = np.tile(np.array([[80, 140, 200]], dtype=np.uint8), (len(points), 1))
    trajectory = np.column_stack((np.linspace(-0.8, 0.8, 20), np.zeros(20), np.zeros(20)))
    render_topdown(tmp_path, points, colors, trajectory, floor_y=1.6, resolution=0.05)
    write_ply(tmp_path / "map.ply", points, colors)
    for name in ("topdown_rgb.png", "topdown_occupancy.png", "topdown_with_trajectory.png", "map.ply"):
        assert (tmp_path / name).stat().st_size > 0
