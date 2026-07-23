from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from vipe.slam.components.hinge_correction import (
    apply_hinge_correction,
    detect_rotation_hinge,
)


def _turnaround_trajectory() -> np.ndarray:
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 20, axis=0)
    yaws = [0.0] * 9 + [30.0, 100.0, 180.0] + [180.0] * 8
    positions = [
        min(index, 8) * 0.10
        if index <= 11
        else 0.8 - (index - 11) * 0.10
        for index in range(20)
    ]
    for index, (yaw, position) in enumerate(zip(yaws, positions, strict=True)):
        c2w[index, :3, :3] = Rotation.from_euler(
            "y",
            yaw,
            degrees=True,
        ).as_matrix()
        c2w[index, 0, 3] = position
    return np.linalg.inv(c2w)


def test_detect_rotation_hinge_finds_turnaround() -> None:
    detection = detect_rotation_hinge(
        _turnaround_trajectory(),
        2,
        18,
        window_radius=1,
        search_margin=2,
    )

    assert 8 <= detection.hinge_edge <= 11
    assert detection.rotation_sum_deg > 100.0


def test_hinge_warp_keeps_both_legs_rigid() -> None:
    original = _turnaround_trajectory()
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = Rotation.from_euler(
        "y",
        8.0,
        degrees=True,
    ).as_matrix()
    correction[:3, 3] = [0.20, 0.0, -0.05]

    corrected, report = apply_hinge_correction(
        original,
        correction,
        hinge_edge=10,
        transition_radius=2,
    )

    original_c2w = np.linalg.inv(original)
    corrected_c2w = np.linalg.inv(corrected)
    assert np.allclose(corrected_c2w[0], original_c2w[0])
    assert np.allclose(
        corrected_c2w[-1],
        correction @ original_c2w[-1],
    )
    # Outside the short transition, left-multiplying an entire leg by one
    # rigid transform preserves its local camera motion exactly.
    for source, target in ((2, 3), (15, 16)):
        old_relative = original[target] @ np.linalg.inv(original[source])
        new_relative = corrected[target] @ np.linalg.inv(corrected[source])
        assert np.allclose(old_relative, new_relative)
    assert report["max_local_odometry_change"] < 0.15
