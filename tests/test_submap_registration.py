from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from vipe.slam.components.submap_registration import (
    SubmapRegistrationOptions,
    independent_transform_support,
    register_packed_submaps,
    transforms_are_consistent,
)


def _structured_room(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = 900
    # Three non-symmetric surfaces plus a coloured rail make the global
    # translation observable without relying on a single repeated wall.
    floor = np.column_stack(
        (
            rng.uniform(-2.0, 2.0, count),
            rng.uniform(-1.4, 1.4, count),
            np.zeros(count),
        )
    )
    wall_x = np.column_stack(
        (
            np.full(count, 1.65),
            rng.uniform(-1.4, 1.4, count),
            rng.uniform(0.0, 2.4, count),
        )
    )
    wall_y = np.column_stack(
        (
            rng.uniform(-2.0, 0.7, count),
            np.full(count, -1.25),
            rng.uniform(0.0, 2.4, count),
        )
    )
    rail_t = rng.uniform(0.0, 1.0, count // 2)
    rail = np.column_stack(
        (
            -1.4 + 2.2 * rail_t,
            0.7 + 0.25 * np.sin(rail_t * np.pi),
            0.35 + 1.4 * rail_t,
        )
    )
    points = np.concatenate((floor, wall_x, wall_y, rail))
    colors = np.concatenate(
        (
            np.tile([0.55, 0.42, 0.28], (len(floor), 1)),
            np.tile([0.88, 0.86, 0.82], (len(wall_x), 1)),
            np.tile([0.70, 0.78, 0.86], (len(wall_y), 1)),
            np.tile([0.72, 0.24, 0.10], (len(rail), 1)),
        )
    )
    return points, colors


def _packed_revisit(
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9)
    room, color = _structured_room()
    target_room = (
        (room - transform[:3, 3])
        @ transform[:3, :3]
    )
    xyz_parts = []
    rgb_parts = []
    packinfo = np.zeros((14, 1, 2), dtype=np.int64)
    cursor = 0
    for keyframe in range(14):
        if 1 <= keyframe <= 5:
            source = room
        elif 8 <= keyframe <= 12:
            source = target_room
        else:
            source = rng.uniform(6.0, 8.0, (700, 3))
        selected = rng.choice(len(source), min(1100, len(source)), replace=False)
        points = source[selected] + rng.normal(0.0, 0.006, (len(selected), 3))
        point_color = (
            color[selected]
            if len(source) == len(room)
            else rng.uniform(0.0, 1.0, (len(selected), 3))
        )
        xyz_parts.append(points)
        rgb_parts.append(point_color)
        packinfo[keyframe, 0] = (cursor, len(points))
        cursor += len(points)
    return np.concatenate(xyz_parts), np.concatenate(rgb_parts), packinfo


def test_multiscale_submap_registration_recovers_opposite_view_loop() -> None:
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler(
        "xyz",
        [1.0, -2.0, 3.0],
        degrees=True,
    ).as_matrix()
    expected[:3, 3] = [0.85, -0.25, 0.45]
    xyz, rgb, packinfo = _packed_revisit(expected)

    result = register_packed_submaps(
        xyz,
        rgb,
        packinfo,
        source=3,
        target=10,
        options=SubmapRegistrationOptions(
            radius_small=2,
            radius_large=3,
            correlation_peaks=4,
            min_mutual_correspondences=120,
            min_ambiguity_ratio=1.0,
        ),
    )

    assert result.accepted, result.reason
    assert result.target_world_to_source_world is not None
    delta = np.linalg.inv(expected) @ result.target_world_to_source_world
    assert np.linalg.norm(delta[:3, 3]) < 0.12
    assert np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude()) < 2.0
    assert result.metrics is not None
    assert result.metrics.symmetric_overlap > 0.70


def test_multiscale_submap_registration_rejects_unrelated_geometry() -> None:
    expected = np.eye(4)
    expected[:3, 3] = [0.8, 0.0, 0.4]
    xyz, rgb, packinfo = _packed_revisit(expected)
    # Replace the target windows by a compact sphere that cannot satisfy the
    # room's three-plane geometry at both scales.
    rng = np.random.default_rng(17)
    for keyframe in range(8, 13):
        start, count = packinfo[keyframe, 0]
        direction = rng.normal(size=(count, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        xyz[start : start + count] = 0.7 * direction + [1.0, 1.0, 1.0]

    result = register_packed_submaps(
        xyz,
        rgb,
        packinfo,
        source=3,
        target=10,
        options=SubmapRegistrationOptions(
            radius_small=2,
            radius_large=3,
            correlation_peaks=4,
            min_mutual_correspondences=120,
            min_ambiguity_ratio=1.0,
        ),
    )

    assert not result.accepted


def test_independent_world_corrections_must_agree() -> None:
    first = np.eye(4)
    first[:3, :3] = Rotation.from_euler(
        "xyz",
        [0.0, 1.0, 2.0],
        degrees=True,
    ).as_matrix()
    first[:3, 3] = [1.95, 0.10, 2.05]

    agreeing = np.eye(4)
    agreeing[:3, :3] = Rotation.from_euler(
        "xyz",
        [0.5, 0.0, 4.0],
        degrees=True,
    ).as_matrix()
    agreeing[:3, 3] = [1.90, 0.28, 2.16]
    assert transforms_are_consistent(
        first,
        agreeing,
        max_translation=0.40,
        max_rotation_deg=4.0,
    )

    conflicting = agreeing.copy()
    conflicting[:3, 3] += [0.8, 0.0, 0.0]
    assert not transforms_are_consistent(
        first,
        conflicting,
        max_translation=0.40,
        max_rotation_deg=4.0,
    )

    support = independent_transform_support(
        [
            (17, 76, first),
            (24, 74, agreeing),
            (35, 47, np.eye(4)),
        ],
        index_radius=8,
        max_translation=0.40,
        max_rotation_deg=4.0,
    )
    np.testing.assert_array_equal(support, [2, 2, 1])
