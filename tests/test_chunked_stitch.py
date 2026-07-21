from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from vipe_roomtour.artifacts import Calibration, write_ply
from vipe_roomtour.chunked import (
    ChunkResult,
    ChunkSpec,
    SimilarityTransform,
    StitchThresholds,
    estimate_pose_similarity,
    merge_chunk_calibrations,
    plan_chunks,
    read_binary_ply,
)


def _calibration(c2w: np.ndarray) -> Calibration:
    n = len(c2w)
    intrinsics = np.repeat(np.array([[800.0, 800.0, 640.0, 360.0]]), n, axis=0)
    K = np.repeat(np.eye(3)[None], n, axis=0)
    return Calibration(
        indices=np.arange(n, dtype=np.int64),
        c2w=c2w,
        w2c=np.linalg.inv(c2w),
        intrinsics=intrinsics,
        K=K,
        camera_types=tuple("PINHOLE" for _ in range(n)),
    )


def _trajectory(n: int) -> np.ndarray:
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    t = np.linspace(0.0, 1.0, n)
    poses[:, :3, 3] = np.column_stack((2.0 * t, 0.2 * np.sin(4 * t), 0.7 * t**2))
    poses[:, :3, :3] = Rotation.from_euler(
        "zyx",
        np.column_stack((0.3 * t, 0.05 * np.sin(t), -0.08 * t)),
    ).as_matrix()
    return poses


def test_plan_chunks_uses_fixed_overlap_and_maximum_size() -> None:
    chunks = plan_chunks(25_000, 5_000, 500)
    assert [(chunk.start_frame, chunk.end_frame) for chunk in chunks] == [
        (0, 5_000),
        (4_500, 9_500),
        (9_000, 14_000),
        (13_500, 18_500),
        (18_000, 23_000),
        (22_500, 25_000),
    ]
    assert all(chunk.frame_count <= 5_000 for chunk in chunks)
    assert all(left.end_frame - right.start_frame == 500 for left, right in zip(chunks[:-1], chunks[1:]))


def test_pose_similarity_recovers_rotation_translation_and_scale() -> None:
    source = _trajectory(120)
    expected = SimilarityTransform(
        scale=1.27,
        rotation=Rotation.from_euler("xyz", [0.12, -0.08, 0.31]).as_matrix(),
        translation=np.array([4.0, -1.5, 2.2]),
    )
    target = expected.transform_poses(source)
    actual, metrics = estimate_pose_similarity(
        source,
        target,
        thresholds=StitchThresholds(
            min_overlap_frames=30,
            min_baseline=0.1,
            max_position_rmse=1e-5,
            max_rotation_median_deg=1e-4,
        ),
    )
    np.testing.assert_allclose(actual.scale, expected.scale, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(actual.rotation, expected.rotation, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(actual.translation, expected.translation, rtol=1e-7, atol=1e-7)
    assert metrics["position_rmse"] < 1e-7


def test_pose_similarity_rejects_bad_overlap_pose_outliers() -> None:
    source = _trajectory(120)
    expected = SimilarityTransform(
        scale=0.91,
        rotation=Rotation.from_euler("xyz", [-0.1, 0.16, -0.24]).as_matrix(),
        translation=np.array([-3.0, 0.8, 1.4]),
    )
    target = expected.transform_poses(source)
    target[:8, :3, 3] += np.array([8.0, -5.0, 6.0])
    target[:8, :3, :3] = Rotation.from_euler("x", 1.0).as_matrix() @ target[:8, :3, :3]
    actual, metrics = estimate_pose_similarity(
        source,
        target,
        thresholds=StitchThresholds(min_overlap_frames=30, min_baseline=0.1, max_position_rmse=0.05),
    )
    np.testing.assert_allclose(actual.scale, expected.scale, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual.rotation, expected.rotation, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual.translation, expected.translation, rtol=1e-5, atol=1e-5)
    assert metrics["rotation_inliers"] == 112
    assert metrics["position_inliers"] == 112


def test_overlap_blending_covers_every_frame_and_preserves_exact_global_poses(tmp_path: Path) -> None:
    global_poses = _trajectory(180)
    first_spec = ChunkSpec(0, 0, 100)
    second_spec = ChunkSpec(1, 80, 180)
    second_transform = SimilarityTransform(
        1.15,
        Rotation.from_euler("z", 0.2).as_matrix(),
        np.array([2.0, 0.4, -1.0]),
    )
    inverse_rotation = second_transform.rotation.T
    second_local = global_poses[80:].copy()
    second_local[:, :3, :3] = np.einsum("ij,njk->nik", inverse_rotation, second_local[:, :3, :3])
    second_local[:, :3, 3] = (
        (second_local[:, :3, 3] - second_transform.translation) @ second_transform.rotation
    ) / second_transform.scale

    chunks = [
        ChunkResult(first_spec, None, tmp_path, _calibration(global_poses[:100]), SimilarityTransform.identity()),  # type: ignore[arg-type]
        ChunkResult(second_spec, None, tmp_path, _calibration(second_local), second_transform),  # type: ignore[arg-type]
    ]
    merged, depth_scales, primary = merge_chunk_calibrations(chunks, total_frames=180, overlap_frames=20)
    np.testing.assert_array_equal(merged.indices, np.arange(180))
    np.testing.assert_allclose(merged.c2w, global_poses, atol=1e-7)
    assert depth_scales[79] == 1.0
    assert depth_scales[100] == 1.15
    assert primary[80] == 0
    assert primary[99] == 1


def test_binary_ply_round_trip(tmp_path: Path) -> None:
    points = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]], dtype=np.float32)
    colors = np.array([[10, 20, 30], [240, 120, 60]], dtype=np.uint8)
    observations = np.array([2, 7], dtype=np.uint16)
    path = tmp_path / "cloud.ply"
    write_ply(path, points, colors, observations)
    actual_points, actual_colors, actual_observations = read_binary_ply(path)
    np.testing.assert_array_equal(actual_points, points)
    np.testing.assert_array_equal(actual_colors, colors)
    np.testing.assert_array_equal(actual_observations, observations)
