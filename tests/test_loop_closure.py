from __future__ import annotations

import numpy as np
import torch

from vipe.slam.components.loop_closure import (
    LoopClosureOptions,
    LoopConstraint,
    _constraint_world_correction,
    _optimize_pose_graph_matrices,
    _retrieval_descriptors,
    _sequence_score,
    _sequence_filter,
)


def test_retrieval_ignores_preallocated_buffer_capacity() -> None:
    class FakeBuffer:
        n_frames = 124
        # Capacity is 128, but only the first 124 slots are valid keyframes.
        fmaps = torch.randn(128, 1, 4, 2, 2)

    descriptors = _retrieval_descriptors(FakeBuffer(), batch_size=32)

    assert descriptors.shape[0] == FakeBuffer.n_frames


def _constraint(source: int, target: int, transform: np.ndarray) -> LoopConstraint:
    return LoopConstraint(
        source=source,
        target=target,
        source_frame=source * 10,
        target_frame=target * 10,
        similarity=0.9,
        sequence_similarity=0.9,
        sequence_direction=1,
        matches=100,
        inliers=80,
        inlier_ratio=0.8,
        essential_inliers=80,
        essential_rotation_error_deg=0.1,
        median_reprojection_error=0.5,
        cycle_rotation_deg=0.1,
        cycle_translation=0.01,
        current_rotation_error_deg=0.0,
        current_translation_error=1.0,
        target_from_source=transform,
    )


def test_sequence_filter_requires_neighboring_loop_pairs() -> None:
    identity = np.eye(4)
    clustered = [_constraint(2, 20, identity), _constraint(4, 22, identity)]
    isolated = _constraint(12, 40, identity)
    options = LoopClosureOptions(cluster_radius=3, min_cluster_support=2)

    retained = _sequence_filter(clustered + [isolated], options)

    assert {(item.source, item.target) for item in retained} == {(2, 20), (4, 22)}


def test_sequence_score_recognizes_reverse_traversal() -> None:
    similarities = np.zeros((12, 12), dtype=np.float32)
    # Source sequence 2,3,4 reappears in reverse as 9,8,7.
    similarities[2, 9] = 0.8
    similarities[3, 8] = 0.9
    similarities[4, 7] = 0.85

    score, direction = _sequence_score(similarities, 3, 8, radius=1)

    assert direction == -1
    assert score > 0.84


def test_image_loop_converts_to_target_leg_world_correction() -> None:
    w2c = np.repeat(np.eye(4)[None], 3, axis=0)
    w2c[2, 0, 3] = -1.0
    measurement = np.eye(4)
    constraint = _constraint(0, 2, measurement)

    correction = _constraint_world_correction(constraint, w2c)
    corrected_target_w2c = np.linalg.inv(
        correction @ np.linalg.inv(w2c[constraint.target])
    )
    corrected_relative = (
        corrected_target_w2c @ np.linalg.inv(w2c[constraint.source])
    )

    assert np.allclose(corrected_relative, measurement)


def test_pose_graph_distributes_a_verified_loop_correction() -> None:
    # A drifted trajectory ends one metre away even though frames 0 and 9
    # observe the same camera pose.  The verified identity loop should close
    # the endpoint while changing each local step only a little.
    original = np.repeat(np.eye(4)[None], 10, axis=0)
    original[:, 0, 3] = -np.linspace(0.0, 1.0, len(original))
    constraint = _constraint(0, 9, np.eye(4))

    corrected, report = _optimize_pose_graph_matrices(
        original,
        [constraint],
        LoopClosureOptions(pose_graph_max_nfev=80, min_cluster_support=1),
    )

    old_endpoint_error = np.linalg.norm(original[9, :3, 3] - original[0, :3, 3])
    new_endpoint_error = np.linalg.norm(corrected[9, :3, 3] - corrected[0, :3, 3])
    assert report["cost_after"] < report["cost_before"]
    assert new_endpoint_error < 0.25 * old_endpoint_error
    assert report["max_local_odometry_change"] < 0.3


def test_switchable_pose_graph_rejects_a_contradictory_loop() -> None:
    original = np.repeat(np.eye(4)[None], 20, axis=0)
    original[:, 0, 3] = -np.linspace(0.0, 2.0, len(original))
    identity = np.eye(4)
    contradictory = np.eye(4)
    contradictory[0, 3] = 2.5

    _, report = _optimize_pose_graph_matrices(
        original,
        [
            _constraint(0, 19, identity),
            _constraint(1, 18, identity),
            _constraint(5, 15, contradictory),
        ],
        LoopClosureOptions(pose_graph_max_nfev=100, min_cluster_support=2),
    )

    good_a, good_b, outlier = report["robust_loop_switch_weights"]
    assert report["applied"]
    assert min(good_a, good_b) > 0.8
    assert outlier < 0.1
    assert report["robust_loop_edges_retained"] == 2
