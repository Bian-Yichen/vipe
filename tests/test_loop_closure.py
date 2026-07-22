from __future__ import annotations

import numpy as np

from vipe.slam.components.loop_closure import (
    LoopClosureOptions,
    LoopConstraint,
    _optimize_pose_graph_matrices,
    _sequence_filter,
)


def _constraint(source: int, target: int, transform: np.ndarray) -> LoopConstraint:
    return LoopConstraint(
        source=source,
        target=target,
        source_frame=source * 10,
        target_frame=target * 10,
        similarity=0.9,
        matches=100,
        inliers=80,
        inlier_ratio=0.8,
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
        LoopClosureOptions(pose_graph_max_nfev=80),
    )

    old_endpoint_error = np.linalg.norm(original[9, :3, 3] - original[0, :3, 3])
    new_endpoint_error = np.linalg.norm(corrected[9, :3, 3] - corrected[0, :3, 3])
    assert report["cost_after"] < report["cost_before"]
    assert new_endpoint_error < 0.25 * old_endpoint_error
    assert report["max_local_odometry_change"] < 0.3
