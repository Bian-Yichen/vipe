# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure NumPy hinge-aware trajectory correction for loop diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class HingeDetection:
    hinge_edge: int
    score: float
    rotation_sum_deg: float
    translation_sum: float
    source: int
    target: int
    candidates: list[dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rotation_degrees(matrix: np.ndarray) -> float:
    return float(
        np.degrees(
            Rotation.from_matrix(matrix[:3, :3]).magnitude()
        )
    )


def detect_rotation_hinge(
    w2c: np.ndarray,
    source: int,
    target: int,
    *,
    window_radius: int = 3,
    search_margin: int = 3,
) -> HingeDetection:
    """Find a compact high-rotation, low-translation interval inside a loop."""

    w2c = np.asarray(w2c, dtype=np.float64)
    if w2c.ndim != 3 or w2c.shape[1:] != (4, 4):
        raise ValueError(f"Expected [N,4,4] W2C poses, got {w2c.shape}")
    if not 0 <= source < target < len(w2c):
        raise ValueError(
            f"Invalid loop keyframes {source}->{target} for {len(w2c)} poses"
        )
    c2w = np.linalg.inv(w2c)
    rotations = []
    translations = []
    for edge in range(len(c2w) - 1):
        relative = np.linalg.inv(c2w[edge]) @ c2w[edge + 1]
        rotations.append(_rotation_degrees(relative))
        translations.append(float(np.linalg.norm(relative[:3, 3])))
    rotations = np.asarray(rotations)
    translations = np.asarray(translations)

    first_edge = min(
        max(source + search_margin, source),
        target - 1,
    )
    last_edge = max(
        min(target - search_margin - 1, target - 1),
        first_edge,
    )
    records = []
    for edge in range(first_edge, last_edge + 1):
        begin = max(source, edge - window_radius)
        end = min(target, edge + window_radius + 1)
        rotation_sum = float(rotations[begin:end].sum())
        translation_sum = float(translations[begin:end].sum())
        # A turn-in-place should score above ordinary forward walking. The
        # constant keeps the score finite when translation is almost zero.
        score = rotation_sum / (0.15 + translation_sum)
        records.append(
            {
                "hinge_edge": edge,
                "score": score,
                "rotation_sum_deg": rotation_sum,
                "translation_sum": translation_sum,
                "window_begin_edge": begin,
                "window_end_edge": end - 1,
            }
        )
    best = max(records, key=lambda item: item["score"])
    top = sorted(
        records,
        key=lambda item: item["score"],
        reverse=True,
    )[:8]
    return HingeDetection(
        hinge_edge=int(best["hinge_edge"]),
        score=float(best["score"]),
        rotation_sum_deg=float(best["rotation_sum_deg"]),
        translation_sum=float(best["translation_sum"]),
        source=source,
        target=target,
        candidates=top,
    )


def interpolate_world_correction(
    correction: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate identity-to-correction without changing scale."""

    correction = np.asarray(correction, dtype=np.float64)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    output = np.eye(4, dtype=np.float64)
    rotvec = Rotation.from_matrix(
        correction[:3, :3]
    ).as_rotvec()
    output[:3, :3] = Rotation.from_rotvec(alpha * rotvec).as_matrix()
    output[:3, 3] = alpha * correction[:3, 3]
    return output


def apply_hinge_correction(
    w2c: np.ndarray,
    world_correction: np.ndarray,
    hinge_edge: int,
    *,
    transition_radius: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep both trajectory legs rigid and blend correction near the hinge."""

    w2c = np.asarray(w2c, dtype=np.float64)
    world_correction = np.asarray(world_correction, dtype=np.float64)
    if not 0 <= hinge_edge < len(w2c) - 1:
        raise ValueError(
            f"Invalid hinge edge {hinge_edge} for {len(w2c)} poses"
        )
    if transition_radius < 1:
        raise ValueError("transition_radius must be >= 1")

    c2w = np.linalg.inv(w2c)
    corrected_c2w = np.empty_like(c2w)
    transition_start = max(0, hinge_edge - transition_radius + 1)
    transition_end = min(
        len(w2c) - 1,
        hinge_edge + transition_radius,
    )
    alphas = []
    for index, pose in enumerate(c2w):
        if index <= transition_start:
            alpha = 0.0
        elif index >= transition_end:
            alpha = 1.0
        else:
            value = (
                (index - transition_start)
                / max(transition_end - transition_start, 1)
            )
            # Smoothstep avoids angular-velocity discontinuities at either leg.
            alpha = value * value * (3.0 - 2.0 * value)
        alphas.append(alpha)
        corrected_c2w[index] = (
            interpolate_world_correction(world_correction, alpha)
            @ pose
        )
    corrected_w2c = np.linalg.inv(corrected_c2w)

    local_changes = []
    for source in range(len(w2c) - 1):
        old_relative = w2c[source + 1] @ np.linalg.inv(w2c[source])
        new_relative = (
            corrected_w2c[source + 1]
            @ np.linalg.inv(corrected_w2c[source])
        )
        delta = np.linalg.inv(old_relative) @ new_relative
        local_changes.append(
            float(
                np.linalg.norm(delta[:3, 3])
                + 0.1
                * Rotation.from_matrix(
                    delta[:3, :3]
                ).magnitude()
            )
        )
    original_centers = c2w[:, :3, 3]
    corrected_centers = corrected_c2w[:, :3, 3]
    center_change = np.linalg.norm(
        corrected_centers - original_centers,
        axis=1,
    )
    report = {
        "transition_start_keyframe": transition_start,
        "transition_end_keyframe": transition_end,
        "transition_radius": transition_radius,
        "full_correction_translation": float(
            np.linalg.norm(world_correction[:3, 3])
        ),
        "full_correction_rotation_deg": _rotation_degrees(
            world_correction
        ),
        "median_camera_center_correction": float(
            np.median(center_change)
        ),
        "max_camera_center_correction": float(
            np.max(center_change, initial=0.0)
        ),
        "max_local_odometry_change": float(
            np.max(local_changes, initial=0.0)
        ),
        "alphas": alphas,
    }
    return corrected_w2c, report
