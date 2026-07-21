# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the expensive post-SLAM dense-depth stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


_PRESETS: dict[str, dict[str, Any]] = {
    # Backwards-compatible maximum-quality mode.  The SLAM-side keyframe depth
    # model is deliberately not controlled by this configuration.
    "quality": {
        "model": "giant",
        "frame_step": 1,
        "process_res": 504,
        "process_res_method": "lower_bound_resize",
        "output_resolution": "original",
    },
    # A practical production compromise for long room tours.
    "balanced": {
        "model": "large",
        "frame_step": 2,
        "process_res": 504,
        "process_res_method": "lower_bound_resize",
        "output_resolution": "model",
    },
    # Fast pose/map QA.  At 30 fps, frame_step=5 still supplies depth at 6 Hz.
    "preview": {
        "model": "large",
        "frame_step": 5,
        "process_res": 504,
        "process_res_method": "upper_bound_resize",
        "output_resolution": "model",
    },
}


@dataclass(frozen=True)
class DenseDepthOptions:
    """Options that affect only post-SLAM DAv3 depth and RGB-D fusion."""

    preset: str = "quality"
    model: str = "giant"
    model_path: str | None = None
    frame_step: int = 1
    process_res: int = 504
    process_res_method: str = "lower_bound_resize"
    output_resolution: str = "original"
    shard_size: int = 490
    # Keep ViPE's existing temporal context exactly as requested.  A shard is
    # aligned to a multiple of window_size-overlap_size for exact resume.
    window_size: int = 10
    overlap_size: int = 3

    def __post_init__(self) -> None:
        if self.preset not in _PRESETS:
            raise ValueError(f"Unknown depth preset: {self.preset}")
        if self.model not in {"giant", "large", "base", "small"}:
            raise ValueError(f"Unsupported DAv3 model: {self.model}")
        if self.frame_step < 1 or self.process_res < 56:
            raise ValueError("depth frame step must be >= 1 and process resolution >= 56")
        if self.process_res_method not in {"lower_bound_resize", "upper_bound_resize"}:
            raise ValueError(f"Unsupported DAv3 resize method: {self.process_res_method}")
        if self.output_resolution not in {"original", "model"}:
            raise ValueError("depth output resolution must be 'original' or 'model'")
        stride = self.window_size - self.overlap_size
        if self.window_size <= self.overlap_size or self.overlap_size != 3:
            raise ValueError("DAv3 must retain the existing overlap_size=3")
        if self.shard_size < stride or self.shard_size % stride:
            raise ValueError(
                f"depth shard size must be a positive multiple of the DAv3 window stride ({stride})"
            )

    @classmethod
    def from_preset(
        cls,
        preset: str,
        *,
        model: str | None = None,
        model_path: str | None = None,
        frame_step: int | None = None,
        process_res: int | None = None,
        process_res_method: str | None = None,
        output_resolution: str | None = None,
        shard_size: int = 490,
    ) -> "DenseDepthOptions":
        if preset not in _PRESETS:
            raise ValueError(f"Unknown depth preset: {preset}")
        options = cls(preset=preset, shard_size=shard_size, **_PRESETS[preset])
        overrides = {
            "model": model,
            "model_path": model_path,
            "frame_step": frame_step,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "output_resolution": output_resolution,
        }
        return replace(options, **{key: value for key, value in overrides.items() if value is not None})

    @property
    def window_stride(self) -> int:
        return self.window_size - self.overlap_size

    def expected_frame_indices(self, n_frames: int) -> list[int]:
        return list(range(0, n_frames, self.frame_step))

    def resume_inference_ordinal(self, completed: int) -> int:
        if completed < 0:
            raise ValueError("completed depth count must be non-negative")
        return max(0, completed - self.window_stride) if completed else 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def inference_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_path": self.model_path,
            "frame_step": self.frame_step,
            "process_res": self.process_res,
            "process_res_method": self.process_res_method,
            "output_resolution": self.output_resolution,
            "window_size": self.window_size,
            "overlap_size": self.overlap_size,
        }

    def subprocess_args(self) -> list[str]:
        args = [
            "--depth-preset",
            self.preset,
            "--dav3-model",
            self.model,
            "--depth-frame-step",
            str(self.frame_step),
            "--dav3-process-res",
            str(self.process_res),
            "--dav3-resize-method",
            self.process_res_method,
            "--depth-output-resolution",
            self.output_resolution,
            "--depth-shard-size",
            str(self.shard_size),
        ]
        if self.model_path is not None:
            args.extend(["--dav3-model-path", self.model_path])
        return args
