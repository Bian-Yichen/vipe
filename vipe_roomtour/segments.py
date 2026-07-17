# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual room-tour segment parsing and loss-controlled FFmpeg cutting."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Segment:
    name: str
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def parse_segment(spec: str) -> Segment:
    try:
        name, start, end = spec.rsplit(":", 2)
        segment = Segment(name, float(start), float(end))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid segment '{spec}'; expected NAME:START_SECONDS:END_SECONDS") from exc
    if not _SAFE_NAME.fullmatch(segment.name):
        raise ValueError("Segment name may contain only letters, digits, dot, underscore and hyphen")
    if segment.start_seconds < 0 or segment.end_seconds <= segment.start_seconds:
        raise ValueError(f"Invalid time interval in segment '{spec}'")
    return segment


def cut_segment(input_video: Path, output_video: Path, segment: Segment, *, overwrite: bool = False) -> None:
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y" if overwrite else "-n",
        "-ss",
        f"{segment.start_seconds:.6f}",
        "-i",
        str(input_video),
        "-t",
        f"{segment.duration_seconds:.6f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "14",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found; install FFmpeg first") from exc
