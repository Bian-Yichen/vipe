# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output-path policy shared by every room-tour entry point."""

from __future__ import annotations

from pathlib import Path


def resolve_output_path(path: Path | str) -> Path:
    """Expand and resolve an explicitly absolute output path.

    Rejecting relative paths prevents a command such as ``... video.mp4 output``
    from silently writing below whichever repository happens to be the current
    working directory.
    """

    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        raise ValueError(
            f"Output path must be absolute, got '{path}'. "
            "For example: /mnt/petrelfs/bianyichen/3DVision/outputs/video_001"
        )
    return expanded.resolve()
