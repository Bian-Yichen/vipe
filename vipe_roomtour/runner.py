# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end orchestration: optional cutting, VIPE inference, then RGB-D mapping."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .artifacts import ArtifactSet
from .map_builder import MapOptions, build_map
from .segments import Segment, cut_segment

logger = logging.getLogger(__name__)


def _run_vipe(video: Path, artifact_root: Path, pipeline: str, visualize: bool) -> None:
    command = [
        sys.executable,
        "-m",
        "vipe_roomtour.basic_vipe",
        str(video),
        "--output",
        str(artifact_root),
        "--pipeline",
        pipeline,
    ]
    if visualize:
        command.append("--visualize")
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)


def run_roomtour(
    input_video: Path,
    output_root: Path,
    *,
    segments: list[Segment],
    pipeline: str,
    map_options: MapOptions,
    visualize_vipe: bool = False,
    skip_inference: bool = False,
    skip_map: bool = False,
    overwrite_segments: bool = False,
) -> dict:
    input_video = Path(input_video).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = output_root / "vipe_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path, float]] = []
    if segments:
        segment_dir = output_root / "segments"
        for segment in segments:
            segment_path = segment_dir / f"{segment.name}.mp4"
            if not segment_path.exists() or overwrite_segments:
                cut_segment(input_video, segment_path, segment, overwrite=overwrite_segments)
            jobs.append((segment.name, segment_path, segment.start_seconds))
    else:
        jobs.append((input_video.stem, input_video, 0.0))

    manifest = {
        "input_video": str(input_video),
        "output_root": str(output_root),
        "pipeline": pipeline,
        "segments": [asdict(segment) for segment in segments],
        "jobs": [],
    }
    for name, video_path, source_offset in jobs:
        artifact = ArtifactSet(artifact_root, name)
        if not skip_inference:
            if artifact.pose.exists():
                logger.info("Artifacts already exist for %s; skipping inference", name)
            else:
                _run_vipe(video_path, artifact_root, pipeline, visualize_vipe)
        artifact.validate()
        map_dir = output_root / "maps" / name
        metrics = None
        if not skip_map:
            metrics = build_map(artifact, map_dir, map_options, source_time_offset=source_offset)
        manifest["jobs"].append(
            {
                "name": name,
                "video": str(video_path),
                "source_time_offset_seconds": source_offset,
                "artifact_root": str(artifact_root),
                "map_dir": str(map_dir),
                "metrics": metrics,
            }
        )
        (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
