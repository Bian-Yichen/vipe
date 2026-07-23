# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end orchestration: optional cutting, VIPE inference, then RGB-D mapping."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .artifacts import ArtifactSet, export_calibration, load_calibration
from .depth_options import DenseDepthOptions
from .map_builder import MapOptions, build_map, probe_video
from .output_paths import resolve_output_path
from .segments import Segment, cut_segment

logger = logging.getLogger(__name__)


def _run_vipe(
    video: Path,
    artifact_root: Path,
    pipeline: str,
    visualize: bool,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    artifact_name: str | None = None,
    mode: str = "full",
    depth_options: DenseDepthOptions | None = None,
    cuda_visible_device: str | None = None,
    loop_closure: bool = False,
    loop_experiment: str = "normal",
) -> None:
    command = [
        sys.executable,
        "-m",
        "vipe_roomtour.basic_vipe",
        str(video),
        "--output",
        str(artifact_root),
        "--pipeline",
        pipeline,
        "--mode",
        mode,
    ]
    if visualize:
        command.append("--visualize")
    if loop_closure:
        command.append("--loop-closure")
        command.extend(["--loop-experiment", loop_experiment])
    if start_frame is not None:
        command.extend(["--start-frame", str(start_frame)])
    if end_frame is not None:
        command.extend(["--end-frame", str(end_frame)])
    if artifact_name is not None:
        command.extend(["--artifact-name", artifact_name])
    if depth_options is not None:
        command.extend(depth_options.subprocess_args())
    logger.info("Running: %s", " ".join(command))
    environment = None
    if cuda_visible_device is not None:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
        logger.info("Assigning dense-depth subprocess to CUDA_VISIBLE_DEVICES=%s", cuda_visible_device)
    subprocess.run(command, check=True, env=environment)


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
    inference_mode: str = "full",
    depth_options: DenseDepthOptions | None = None,
    loop_closure: bool = False,
    loop_experiment: str = "normal",
) -> dict:
    input_video = Path(input_video).resolve()
    output_root = resolve_output_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = output_root / "vipe_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    depth_options = depth_options or DenseDepthOptions.from_preset("quality")

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
        "inference_mode": inference_mode,
        "dense_depth": depth_options.to_dict(),
        "loop_closure": bool(loop_closure),
        "loop_experiment": loop_experiment,
        "segments": [asdict(segment) for segment in segments],
        "jobs": [],
    }
    for name, video_path, source_offset in jobs:
        artifact = ArtifactSet(artifact_root, name)
        if inference_mode == "pose":
            calibration_complete = artifact.slam_matches(
                loop_closure,
                loop_experiment,
            )
            if not calibration_complete:
                if skip_inference:
                    raise FileNotFoundError(
                        "Missing SLAM checkpoint matching "
                        f"loop_closure={loop_closure}, "
                        f"loop_experiment={loop_experiment} for {name}"
                    )
                _run_vipe(
                    video_path,
                    artifact_root,
                    pipeline,
                    visualize_vipe,
                    mode="pose",
                    loop_closure=loop_closure,
                    loop_experiment=loop_experiment,
                )
            artifact.validate_calibration()
            calibration = load_calibration(artifact)
            map_dir = output_root / "maps" / name
            video_info = probe_video(video_path)
            export_calibration(
                calibration,
                map_dir,
                fps=float(video_info["fps"]),
                source_time_offset=source_offset,
                image_wh=(int(video_info["width"]), int(video_info["height"])),
            )
            manifest["jobs"].append(
                {
                    "name": name,
                    "video": str(video_path),
                    "source_time_offset_seconds": source_offset,
                    "artifact_root": str(artifact_root),
                    "map_dir": str(map_dir),
                    "metrics": None,
                }
            )
            (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            continue

        if skip_inference:
            artifact.validate()
            if not artifact.depth_matches(
                depth_options,
                loop_closure=loop_closure,
                loop_experiment=loop_experiment,
            ):
                raise FileNotFoundError(f"Existing depth artifacts do not match requested config for {name}")
        elif artifact.depth_matches(
            depth_options,
            loop_closure=loop_closure,
            loop_experiment=loop_experiment,
        ):
            logger.info("Matching complete artifacts already exist for %s; skipping inference", name)
        else:
            _run_vipe(
                video_path,
                artifact_root,
                pipeline,
                visualize_vipe,
                mode="depth" if inference_mode == "depth" else "full",
                depth_options=depth_options,
                loop_closure=loop_closure,
                loop_experiment=loop_experiment,
            )
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
