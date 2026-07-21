# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line entry point for room-tour calibration and top-down mapping."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import click

from .artifacts import discover_artifacts
from .chunked import StitchThresholds, run_chunked_roomtour
from .map_builder import MapOptions, build_map
from .runner import run_roomtour
from .segments import parse_segment


def _map_options(**kwargs) -> MapOptions:
    return MapOptions(**kwargs)


def _common_map_options(function):
    options = [
        click.option("--frame-step", type=click.IntRange(min=1), default=5, show_default=True, help="Fuse every Nth frame."),
        click.option("--pixel-stride", type=click.IntRange(min=1), default=4, show_default=True, help="Depth-pixel sampling stride."),
        click.option("--voxel-size", type=click.FloatRange(min=0.001), default=0.03, show_default=True, help="3D fusion voxel size in metres."),
        click.option("--min-voxel-observations", type=click.IntRange(min=1), default=2, show_default=True, help="Minimum different frames supporting a voxel."),
        click.option("--min-depth", type=click.FloatRange(min=0.0), default=0.15, show_default=True),
        click.option("--max-depth", type=click.FloatRange(min=0.01), default=15.0, show_default=True),
        click.option("--depth-edge-threshold", type=click.FloatRange(min=0.0), default=0.08, show_default=True, help="Reject relative 3x3 depth jumps; 0 disables."),
        click.option("--topdown-resolution", type=click.FloatRange(min=0.001), default=0.025, show_default=True, help="Requested metres per pixel."),
        click.option("--topdown-min-height", type=float, default=-0.10, show_default=True, help="Lowest retained height above floor."),
        click.option("--topdown-max-height", type=float, default=2.20, show_default=True, help="Highest retained height above floor."),
        click.option("--max-raster-size", type=click.IntRange(min=256), default=4096, show_default=True),
        click.option("--floor-y", type=float, default=None, help="Manual floor y-down value in leveled coordinates."),
        click.option("--fallback-camera-height", type=click.FloatRange(min=0.1), default=1.6, show_default=True),
    ]
    for option in reversed(options):
        function = option(function)
    return function


@click.group()
@click.option("--verbose", is_flag=True, help="Enable detailed logging.")
def main(verbose: bool) -> None:
    """Calibrate room-tour videos with VIPE and build global bird's-eye maps."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@main.command("run")
@click.argument("input_video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_root", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--segment",
    "segment_specs",
    multiple=True,
    help="Process an interval independently: NAME:START_SECONDS:END_SECONDS. Repeatable.",
)
@click.option("--pipeline", default="roomtour_dav3", show_default=True, help="VIPE Hydra pipeline config.")
@click.option("--visualize-vipe", is_flag=True, help="Also write VIPE's diagnostic video.")
@click.option("--skip-inference", is_flag=True, help="Use artifacts already under OUTPUT/vipe_artifacts.")
@click.option("--skip-map", is_flag=True, help="Run VIPE only; do not fuse RGB-D.")
@click.option("--overwrite-segments", is_flag=True, help="Recreate already cut segment videos.")
@_common_map_options
def run_command(
    input_video: Path,
    output_root: Path,
    segment_specs: tuple[str, ...],
    pipeline: str,
    visualize_vipe: bool,
    skip_inference: bool,
    skip_map: bool,
    overwrite_segments: bool,
    **map_kwargs,
) -> None:
    """Optionally cut INPUT_VIDEO, run VIPE, and create all deliverables."""

    try:
        segments = [parse_segment(spec) for spec in segment_specs]
        run_roomtour(
            input_video,
            output_root,
            segments=segments,
            pipeline=pipeline,
            map_options=_map_options(**map_kwargs),
            visualize_vipe=visualize_vipe,
            skip_inference=skip_inference,
            skip_map=skip_map,
            overwrite_segments=overwrite_segments,
        )
    except (ValueError, FileNotFoundError, RuntimeError, NotImplementedError, subprocess.CalledProcessError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("chunked-run")
@click.argument("input_video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_root", type=click.Path(file_okay=False, path_type=Path))
@click.option("--pipeline", default="roomtour_dav3", show_default=True)
@click.option(
    "--chunk-frames",
    type=click.IntRange(min=100),
    default=5000,
    show_default=True,
    help="Maximum frames in each independently solved chunk, including overlap.",
)
@click.option(
    "--overlap-frames",
    type=click.IntRange(min=30),
    default=500,
    show_default=True,
    help="Shared source frames used for adjacent Sim(3) alignment.",
)
@click.option("--min-stitch-baseline", type=click.FloatRange(min=0.0), default=0.25, show_default=True)
@click.option("--max-stitch-rmse", type=click.FloatRange(min=0.0), default=0.75, show_default=True)
@click.option("--max-stitch-rotation-deg", type=click.FloatRange(min=0.0), default=10.0, show_default=True)
@click.option("--min-stitch-scale", type=click.FloatRange(min=0.001), default=0.5, show_default=True)
@click.option("--max-stitch-scale", type=click.FloatRange(min=0.001), default=2.0, show_default=True)
@click.option("--skip-inference", is_flag=True, help="Require and reuse all chunk VIPE artifacts.")
@click.option("--skip-chunk-maps", is_flag=True, help="Require and reuse chunk maps with identical map options.")
@_common_map_options
def chunked_run_command(
    input_video: Path,
    output_root: Path,
    pipeline: str,
    chunk_frames: int,
    overlap_frames: int,
    min_stitch_baseline: float,
    max_stitch_rmse: float,
    max_stitch_rotation_deg: float,
    min_stitch_scale: float,
    max_stitch_scale: float,
    skip_inference: bool,
    skip_chunk_maps: bool,
    **map_kwargs,
) -> None:
    """Solve overlapping chunks independently and stitch them in Sim(3)."""

    try:
        if min_stitch_scale >= max_stitch_scale:
            raise ValueError("--max-stitch-scale must be greater than --min-stitch-scale")
        thresholds = StitchThresholds(
            min_baseline=min_stitch_baseline,
            max_position_rmse=max_stitch_rmse,
            max_rotation_median_deg=max_stitch_rotation_deg,
            min_scale=min_stitch_scale,
            max_scale=max_stitch_scale,
        )
        run_chunked_roomtour(
            input_video,
            output_root,
            pipeline=pipeline,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
            map_options=_map_options(**map_kwargs),
            thresholds=thresholds,
            skip_inference=skip_inference,
            skip_chunk_maps=skip_chunk_maps,
        )
    except (ValueError, FileNotFoundError, RuntimeError, NotImplementedError, subprocess.CalledProcessError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("map")
@click.argument("artifact_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_root", type=click.Path(file_okay=False, path_type=Path))
@click.option("--artifact-name", default=None, help="Map one artifact; by default map every pose NPZ.")
@click.option("--source-time-offset", type=float, default=0.0, show_default=True)
@_common_map_options
def map_command(
    artifact_root: Path,
    output_root: Path,
    artifact_name: str | None,
    source_time_offset: float,
    **map_kwargs,
) -> None:
    """Build maps from an existing VIPE artifact directory without rerunning inference."""

    try:
        artifacts = discover_artifacts(artifact_root, artifact_name)
        for artifact in artifacts:
            destination = output_root / artifact.name
            build_map(artifact, destination, _map_options(**map_kwargs), source_time_offset=source_time_offset)
            click.echo(f"Wrote {destination}")
    except (ValueError, FileNotFoundError, RuntimeError, NotImplementedError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
