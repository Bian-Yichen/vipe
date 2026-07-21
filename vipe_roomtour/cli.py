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
from .depth_options import DenseDepthOptions
from .map_builder import MapOptions, build_map
from .runner import run_roomtour
from .segments import parse_segment


def _map_options(*, default_frame_step: int = 5, **kwargs) -> MapOptions:
    if kwargs.get("frame_step") is None:
        kwargs["frame_step"] = default_frame_step
    return MapOptions(**kwargs)


def _depth_options(
    depth_preset: str,
    dav3_model: str | None,
    dav3_model_path: Path | None,
    depth_frame_step: int | None,
    dav3_process_res: int | None,
    dav3_resize_method: str | None,
    depth_output_resolution: str | None,
    depth_shard_size: int,
) -> DenseDepthOptions:
    return DenseDepthOptions.from_preset(
        depth_preset,
        model=dav3_model,
        model_path=str(dav3_model_path) if dav3_model_path is not None else None,
        frame_step=depth_frame_step,
        process_res=dav3_process_res,
        process_res_method=dav3_resize_method,
        output_resolution=depth_output_resolution,
        shard_size=depth_shard_size,
    )


def _common_depth_options(function):
    options = [
        click.option(
            "--depth-preset",
            type=click.Choice(("preview", "balanced", "quality")),
            default="quality",
            show_default=True,
        ),
        click.option("--dav3-model", type=click.Choice(("giant", "large", "base", "small")), default=None),
        click.option(
            "--dav3-model-path",
            type=click.Path(exists=True, path_type=Path),
            default=None,
            help="Local DAv3 checkpoint directory (useful on offline compute nodes).",
        ),
        click.option("--depth-frame-step", type=click.IntRange(min=1), default=None),
        click.option("--dav3-process-res", type=click.IntRange(min=56), default=None),
        click.option(
            "--dav3-resize-method",
            type=click.Choice(("lower_bound_resize", "upper_bound_resize")),
            default=None,
        ),
        click.option(
            "--depth-output-resolution",
            type=click.Choice(("original", "model")),
            default=None,
        ),
        click.option(
            "--depth-shard-size",
            type=click.IntRange(min=7),
            default=490,
            show_default=True,
            help="Resume checkpoint size; must be divisible by 7 because overlap remains 3/10.",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def _common_map_options(function):
    options = [
        click.option(
            "--frame-step",
            type=click.IntRange(min=1),
            default=None,
            help="Fuse every Nth source frame; defaults to --depth-frame-step for run commands.",
        ),
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
@click.option("--pose-only", is_flag=True, help="Stop after all-frame SLAM pose/intrinsics; do not load dense DAv3.")
@click.option("--depth-only", is_flag=True, help="Require an existing SLAM checkpoint and run only dense depth/map.")
@_common_depth_options
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
    pose_only: bool,
    depth_only: bool,
    depth_preset: str,
    dav3_model: str | None,
    dav3_model_path: Path | None,
    depth_frame_step: int | None,
    dav3_process_res: int | None,
    dav3_resize_method: str | None,
    depth_output_resolution: str | None,
    depth_shard_size: int,
    **map_kwargs,
) -> None:
    """Optionally cut INPUT_VIDEO, run VIPE, and create all deliverables."""

    try:
        if pose_only and depth_only:
            raise ValueError("--pose-only and --depth-only are mutually exclusive")
        dense_depth = _depth_options(
            depth_preset,
            dav3_model,
            dav3_model_path,
            depth_frame_step,
            dav3_process_res,
            dav3_resize_method,
            depth_output_resolution,
            depth_shard_size,
        )
        segments = [parse_segment(spec) for spec in segment_specs]
        run_roomtour(
            input_video,
            output_root,
            segments=segments,
            pipeline=pipeline,
            map_options=_map_options(
                default_frame_step=dense_depth.frame_step if dense_depth.frame_step > 1 else 5,
                **map_kwargs,
            ),
            visualize_vipe=visualize_vipe,
            skip_inference=skip_inference,
            skip_map=skip_map,
            overwrite_segments=overwrite_segments,
            inference_mode="pose" if pose_only else "depth" if depth_only else "full",
            depth_options=dense_depth,
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
@click.option("--pose-only", is_flag=True, help="Stitch/export all-frame poses without running dense DAv3.")
@click.option("--depth-only", is_flag=True, help="Require existing chunk SLAM checkpoints; run depth and maps only.")
@click.option("--depth-workers", type=click.IntRange(min=1), default=1, show_default=True, help="Parallel DAv3 workers, one per visible GPU.")
@_common_depth_options
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
    pose_only: bool,
    depth_only: bool,
    depth_workers: int,
    depth_preset: str,
    dav3_model: str | None,
    dav3_model_path: Path | None,
    depth_frame_step: int | None,
    dav3_process_res: int | None,
    dav3_resize_method: str | None,
    depth_output_resolution: str | None,
    depth_shard_size: int,
    **map_kwargs,
) -> None:
    """Solve overlapping chunks independently and stitch them in Sim(3)."""

    try:
        if pose_only and depth_only:
            raise ValueError("--pose-only and --depth-only are mutually exclusive")
        if min_stitch_scale >= max_stitch_scale:
            raise ValueError("--max-stitch-scale must be greater than --min-stitch-scale")
        dense_depth = _depth_options(
            depth_preset,
            dav3_model,
            dav3_model_path,
            depth_frame_step,
            dav3_process_res,
            dav3_resize_method,
            depth_output_resolution,
            depth_shard_size,
        )
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
            map_options=_map_options(
                default_frame_step=dense_depth.frame_step if dense_depth.frame_step > 1 else 5,
                **map_kwargs,
            ),
            thresholds=thresholds,
            skip_inference=skip_inference,
            skip_chunk_maps=skip_chunk_maps,
            inference_mode="pose" if pose_only else "depth" if depth_only else "full",
            depth_options=dense_depth,
            depth_workers=depth_workers,
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
            build_map(
                artifact,
                destination,
                _map_options(default_frame_step=5, **map_kwargs),
                source_time_offset=source_time_offset,
            )
            click.echo(f"Wrote {destination}")
    except (ValueError, FileNotFoundError, RuntimeError, NotImplementedError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
