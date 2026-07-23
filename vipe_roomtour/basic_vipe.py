# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Hydra-free VIPE inference used by the room-tour workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from .depth_options import DenseDepthOptions
from .output_paths import resolve_output_path


def build_pipeline(
    output: Path,
    pipeline_name: str,
    save_viz: bool = False,
    depth_options: DenseDepthOptions | None = None,
    loop_closure: bool = False,
    loop_experiment: str = "normal",
):
    """Construct the bounded-memory annotation pipeline without Hydra."""

    from vipe import get_config_path

    from .streaming_vipe import StreamingRoomTourPipeline

    presets = {
        "roomtour_dav3": ("dav3", "mvd_dav3"),
        "roomtour_default": ("unidepth-l", "adaptive_unidepth-l_svda"),
    }
    if pipeline_name not in presets:
        choices = ", ".join(sorted(presets))
        raise ValueError(f"Unsupported minimal pipeline '{pipeline_name}'. Choose one of: {choices}")
    keyframe_depth, dense_depth = presets[pipeline_name]
    depth_options = depth_options or DenseDepthOptions.from_preset("quality")
    if loop_experiment != "normal" and not loop_closure:
        raise ValueError("loop_experiment requires loop_closure=True")

    slam = OmegaConf.load(get_config_path() / "slam" / "default.yaml")
    slam.optimize_intrinsics = True
    slam.keyframe_depth = keyframe_depth
    slam.visualize = False
    slam.loop_closure.enabled = bool(loop_closure)
    slam.loop_closure.experiment_mode = str(loop_experiment)

    init = OmegaConf.create(
        {
            "camera_type": "pinhole",
            "intrinsics": "geocalib",
            "async_prefetch": True,
            "prefetch_queue_size": 16,
            # The room-tour dataset is static, so no tracking/masking model is needed.
            "instance": None,
        }
    )
    post = OmegaConf.create(
        {
            "depth_align_model": dense_depth,
            "dav3_model": depth_options.model,
            "dav3_model_path": depth_options.model_path,
            "depth_frame_step": depth_options.frame_step,
            "dav3_process_res": depth_options.process_res,
            "dav3_process_res_method": depth_options.process_res_method,
            "depth_output_resolution": depth_options.output_resolution,
            "depth_shard_size": depth_options.shard_size,
            "depth_inference_start_ordinal": 0,
            "depth_emit_start_ordinal": 0,
        }
    )
    output_config = OmegaConf.create(
        {
            "path": str(Path(output).resolve()),
            "skip_exists": False,
            "save_artifacts": True,
            "save_slam_map": True,
            "save_viz": bool(save_viz),
            "viz_downsample": 2,
            "viz_attributes": [["rgb", "depth"], ["pcd"]],
        }
    )
    return StreamingRoomTourPipeline(init=init, slam=slam, post=post, output=output_config)


def run_inference(
    video: Path,
    output: Path,
    pipeline_name: str,
    save_viz: bool = False,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    artifact_name: str | None = None,
    mode: str = "full",
    depth_options: DenseDepthOptions | None = None,
    loop_closure: bool = False,
    loop_experiment: str = "normal",
) -> None:
    """Decode one continuous video and run the minimal static-scene pipeline."""

    from vipe.streams.base import ProcessedVideoStream
    from vipe.streams.raw_mp4_stream import RawMp4Stream
    from vipe.utils.logging import configure_logging

    logger = configure_logging()
    video = Path(video).resolve()
    output = resolve_output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    logger.info("Processing video %s with minimal pipeline %s", video, pipeline_name)
    seek_range = None
    if start_frame is not None or end_frame is not None:
        start = 0 if start_frame is None else int(start_frame)
        end = -1 if end_frame is None else int(end_frame)
        if start < 0 or (end != -1 and end <= start):
            raise ValueError(f"Invalid half-open frame range [{start}, {end})")
        seek_range = range(start, end)
        logger.info("Restricting input to source frames [%d, %s)", start, "end" if end == -1 else end)

    # RawMp4Stream is re-iterable. Keeping it streaming avoids retaining every
    # full-resolution float32 RGB frame for the lifetime of a long job.
    stream = ProcessedVideoStream(
        RawMp4Stream(video, seek_range=seek_range, name=artifact_name),
        [],
    )
    pipeline = build_pipeline(
        output,
        pipeline_name,
        save_viz=save_viz,
        depth_options=depth_options,
        loop_closure=loop_closure,
        loop_experiment=loop_experiment,
    )
    pipeline.run_mode(stream, mode=mode, frame_index_offset=0 if start_frame is None else int(start_frame))
    logger.info("Finished minimal VIPE inference")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Hydra-free VIPE inference for room tours")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--pipeline",
        "-p",
        choices=("roomtour_dav3", "roomtour_default"),
        default="roomtour_dav3",
    )
    parser.add_argument("--visualize", action="store_true", help="Write VIPE's diagnostic video")
    parser.add_argument("--start-frame", type=int, default=None, help="Inclusive source frame for chunked inference")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive source frame for chunked inference")
    parser.add_argument("--artifact-name", default=None, help="Override the artifact basename")
    parser.add_argument("--mode", choices=("full", "pose", "depth"), default="full")
    parser.add_argument(
        "--loop-closure",
        action="store_true",
        help="Enable geometry-verified long-range loop closure inside this SLAM solve",
    )
    parser.add_argument(
        "--loop-experiment",
        choices=("normal", "skip-final-ba", "hinge-warp"),
        default="normal",
        help="How to preserve or localize a verified loop correction",
    )
    parser.add_argument("--depth-preset", choices=("preview", "balanced", "quality"), default="quality")
    parser.add_argument("--dav3-model", choices=("giant", "large", "base", "small"), default=None)
    parser.add_argument(
        "--dav3-model-path",
        default=None,
        help="Local model.safetensors file or directory containing it",
    )
    parser.add_argument("--depth-frame-step", type=int, default=None)
    parser.add_argument("--dav3-process-res", type=int, default=None)
    parser.add_argument(
        "--dav3-resize-method",
        choices=("lower_bound_resize", "upper_bound_resize"),
        default=None,
    )
    parser.add_argument("--depth-output-resolution", choices=("original", "model"), default=None)
    parser.add_argument("--depth-shard-size", type=int, default=490)
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    if args.loop_experiment != "normal" and not args.loop_closure:
        parser.error("--loop-experiment requires --loop-closure")
    depth_options = DenseDepthOptions.from_preset(
        args.depth_preset,
        model=args.dav3_model,
        model_path=args.dav3_model_path,
        frame_step=args.depth_frame_step,
        process_res=args.dav3_process_res,
        process_res_method=args.dav3_resize_method,
        output_resolution=args.depth_output_resolution,
        shard_size=args.depth_shard_size,
    )
    run_inference(
        args.video,
        args.output,
        args.pipeline,
        save_viz=args.visualize,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        artifact_name=args.artifact_name,
        mode=args.mode,
        depth_options=depth_options,
        loop_closure=args.loop_closure,
        loop_experiment=args.loop_experiment,
    )


if __name__ == "__main__":
    main()
