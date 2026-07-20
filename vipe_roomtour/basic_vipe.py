# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Hydra-free VIPE inference used by the room-tour workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def build_pipeline(output: Path, pipeline_name: str, save_viz: bool = False):
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

    slam = OmegaConf.load(get_config_path() / "slam" / "default.yaml")
    slam.optimize_intrinsics = True
    slam.keyframe_depth = keyframe_depth
    slam.visualize = False

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
    post = OmegaConf.create({"depth_align_model": dense_depth})
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


def run_inference(video: Path, output: Path, pipeline_name: str, save_viz: bool = False) -> None:
    """Decode one continuous video and run the minimal static-scene pipeline."""

    from vipe.streams.base import ProcessedVideoStream
    from vipe.streams.raw_mp4_stream import RawMp4Stream
    from vipe.utils.logging import configure_logging

    logger = configure_logging()
    video = Path(video).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logger.info("Processing video %s with minimal pipeline %s", video, pipeline_name)
    # RawMp4Stream is re-iterable.  Keeping it streaming avoids retaining every
    # full-resolution float32 RGB frame for the lifetime of a long job.
    stream = ProcessedVideoStream(RawMp4Stream(video), [])
    build_pipeline(output, pipeline_name, save_viz=save_viz).run(stream)
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
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    run_inference(args.video, args.output, args.pipeline, save_viz=args.visualize)


if __name__ == "__main__":
    main()
