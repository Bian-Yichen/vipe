# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory-bounded ViPE execution for long, single-view room-tour videos.

The stock annotation pipeline caches the final ``VideoFrame`` for every frame
before writing artifacts.  A final frame contains both float32 RGB and dense
float32 depth, so that strategy grows with video length and can consume
hundreds of GiB.  This module keeps the inference algorithm unchanged but
writes RGB and depth while the two-pass multiview-depth stream is consumed.
"""

from __future__ import annotations

import gc
import logging
import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

import Imath
import numpy as np
import OpenEXR
import torch

from vipe.ext.lietorch import SE3
from vipe.pipeline import AnnotationPipelineOutput
from vipe.pipeline.default import DefaultAnnotationPipeline
from vipe.slam.interface import SLAMMap, SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoFrame,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.visualization import VideoWriter

logger = logging.getLogger(__name__)


def _temporary_media_path(path: Path) -> Path:
    """Return a sibling temporary path while retaining the media suffix."""

    return path.with_name(f"{path.stem}.partial{path.suffix}")


class StreamingRGBWriterProcessor(StreamProcessor):
    """Write RGB during multiview depth's existing pre-pass.

    ``MultiviewDepthProcessor`` already requires a first pass to collect its
    anchor keyframes.  Encoding RGB in that same pass avoids an additional
    video decode and produces the same H264 artifact as ``save_rgb_artifacts``.
    """

    n_passes_required = 2

    def __init__(self, path: Path, fps: float) -> None:
        self.path = Path(path)
        self.fps = float(fps)

    def update_fps(self, previous_fps: float) -> float:
        return previous_fps

    def update_frame_size(self, previous_frame_size: tuple[int, int]) -> tuple[int, int]:
        return previous_frame_size

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        return frame

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if pass_idx != 0:
            yield from previous_iterator
            return

        self.path.parent.mkdir(exist_ok=True, parents=True)
        partial = _temporary_media_path(self.path)
        partial.unlink(missing_ok=True)
        try:
            with VideoWriter(partial, self.fps) as writer:
                for frame in previous_iterator:
                    writer.write((frame.rgb.detach().cpu().numpy() * 255).astype(np.uint8))
                    yield frame
            os.replace(partial, self.path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise


def _write_depth_exr(depth: np.ndarray, destination: Path) -> None:
    """Write one depth image exactly like ViPE's stock artifact writer."""

    height, width = depth.shape
    header = OpenEXR.Header(width, height)
    header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
    exr = OpenEXR.OutputFile(str(destination), header)
    try:
        exr.writePixels({"Z": depth.astype(np.float16, copy=False).tobytes()})
    finally:
        exr.close()


def save_depth_artifacts_streaming(path: Path, stream: VideoStream) -> int:
    """Consume a depth stream once and incrementally create the EXR zip.

    Only the current frame is materialized on CPU.  The final filename is
    published atomically so a failed job cannot be mistaken for a complete
    artifact on the next invocation.
    """

    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    frame_count = 0
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
            for frame_idx, frame in enumerate(stream):
                if frame.metric_depth is None:
                    continue
                depth = frame.metric_depth.detach().cpu().numpy().astype(np.float16, copy=False)
                with tempfile.NamedTemporaryFile(suffix=".exr") as temp:
                    _write_depth_exr(depth, Path(temp.name))
                    archive.write(temp.name, f"{frame_idx:05d}.exr")
                frame_count += 1
                # Release references before the next DAv3 window is produced.
                del depth, frame
        if frame_count == 0:
            raise RuntimeError("The dense-depth stream produced no frames")
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return frame_count


def _save_slam_checkpoint(artifact: io.ArtifactPath, slam_output: SLAMOutput, n_frames: int) -> None:
    """Persist all small calibration artifacts and the SLAM map before DAv3."""

    artifact.pose_path.parent.mkdir(exist_ok=True, parents=True)
    artifact.intrinsics_path.parent.mkdir(exist_ok=True, parents=True)
    artifact.meta_info_path.parent.mkdir(exist_ok=True, parents=True)

    indices = np.arange(n_frames, dtype=np.int64)
    trajectory = slam_output.get_view_trajectory(0).matrix().detach().cpu().numpy()
    if len(trajectory) != n_frames:
        raise ValueError(f"Expected {n_frames} poses, got {len(trajectory)}")
    np.savez(artifact.pose_path, data=trajectory, inds=indices)

    intrinsics = slam_output.intrinsics[0].detach().cpu().numpy()
    intrinsics_per_frame = np.repeat(intrinsics[None], n_frames, axis=0)
    np.savez(artifact.intrinsics_path, data=intrinsics_per_frame, inds=indices)
    with artifact.camera_type_path.open("w") as handle:
        for frame_idx in range(n_frames):
            handle.write(f"{frame_idx}: {CameraType.PINHOLE.name}\n")

    with artifact.meta_info_path.open("wb") as handle:
        pickle.dump({"ba_residual": slam_output.ba_residual}, handle)

    slam_map = slam_output.slam_map
    if slam_map is None:
        raise RuntimeError("SLAM did not return a map")
    slam_map.save(artifact.slam_map_path)


def _load_slam_checkpoint(artifact: io.ArtifactPath, n_frames: int) -> SLAMOutput | None:
    """Load a complete streaming checkpoint, or return ``None`` if incomplete."""

    paths = (artifact.pose_path, artifact.intrinsics_path, artifact.camera_type_path, artifact.slam_map_path)
    if not all(path.exists() for path in paths):
        return None
    try:
        pose_data = np.load(artifact.pose_path)
        intrinsics_data = np.load(artifact.intrinsics_path)
        if len(pose_data["inds"]) != n_frames or len(intrinsics_data["inds"]) != n_frames:
            logger.warning("Ignoring incomplete SLAM checkpoint for %s", artifact.artifact_name)
            return None
        trajectory = se3_matrix_to_se3(pose_data["data"]).cuda()
        intrinsics = torch.from_numpy(intrinsics_data["data"][0]).float().cuda()[None]
        slam_map = SLAMMap.load(artifact.slam_map_path, device=torch.device("cpu"))
        ba_residual = 0.0
        if artifact.meta_info_path.exists():
            with artifact.meta_info_path.open("rb") as handle:
                meta_info = pickle.load(handle)
            ba_residual = float(meta_info.get("ba_residual", 0.0))
        logger.info("Resuming dense-depth export from saved SLAM checkpoint for %s", artifact.artifact_name)
        return SLAMOutput(
            trajectory=trajectory,
            intrinsics=intrinsics,
            rig=SE3.Identity(1).cuda(),
            slam_map=slam_map,
            ba_residual=ba_residual,
        )
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, KeyError, RuntimeError) as exc:
        logger.warning("Could not load SLAM checkpoint for %s: %s", artifact.artifact_name, exc)
        return None


def _move_slam_map_to_cpu(slam_output: SLAMOutput) -> None:
    """Release the dense keyframe map from GPU after it has been saved."""

    if slam_output.slam_map is None:
        return
    slam_output.slam_map.dense_disp_xyz = slam_output.slam_map.dense_disp_xyz.cpu()
    slam_output.slam_map.dense_disp_rgb = slam_output.slam_map.dense_disp_rgb.cpu()
    slam_output.slam_map.dense_disp_packinfo = slam_output.slam_map.dense_disp_packinfo.cpu()
    if slam_output.slam_map.backend_graph is not None:
        slam_output.slam_map.backend_graph = slam_output.slam_map.backend_graph.cpu()


class StreamingRoomTourPipeline(DefaultAnnotationPipeline):
    """Stock single-view ViPE inference with bounded-memory artifact output."""

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            raise NotImplementedError("The room-tour streaming pipeline currently supports one video stream")

        artifact = io.ArtifactPath(self.out_path, video_data.name())
        n_frames = len(video_data)
        slam_output = _load_slam_checkpoint(artifact, n_frames)

        if slam_output is None:
            slam_stream = self._add_init_processors(video_data)
            slam_pipeline = SLAMSystem(
                device=torch.device("cuda"),
                config=self.slam_cfg,
                model_cache=self.model_cache,
            )
            slam_output = slam_pipeline.run([slam_stream], rig=None, camera_type=self.camera_type)
            _save_slam_checkpoint(artifact, slam_output, n_frames)
            _move_slam_map_to_cpu(slam_output)

            # DROID, GeoCalib and the keyframe metric-depth model are no longer
            # needed once the all-frame trajectory and keyframe map are saved.
            self.model_cache.clear()
            del slam_pipeline
            gc.collect()
            torch.cuda.empty_cache()
        else:
            # A resumed post-process does not need to load GeoCalib again.  It
            # only needs to restore the camera model attribute that GeoCalib
            # would attach to every frame.
            slam_stream = ProcessedVideoStream(
                video_data,
                [
                    AssignAttributesProcessor(
                        {FrameAttribute.CAMERA_TYPE: [self.camera_type] * n_frames}
                    )
                ],
            )

        output_stream = self._add_post_processors(0, slam_stream, slam_output)
        # This processor writes RGB during MultiviewDepthProcessor's mandatory
        # keyframe-recording pass, so it adds no extra decode pass.
        output_stream.processors.insert(1, StreamingRGBWriterProcessor(artifact.rgb_path, video_data.fps()))

        logger.info("Streaming RGB and dense depth artifacts to %s", artifact.base_path)
        depth_count = save_depth_artifacts_streaming(artifact.depth_path, output_stream)
        if depth_count != n_frames:
            raise RuntimeError(f"Expected {n_frames} depth frames, wrote {depth_count}")

        if self.out_cfg.save_viz:
            logger.warning(
                "--visualize-vipe is skipped by the bounded-memory writer; "
                "use the saved RGB/depth/pose artifacts for visualization"
            )

        return AnnotationPipelineOutput()
