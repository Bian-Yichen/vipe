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
import json
import logging
import os
import pickle
import shutil
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
from vipe.pipeline.processors import MultiviewDepthProcessor
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

from .artifacts import LOOP_CLOSURE_CACHE_VERSION

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

        if self.path.exists():
            # A resumed depth-only job can reuse the already committed RGB.
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


def _depth_parts_dir(path: Path) -> Path:
    return path.with_name(f"{path.name}.parts")


def _depth_metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_metadata.json")


def _shard_path(parts_dir: Path, start: int, end: int) -> Path:
    return parts_dir / f"{start:08d}_{end:08d}.zip"


def _zip_frame_indices(path: Path) -> list[int]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            if archive.testzip() is not None:
                return []
            return [int(name.rsplit("/", 1)[-1].split(".", 1)[0]) for name in sorted(archive.namelist())]
    except (OSError, ValueError, zipfile.BadZipFile):
        return []


def prepare_depth_shards(
    path: Path,
    expected_indices: list[int],
    shard_size: int,
    config: dict,
) -> tuple[Path, int]:
    """Return the current config's shard directory and contiguous completed prefix."""

    parts_dir = _depth_parts_dir(path)
    manifest_path = parts_dir / "manifest.json"
    if parts_dir.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            existing = None
        if existing != config:
            logger.warning("Discarding incompatible partial depth shards in %s", parts_dir)
            shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    completed = 0
    while completed < len(expected_indices):
        end = min(completed + shard_size, len(expected_indices))
        shard = _shard_path(parts_dir, completed, end)
        if _zip_frame_indices(shard) != expected_indices[completed:end]:
            break
        completed = end
    return parts_dir, completed


def _assemble_depth_archive(path: Path, parts_dir: Path, expected_indices: list[int], shard_size: int) -> None:
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as output:
            for start in range(0, len(expected_indices), shard_size):
                end = min(start + shard_size, len(expected_indices))
                shard = _shard_path(parts_dir, start, end)
                with zipfile.ZipFile(shard, "r") as source:
                    for name in sorted(source.namelist()):
                        output.writestr(name, source.read(name))
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def save_depth_artifacts_sharded(
    path: Path,
    stream: VideoStream,
    *,
    expected_indices: list[int],
    shard_size: int,
    parts_dir: Path,
    completed: int,
    frame_index_offset: int,
) -> int:
    """Commit resumable depth shards, then atomically publish the legacy ZIP."""

    path.parent.mkdir(exist_ok=True, parents=True)
    ordinal = completed
    archive: zipfile.ZipFile | None = None
    partial_shard: Path | None = None
    final_shard: Path | None = None
    try:
        for frame in stream:
            if frame.metric_depth is None:
                continue
            local_index = int(frame.raw_frame_idx) - int(frame_index_offset)
            if ordinal >= len(expected_indices) or local_index != expected_indices[ordinal]:
                raise RuntimeError(
                    f"Unexpected sparse depth frame {local_index}; expected ordinal {ordinal} "
                    f"({expected_indices[ordinal] if ordinal < len(expected_indices) else 'end'})"
                )
            if archive is None:
                shard_start = (ordinal // shard_size) * shard_size
                shard_end = min(shard_start + shard_size, len(expected_indices))
                final_shard = _shard_path(parts_dir, shard_start, shard_end)
                partial_shard = final_shard.with_name(f"{final_shard.name}.partial")
                partial_shard.unlink(missing_ok=True)
                archive = zipfile.ZipFile(partial_shard, "w", zipfile.ZIP_DEFLATED)

            depth = frame.metric_depth.detach().cpu().numpy().astype(np.float16, copy=False)
            with tempfile.NamedTemporaryFile(suffix=".exr") as temp:
                _write_depth_exr(depth, Path(temp.name))
                archive.write(temp.name, f"{local_index:08d}.exr")
            ordinal += 1
            if ordinal % shard_size == 0 or ordinal == len(expected_indices):
                archive.close()
                archive = None
                assert partial_shard is not None and final_shard is not None
                os.replace(partial_shard, final_shard)
            del depth, frame
    except BaseException:
        if archive is not None:
            archive.close()
        if partial_shard is not None:
            partial_shard.unlink(missing_ok=True)
        raise

    if ordinal != len(expected_indices):
        raise RuntimeError(f"Expected {len(expected_indices)} sparse depth frames, completed {ordinal}")
    _assemble_depth_archive(path, parts_dir, expected_indices, shard_size)
    shutil.rmtree(parts_dir)
    return ordinal


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

    loop_report = slam_output.loop_closure_report
    loop_enabled = loop_report is not None and bool(loop_report.get("enabled", False))
    loop_version = int(loop_report.get("version", 0)) if loop_enabled else 0
    loop_experiment = (
        str(loop_report.get("experiment_mode", "normal"))
        if loop_enabled
        else "normal"
    )
    with artifact.meta_info_path.open("wb") as handle:
        pickle.dump(
            {
                "ba_residual": slam_output.ba_residual,
                "loop_closure_enabled": loop_enabled,
                "loop_closure_version": loop_version,
                "loop_closure_experiment": loop_experiment,
                "loop_closure_report": loop_report,
            },
            handle,
        )
    report_path = artifact.meta_info_path.parent / f"{artifact.artifact_name}_loop_closure.json"
    if loop_report is not None:
        report_path.write_text(json.dumps(loop_report, indent=2, sort_keys=True) + "\n")
    else:
        report_path.unlink(missing_ok=True)

    slam_map = slam_output.slam_map
    if slam_map is None:
        raise RuntimeError("SLAM did not return a map")
    slam_map.save(artifact.slam_map_path)


def _load_slam_checkpoint(
    artifact: io.ArtifactPath,
    n_frames: int,
    *,
    expected_loop_closure: bool,
    expected_loop_experiment: str,
) -> SLAMOutput | None:
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
        loop_report = None
        saved_loop_closure = False
        saved_loop_version = 0
        saved_loop_experiment = "normal"
        if artifact.meta_info_path.exists():
            with artifact.meta_info_path.open("rb") as handle:
                meta_info = pickle.load(handle)
            ba_residual = float(meta_info.get("ba_residual", 0.0))
            saved_loop_closure = bool(meta_info.get("loop_closure_enabled", False))
            saved_loop_version = int(meta_info.get("loop_closure_version", 0))
            saved_loop_experiment = str(
                meta_info.get("loop_closure_experiment", "normal")
            )
            loop_report = meta_info.get("loop_closure_report")
        expected_loop_version = LOOP_CLOSURE_CACHE_VERSION if expected_loop_closure else 0
        if (
            saved_loop_closure != bool(expected_loop_closure)
            or saved_loop_version != expected_loop_version
            or (
                expected_loop_closure
                and saved_loop_experiment != expected_loop_experiment
            )
        ):
            logger.info(
                "Ignoring SLAM checkpoint for %s because loop-closure mode/version changed",
                artifact.artifact_name,
            )
            return None
        logger.info("Resuming dense-depth export from saved SLAM checkpoint for %s", artifact.artifact_name)
        return SLAMOutput(
            trajectory=trajectory,
            intrinsics=intrinsics,
            rig=SE3.Identity(1).cuda(),
            slam_map=slam_map,
            ba_residual=ba_residual,
            loop_closure_report=loop_report,
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

    def _depth_config(self) -> dict:
        return {
            "model": str(self.post_cfg.dav3_model),
            "model_path": self.post_cfg.dav3_model_path,
            "frame_step": int(self.post_cfg.depth_frame_step),
            "process_res": int(self.post_cfg.dav3_process_res),
            "process_res_method": str(self.post_cfg.dav3_process_res_method),
            "output_resolution": str(self.post_cfg.depth_output_resolution),
            "window_size": 10,
            "overlap_size": 3,
            "slam_loop_closure": bool(self.slam_cfg.loop_closure.enabled),
            "slam_loop_closure_version": (
                LOOP_CLOSURE_CACHE_VERSION if self.slam_cfg.loop_closure.enabled else 0
            ),
            "slam_loop_experiment": (
                str(self.slam_cfg.loop_closure.experiment_mode)
                if self.slam_cfg.loop_closure.enabled
                else "normal"
            ),
        }

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        if self.post_cfg.depth_align_model != "mvd_dav3":
            return super()._add_post_processors(view_idx, video_stream, slam_output)
        processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            ),
            MultiviewDepthProcessor(
                slam_output,
                model="mvd_dav3",
                window_size=10,
                overlap_size=3,
                frame_step=int(self.post_cfg.depth_frame_step),
                process_res=int(self.post_cfg.dav3_process_res),
                process_res_method=str(self.post_cfg.dav3_process_res_method),
                output_resolution=str(self.post_cfg.depth_output_resolution),
                dav3_model=str(self.post_cfg.dav3_model),
                dav3_model_path=self.post_cfg.dav3_model_path,
                inference_start_ordinal=int(self.post_cfg.depth_inference_start_ordinal),
                emit_start_ordinal=int(self.post_cfg.depth_emit_start_ordinal),
            ),
        ]
        return ProcessedVideoStream(video_stream, processors)

    def _load_or_run_slam(
        self,
        video_data: VideoStream,
        artifact: io.ArtifactPath,
        *,
        require_checkpoint: bool,
    ) -> tuple[VideoStream, SLAMOutput]:
        n_frames = len(video_data)
        slam_output = _load_slam_checkpoint(
            artifact,
            n_frames,
            expected_loop_closure=bool(self.slam_cfg.loop_closure.enabled),
            expected_loop_experiment=str(
                self.slam_cfg.loop_closure.experiment_mode
            ),
        )
        if slam_output is None:
            if require_checkpoint:
                raise FileNotFoundError(
                    f"Depth-only mode requires a complete SLAM checkpoint for {artifact.artifact_name}"
                )
            slam_stream = self._add_init_processors(video_data)
            slam_pipeline = SLAMSystem(
                device=torch.device("cuda"),
                config=self.slam_cfg,
                model_cache=self.model_cache,
            )
            slam_output = slam_pipeline.run([slam_stream], rig=None, camera_type=self.camera_type)
            _save_slam_checkpoint(artifact, slam_output, n_frames)
            _move_slam_map_to_cpu(slam_output)
            self.model_cache.clear()
            del slam_pipeline
            gc.collect()
            torch.cuda.empty_cache()
        else:
            slam_stream = ProcessedVideoStream(
                video_data,
                [AssignAttributesProcessor({FrameAttribute.CAMERA_TYPE: [self.camera_type] * n_frames})],
            )
        return slam_stream, slam_output

    def run_mode(
        self,
        video_data: VideoStream | MultiviewVideoList,
        *,
        mode: str = "full",
        frame_index_offset: int = 0,
    ) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            raise NotImplementedError("The room-tour streaming pipeline currently supports one video stream")
        if mode not in {"full", "pose", "depth"}:
            raise ValueError(f"Unsupported room-tour inference mode: {mode}")

        artifact = io.ArtifactPath(self.out_path, video_data.name())
        n_frames = len(video_data)
        slam_stream, slam_output = self._load_or_run_slam(
            video_data,
            artifact,
            require_checkpoint=mode == "depth",
        )
        if mode == "pose":
            logger.info("Pose-only stage complete for %s; dense DAv3 was not loaded", artifact.artifact_name)
            return AnnotationPipelineOutput()

        config = self._depth_config()
        expected_indices = list(range(0, n_frames, int(self.post_cfg.depth_frame_step)))
        metadata_path = _depth_metadata_path(artifact.depth_path)
        if artifact.depth_path.exists() and artifact.rgb_path.exists():
            legacy_quality = {
                "model": "giant",
                "model_path": None,
                "frame_step": 1,
                "process_res": 504,
                "process_res_method": "lower_bound_resize",
                "output_resolution": "original",
                "window_size": 10,
                "overlap_size": 3,
                "slam_loop_closure": False,
                "slam_loop_closure_version": 0,
                "slam_loop_experiment": "normal",
            }
            try:
                saved_config = json.loads(metadata_path.read_text()) if metadata_path.exists() else legacy_quality
                if saved_config == config:
                    logger.info("Matching dense-depth artifacts already exist for %s", artifact.artifact_name)
                    return AnnotationPipelineOutput()
            except (OSError, ValueError):
                pass

        parts_dir, completed = prepare_depth_shards(
            artifact.depth_path,
            expected_indices,
            int(self.post_cfg.depth_shard_size),
            {**config, "shard_size": int(self.post_cfg.depth_shard_size)},
        )
        window_stride = 10 - 3
        inference_start = max(0, completed - window_stride) if completed else 0
        self.post_cfg.depth_inference_start_ordinal = inference_start
        self.post_cfg.depth_emit_start_ordinal = completed
        if completed:
            logger.info(
                "Resuming DAv3 at sparse depth %d/%d (context begins at %d)",
                completed,
                len(expected_indices),
                inference_start,
            )
        if completed == len(expected_indices) and artifact.rgb_path.exists():
            _assemble_depth_archive(
                artifact.depth_path,
                parts_dir,
                expected_indices,
                int(self.post_cfg.depth_shard_size),
            )
            shutil.rmtree(parts_dir)
            metadata_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            logger.info("Published depth archive from completed shards without rerunning DAv3")
            return AnnotationPipelineOutput()

        output_stream = self._add_post_processors(0, slam_stream, slam_output)
        # This processor writes RGB during MultiviewDepthProcessor's mandatory
        # keyframe-recording pass, so it adds no extra decode pass.
        output_stream.processors.insert(1, StreamingRGBWriterProcessor(artifact.rgb_path, video_data.fps()))

        logger.info("Streaming RGB and dense depth artifacts to %s", artifact.base_path)
        depth_count = save_depth_artifacts_sharded(
            artifact.depth_path,
            output_stream,
            expected_indices=expected_indices,
            shard_size=int(self.post_cfg.depth_shard_size),
            parts_dir=parts_dir,
            completed=completed,
            frame_index_offset=frame_index_offset,
        )
        if depth_count != len(expected_indices):
            raise RuntimeError(f"Expected {len(expected_indices)} depth frames, wrote {depth_count}")
        metadata_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

        if self.out_cfg.save_viz:
            logger.warning(
                "--visualize-vipe is skipped by the bounded-memory writer; "
                "use the saved RGB/depth/pose artifacts for visualization"
            )

        return AnnotationPipelineOutput()

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        return self.run_mode(video_data)
