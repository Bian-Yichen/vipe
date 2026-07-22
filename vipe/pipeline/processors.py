# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, cast

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from vipe.ext.lietorch import SE3, SO3
from vipe.priors.depth import DepthEstimationInput, make_depth_model
from vipe.priors.depth.alignment import align_inv_depth_to_depth
from vipe.priors.depth.priorda import PriorDAModel
from vipe.priors.depth.videodepthanything import VideoDepthAnythingDepthModel
from vipe.priors.geocalib import GeoCalib
from vipe.priors.track_anything import TrackAnythingPipeline
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import CachedVideoStream, FrameAttribute, StreamProcessor, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.depth import get_camera_rays
from vipe.utils.geometry import project_points_to_panorama
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional
from vipe.utils.model_cache import ModelCache
from vipe.utils.morph import erode

logger = logging.getLogger(__name__)


class IntrinsicEstimationProcessor(StreamProcessor):
    """Override existing intrinsics with estimated intrinsics."""

    def __init__(self, video_stream: VideoStream, gap_sec: float = 1.0) -> None:
        super().__init__()
        gap_frame = int(gap_sec * video_stream.fps())
        gap_frame = min(gap_frame, (len(video_stream) - 1) // 2)
        self.sample_frame_inds = [0, gap_frame, gap_frame * 2]
        self.fov_y = -1.0
        self.camera_type = CameraType.PINHOLE
        self.distortion: list[float] = []

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        assert self.fov_y > 0, "FOV not set"
        frame_height, frame_width = frame.size()
        fx = fy = frame_height / (2 * np.tan(self.fov_y / 2))
        frame.intrinsics = torch.as_tensor(
            [fx, fy, frame_width / 2, frame_height / 2] + self.distortion,
        ).float()
        frame.camera_type = self.camera_type
        return frame


class GeoCalibIntrinsicsProcessor(IntrinsicEstimationProcessor):
    def __init__(
        self,
        video_stream: VideoStream,
        gap_sec: float = 1.0,
        camera_type: CameraType = CameraType.PINHOLE,
        model_cache: ModelCache | None = None,
    ) -> None:
        super().__init__(video_stream, gap_sec)

        is_pinhole = camera_type == CameraType.PINHOLE
        weights = "pinhole" if is_pinhole else "distorted"

        # GeoCalib is used purely for inference; when a cache is provided the
        # weights are loaded once and reused across streams instead of per video.
        def _build_geocalib():
            return GeoCalib(weights=weights).cuda()

        if model_cache is not None:
            model = model_cache.get(f"geocalib/{weights}", _build_geocalib)
        else:
            model = _build_geocalib()
        indexable_stream = CachedVideoStream(video_stream)

        if is_pinhole:
            sample_frames = torch.stack([indexable_stream[i].rgb.moveaxis(-1, 0) for i in self.sample_frame_inds])
            res = model.calibrate(
                sample_frames,
                shared_intrinsics=True,
            )
        else:
            # Use first frame for calibration
            camera_model = {
                CameraType.PINHOLE: "pinhole",
                CameraType.MEI: "simple_mei",
            }[camera_type]
            res = model.calibrate(
                indexable_stream[self.sample_frame_inds[0]].rgb.moveaxis(-1, 0)[None],
                camera_model=camera_model,
            )

        camera_result = cast(Any, res["camera"])
        self.fov_y = camera_result.vfov[0].item()
        self.camera_type = camera_type

        if not is_pinhole:
            # Assign distortion parameter
            self.distortion = [camera_result.dist[0, 0].item()]


class TrackAnythingProcessor(StreamProcessor):
    """
    A processor that tracks a mask caption in the video.
    """

    def __init__(
        self,
        mask_phrases: list[str],
        add_sky: bool,
        sam_run_gap: int = 30,
        mask_expand: int = 5,
        model_cache: ModelCache | None = None,
    ) -> None:
        # Defensive copy: prevent mutation of caller's list
        self.mask_phrases = list(mask_phrases)
        self.sam_run_gap = sam_run_gap
        self.add_sky = add_sky

        if self.add_sky:
            self.mask_phrases.append(VideoFrame.SKY_PROMPT)

        self.tracker = TrackAnythingPipeline(
            self.mask_phrases,
            sam_points_per_side=50,
            sam_run_gap=self.sam_run_gap,
            model_cache=model_cache,
        )
        self.mask_expand = mask_expand
        self.aot_encoder_batch_size = max(1, int(os.environ.get("VIPE_TRACK_ANYTHING_AOT_ENCODER_BATCH_SIZE", "4")))

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INSTANCE, FrameAttribute.MASK}

    def _process_frame(
        self,
        frame_idx: int,
        frame: VideoFrame,
        aot_img_embs: tuple[torch.Tensor, ...] | None = None,
    ) -> VideoFrame:
        frame.instance, frame.instance_phrases = self.tracker.track(frame, aot_img_embs=aot_img_embs)
        self.last_track_frame = frame.raw_frame_idx

        frame_instance_mask = frame.instance == 0
        if self.add_sky:
            # We won't mask out the sky.
            frame_instance_mask |= frame.sky_mask

        frame.mask = erode(frame_instance_mask, self.mask_expand)
        return frame

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        return self._process_frame(frame_idx, frame)

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if self.aot_encoder_batch_size <= 1:
            for frame_idx, frame in enumerate(previous_iterator):
                yield self._process_frame(frame_idx, frame)
            return

        pending_indices: list[int] = []
        pending_frames: list[VideoFrame] = []

        def flush_pending() -> Iterator[VideoFrame]:
            nonlocal pending_indices, pending_frames
            if not pending_frames:
                return
            embeddings = self.tracker.encode_aot_frames(pending_frames)
            for pending_idx, pending_frame, aot_img_embs in zip(pending_indices, pending_frames, embeddings):
                yield self._process_frame(pending_idx, pending_frame, aot_img_embs=aot_img_embs)
            pending_indices = []
            pending_frames = []

        for frame_idx, frame in enumerate(previous_iterator):
            if self.tracker.should_batch_aot_frame(len(pending_frames)):
                pending_indices.append(frame_idx)
                pending_frames.append(frame)
                if len(pending_frames) >= self.aot_encoder_batch_size:
                    yield from flush_pending()
                continue

            yield from flush_pending()
            yield self._process_frame(frame_idx, frame)

        yield from flush_pending()


class AdaptiveDepthProcessor(StreamProcessor):
    """
    Compute projection of the SLAM map onto the current frames.
    If it's well-distributed, then use the fast map-prompted video depth model.
    If not, then use the slow metric depth + video depth alignment model.
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        view_idx: int = 0,
        model: str = "adaptive_unidepth-l_svda",
        share_depth_model: bool = False,
    ):
        super().__init__()
        self.slam_output = slam_output
        self.infill_target_pose = self.slam_output.get_view_trajectory(view_idx)
        assert view_idx == 0, "Adaptive depth processor only supports view_idx=0"
        assert not share_depth_model, "Adaptive depth processor does not support shared depth model"
        self.require_cache = True
        self.model = model

        try:
            prefix, metric_model, video_model = model.split("_")
            assert video_model in ["svda", "vda"]
            self.video_depth_model: VideoDepthAnythingDepthModel | None = VideoDepthAnythingDepthModel(
                model="vits" if video_model == "svda" else "vitl"
            )

        except ValueError:
            prefix, metric_model = model.split("_")
            video_model = None
            self.video_depth_model = None

        assert prefix == "adaptive", "Model name should start with 'adaptive_'"

        self.depth_model = make_depth_model(metric_model)
        self.prompt_model = PriorDAModel()
        self.update_momentum = 0.99

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("AdaptiveDepthProcessor should not be called directly.")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def _compute_uv_score(self, depth: torch.Tensor, patch_count: int = 10) -> float:
        h_shape = depth.size(0) // patch_count
        w_shape = depth.size(1) // patch_count
        depth_crop = (depth > 0)[: h_shape * patch_count, : w_shape * patch_count]
        depth_crop = depth_crop.reshape(patch_count, h_shape, patch_count, w_shape)
        depth_exist = depth_crop.any(dim=(1, 3))
        return depth_exist.float().mean().item()

    def _compute_video_da(self, frame_iterator: Iterator[VideoFrame]) -> tuple[torch.Tensor, list[VideoFrame]]:
        frame_list: list[np.ndarray] = []
        frame_data_list: list[VideoFrame] = []
        for frame in frame_iterator:
            frame_data_list.append(frame.cpu())
            frame_list.append(frame.rgb.cpu().numpy())

        video_depth_model = unpack_optional(self.video_depth_model)
        video_depth_result: torch.Tensor = unpack_optional(
            video_depth_model.estimate(DepthEstimationInput(video_frame_list=frame_list)).relative_inv_depth
        )
        return video_depth_result, frame_data_list

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        # Determine the percentage score of the SLAM map.

        self.cache_scale_bias: tuple[torch.Tensor, torch.Tensor] | None = None
        min_uv_score: float = 1.0
        slam_map = unpack_optional(self.slam_output.slam_map)
        data_iterator: Iterable[VideoFrame]

        if self.video_depth_model is not None:
            video_depth_result, data_iterator = self._compute_video_da(previous_iterator)
        else:
            video_depth_result = None
            data_iterator = previous_iterator

        for frame_idx, frame in pbar(enumerate(data_iterator), desc="Aligning depth"):
            # Convert back to GPU if not already.
            frame = frame.cuda()

            # Compute the minimum UV score only once at the 0-th frame.
            if frame_idx == 0:
                for test_frame_idx in range(self.slam_output.trajectory.shape[0]):
                    if test_frame_idx % 10 != 0:
                        continue
                    depth_infilled = slam_map.project_map(
                        test_frame_idx,
                        0,
                        frame.size(),
                        unpack_optional(frame.intrinsics),
                        self.infill_target_pose[test_frame_idx],
                        unpack_optional(frame.camera_type),
                        infill=False,
                    )
                    uv_score = self._compute_uv_score(depth_infilled)
                    if uv_score < min_uv_score:
                        min_uv_score = uv_score

                logger.info(f"Minimum UV score: {min_uv_score:.4f}")

            if min_uv_score < 0.3:
                prompt_result = self.depth_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(), intrinsics=frame.intrinsics, camera_type=frame.camera_type
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(Metric)"
            else:
                depth_map = slam_map.project_map(
                    frame_idx,
                    0,
                    frame.size(),
                    unpack_optional(frame.intrinsics),
                    self.infill_target_pose[frame_idx],
                    unpack_optional(frame.camera_type),
                    infill=False,
                )
                if frame.mask is not None:
                    depth_map = depth_map * frame.mask.float()
                prompt_result = self.prompt_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(),
                        prompt_metric_depth=depth_map,
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(SLAM)"

            if video_depth_result is not None:
                video_depth_inv_depth = video_depth_result[frame_idx]

                align_mask = video_depth_inv_depth > 1e-3
                if frame.mask is not None:
                    align_mask = align_mask & frame.mask & (~frame.sky_mask)

                try:
                    _, scale_tensor, bias_tensor = align_inv_depth_to_depth(
                        unpack_optional(video_depth_inv_depth),
                        prompt_result,
                        align_mask,
                    )
                except RuntimeError:
                    if self.cache_scale_bias is None:
                        raise
                    scale_tensor, bias_tensor = self.cache_scale_bias

                # momentum update
                if self.cache_scale_bias is None:
                    self.cache_scale_bias = (scale_tensor, bias_tensor)
                scale_tensor = self.cache_scale_bias[0] * self.update_momentum + scale_tensor * (
                    1 - self.update_momentum
                )
                bias_tensor = self.cache_scale_bias[1] * self.update_momentum + bias_tensor * (1 - self.update_momentum)
                self.cache_scale_bias = (scale_tensor, bias_tensor)

                video_inv_depth = video_depth_inv_depth * scale_tensor + bias_tensor
                video_inv_depth[video_inv_depth < 1e-3] = 1e-3
                frame.metric_depth = video_inv_depth.reciprocal()

            else:
                frame.metric_depth = prompt_result

            yield frame


class MultiviewDepthProcessor(StreamProcessor):
    """
    Use multi-view depth model (e.g. DAv3, MapAnything, CAPA) to estimate depth map for each frame.
    To ensure that the depth maps are consistent with the SLAM map/pose (metric), we condition the depth model either with
    (a) sparse points, or (b) camera poses & intrinsics.

    Depth is estimated in a sliding-window manner, and overlapped frames are linearly averaged to sharp transitions.
    To create enough parallex to improve estimation confidence, for each window we optionally also include
    neighboring keyframes, and their secondary neighboring keyframes.
    (Multi-view input video frames are currently not supported)
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        model: str = "mvd_dav3",
        window_size: int = 10,  # Practically this should be as large as possible if memory permits.
        overlap_size: int = 3,
        secondary_keyframe: bool = False,  # This is found to cause jittering for some scenes due to abrupt context changes.
        frame_step: int = 1,
        process_res: int = 504,
        process_res_method: str = "lower_bound_resize",
        output_resolution: str = "original",
        dav3_model: str = "giant",
        dav3_model_path: str | None = None,
        inference_start_ordinal: int = 0,
        emit_start_ordinal: int = 0,
    ):
        super().__init__()
        self.slam_output = slam_output
        self.model = model
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.secondary_keyframe = secondary_keyframe
        self.frame_step = int(frame_step)
        self.process_res = int(process_res)
        self.process_res_method = process_res_method
        self.output_resolution = output_resolution
        self.inference_start_ordinal = int(inference_start_ordinal)
        self.emit_start_ordinal = int(emit_start_ordinal)
        if self.frame_step < 1:
            raise ValueError("frame_step must be >= 1")
        if not 0 <= self.inference_start_ordinal <= self.emit_start_ordinal:
            raise ValueError("Invalid DAv3 resume ordinals")

        self.keyframes_inds = unpack_optional(self.slam_output.slam_map).dense_disp_frame_inds
        self.keyframes_data: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        self.n_frames = 0

        # Need two passes for this iterator to work.
        self.n_passes_required = 2

        if self.model == "mvd_dav3":
            from vipe.priors.depth.dav3 import DepthAnything3
            from vipe.priors.depth.dav3.utils import logger as dav3_logger

            dav3_logger.level = 0  # Disable logging timing information
            checkpoints = {
                "giant": ("depth-anything/DA3-GIANT", "da3-giant"),
                "large": ("depth-anything/DA3-LARGE", "da3-large"),
                "base": ("depth-anything/DA3-BASE", "da3-base"),
                "small": ("depth-anything/DA3-SMALL", "da3-small"),
            }
            if dav3_model not in checkpoints:
                raise ValueError(f"Unsupported DAv3 model: {dav3_model}")
            checkpoint, model_name = checkpoints[dav3_model]
            weights_path = None
            if dav3_model_path is not None:
                local_checkpoint = Path(dav3_model_path).expanduser()
                if local_checkpoint.is_dir():
                    local_checkpoint = local_checkpoint / "model.safetensors"
                if not local_checkpoint.is_file():
                    raise FileNotFoundError(
                        "Local DAv3 checkpoint must be a model.safetensors file or a directory "
                        f"containing it: {dav3_model_path}"
                    )
                weights_path = str(local_checkpoint)
            self.dav3_api = DepthAnything3.from_pretrained(
                checkpoint,
                model_name=model_name,
                weights_path=weights_path,
            )
            self.dav3_api = self.dav3_api.cuda().eval()

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("MultiviewDepthProcessor should not be called directly.")

    def _probe_keyframe_indices(self, frame_idx: int) -> list[int]:
        inds: list[int] = []
        left_idx = np.searchsorted(self.keyframes_inds, frame_idx, side="right").item() - 1
        inds.append(left_idx)
        if frame_idx < self.keyframes_inds[-1]:
            inds.append(left_idx + 1)
        # Pick the farthest secondary keyframe from the left keyframe.
        if self.secondary_keyframe:
            slam_graph = unpack_optional(self.slam_output.slam_map).backend_graph
            if slam_graph is not None:
                matching_secondary_j = slam_graph[slam_graph[:, 0] == left_idx, 1].tolist()
                picked_sj_idx = np.argmax([abs(self.keyframes_inds[j] - frame_idx) for j in matching_secondary_j])
                inds.append(matching_secondary_j[picked_sj_idx])
        return inds

    def record_keyframes(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        for frame_idx, frame in enumerate(previous_iterator):
            self.n_frames += 1
            if frame_idx in self.keyframes_inds:
                # Store exactly the uint8 RGB / w2c / K values consumed by
                # DAv3 instead of retaining a full float32 VideoFrame on GPU.
                self.keyframes_data.append(frame.dav3_conditions())
            yield frame

    def _estimate_window(
        self,
        current_sliding_window: list[VideoFrame],
        current_sliding_window_idx: list[int],
        trailing_depth: torch.Tensor | None,
        *,
        is_last_window: bool,
    ) -> tuple[list[VideoFrame], list[int], torch.Tensor | None, list[tuple[VideoFrame, int]]]:
        """Infer one unchanged-overlap DAv3 window and return frames ready to emit."""

        sw_keyframe_inds = list(
            set(sum([self._probe_keyframe_indices(i) for i in current_sliding_window_idx], []))
        )
        sw_keyframe_inds = [
            t for t in sw_keyframe_inds if self.keyframes_inds[t] not in current_sliding_window_idx
        ]
        sw_images, sw_exts, sw_ints = zip(*[frame.dav3_conditions() for frame in current_sliding_window])
        if sw_keyframe_inds:
            kf_images, kf_exts, kf_ints = zip(*[self.keyframes_data[t] for t in sw_keyframe_inds])
        else:
            kf_images, kf_exts, kf_ints = tuple(), tuple(), tuple()

        result = self.dav3_api.inference(
            list(sw_images + kf_images),
            extrinsics=np.stack(sw_exts + kf_exts, axis=0),
            intrinsics=np.stack(sw_ints + kf_ints, axis=0),
            process_res=self.process_res,
            process_res_method=self.process_res_method,
        )
        sw_depth = torch.from_numpy(result.depth[: len(sw_images)]).float().cuda()
        if self.output_resolution == "original":
            sw_depth = torch.nn.functional.interpolate(
                sw_depth[:, None], current_sliding_window[0].size(), mode="bilinear"
            )[:, 0]

        n_frames_to_yield = (
            len(current_sliding_window)
            if is_last_window
            else self.window_size - self.overlap_size
        )
        if trailing_depth is not None:
            n_interp_frames = len(trailing_depth)
            alpha = torch.linspace(0, 1, n_interp_frames + 2, device=sw_depth.device)[1:-1, None, None]
            sw_depth[:n_interp_frames] = trailing_depth * (1 - alpha) + sw_depth[:n_interp_frames] * alpha

        ready_frames = current_sliding_window[:n_frames_to_yield]
        ready = list(zip(ready_frames, current_sliding_window_idx[:n_frames_to_yield], strict=True))
        for sw_idx, (ready_frame, _) in enumerate(ready):
            ready_frame.metric_depth = sw_depth[sw_idx]
        return (
            current_sliding_window[n_frames_to_yield:],
            current_sliding_window_idx[n_frames_to_yield:],
            sw_depth[n_frames_to_yield:],
            ready,
        )

    def estimate_depth_sliding_window(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        current_sliding_window: list[VideoFrame] = []
        current_sliding_window_idx: list[int] = []
        trailing_depth: torch.Tensor | None = None
        last_selected_index = ((self.n_frames - 1) // self.frame_step) * self.frame_step
        for frame_idx, frame in pbar(enumerate(previous_iterator), desc="Estimating multi-view depth"):
            if frame_idx % self.frame_step:
                continue
            selected_ordinal = frame_idx // self.frame_step
            if selected_ordinal < self.inference_start_ordinal:
                continue
            current_sliding_window.append(frame)
            current_sliding_window_idx.append(frame_idx)
            is_last_selected = frame_idx == last_selected_index
            if len(current_sliding_window) == self.window_size or is_last_selected:
                current_sliding_window, current_sliding_window_idx, trailing_depth, ready = self._estimate_window(
                    current_sliding_window,
                    current_sliding_window_idx,
                    trailing_depth,
                    is_last_window=is_last_selected,
                )
                for ready_frame, ready_index in ready:
                    if ready_index // self.frame_step >= self.emit_start_ordinal:
                        yield ready_frame

        if current_sliding_window:
            # Defensive fallback for a truncated iterator whose declared length
            # exceeded the frames actually decoded.
            current_sliding_window, current_sliding_window_idx, trailing_depth, ready = self._estimate_window(
                current_sliding_window,
                current_sliding_window_idx,
                trailing_depth,
                is_last_window=True,
            )
            for ready_frame, ready_index in ready:
                if ready_index // self.frame_step >= self.emit_start_ordinal:
                    yield ready_frame
        assert not current_sliding_window, "Current sliding window should be empty"

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if pass_idx == 0:
            yield from self.record_keyframes(previous_iterator)
        elif pass_idx == 1:
            yield from self.estimate_depth_sliding_window(previous_iterator)
        else:
            raise ValueError(f"Invalid pass index: {pass_idx}")


class EquirectProjectionProcessor(StreamProcessor):
    """
    Camera convention (with rotation = I, up of panorama is outward, Y is inward):
       -----
      (  Z  )
     (   |   )
    (    Y-X  )
     (       )
      (     )
       -<|>-
         |
    [boundary of image]
    """

    def __init__(self, rotation: SO3, frame_size: tuple[int, int], intrinsics: torch.Tensor) -> None:
        super().__init__()
        self.rotation = rotation.cuda()
        self.intrinsics = intrinsics.cuda()
        rays = get_camera_rays(frame_size[0], frame_size[1], self.intrinsics, normalize=True)
        rays = unpack_optional(self.rotation[None, None].act(rays))
        uv = project_points_to_panorama(rays, return_depth=False)
        self.uv = (uv * 2) - 1
        self.frame_size = frame_size

    @staticmethod
    def yaw_pitch_to_rotation(yaw: float, pitch: float) -> SO3:
        """
        First rotate around yaw, then pitch (positive is heads up, negative is down).
        """
        return SO3.InitFromVec(
            torch.from_numpy(R.from_euler("xyz", [pitch, yaw, 0], degrees=False).as_quat(canonical=True)).float()
        )

    def update_frame_size(self, previous_frame_size: tuple[int, int]):
        return self.frame_size

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        assert frame.metric_depth is None, "Metric depth is not supported for equirect projection"

        if (new_pose := frame.pose) is not None:
            rel_transform = SE3.InitFromVec(torch.cat((torch.zeros(3).cuda(), self.rotation.data)))
            new_pose = new_pose * rel_transform

        new_rgb = (
            torch.nn.functional.grid_sample(frame.rgb.moveaxis(-1, 0)[None], self.uv[None], align_corners=True)
            .squeeze()
            .moveaxis(0, -1)
        )

        if (new_instance := frame.instance) is not None:
            new_instance = torch.nn.functional.grid_sample(
                new_instance[None, None].float(), self.uv[None], align_corners=True, mode="nearest"
            )[0, 0]

        if (new_mask := frame.mask) is not None:
            new_mask = torch.nn.functional.grid_sample(
                new_mask[None, None].float(), self.uv[None], align_corners=True, mode="nearest"
            )[0, 0]

        return VideoFrame(
            raw_frame_idx=frame.raw_frame_idx,
            rgb=new_rgb,
            pose=new_pose,
            intrinsics=self.intrinsics.clone(),
            camera_type=CameraType.PINHOLE,
            instance=new_instance,
            mask=new_mask,
        )
