# Static room-tour calibration and top-down mapping

This extension runs ViPE's full pose/intrinsics/depth pipeline, then uses the
estimated RGB-D frames to create a globally fused point cloud and a leveled
bird's-eye QA image. It is intended for static, continuous room-tour videos.

## What is exported

For each full video or manually selected segment, `OUTPUT/maps/NAME/` contains:

| File | Meaning |
| --- | --- |
| `calibration.npz` | Every calibrated frame's original-resolution intrinsics, 3x3 `K`, OpenCV `c2w`, inverse `w2c`, and timestamps |
| `per_frame_calibration.csv` | The same critical calibration data in a flat, inspectable table |
| `coordinate_convention.json` | Exact matrix and camera-axis convention |
| `trajectory.csv` | Camera centers in original world and gravity-leveled coordinates |
| `global_rgb_map_world.ply` | Fused colored 3D cloud in ViPE world coordinates |
| `global_rgb_map_leveled.ply` | The same cloud with gravity leveled for inspection |
| `topdown_rgb.png` | RGB bird's-eye projection |
| `topdown_occupancy.png` | Log-density bird's-eye projection |
| `topdown_with_trajectory.png` | 1 m grid, global scene content, trajectory, and start/end markers |
| `quality_metrics.json` | Path, camera-height, tilt, voxel support, BA residual (when present), and raster statistics |
| `quality_report.md` | One-page entry point for visual inspection |

Native VIPE artifacts remain in `OUTPUT/vipe_artifacts/`: RGB, per-frame poses,
intrinsics, half-float EXR depth zip, diagnostic metadata, and the SLAM map.

The room-tour entry point writes these artifacts with bounded memory. It does
not cache every RGB-D frame. Pose, intrinsics, and the SLAM map are committed
before post-SLAM DAv3 starts. Dense depth is then committed in resumable shards
and assembled into the normal EXR ZIP after success. The existing DAv3 temporal
window remains 10 frames with 3 overlap frames in every preset.

## Install

The commands below assume Linux, an NVIDIA CUDA GPU, Git, and FFmpeg. From this
branch of the repository:

```bash
git clone https://github.com/Bian-Yichen/vipe.git
cd vipe
git checkout agent/roomtour-intrachunk-loop-closure

# Use the same ViPE installation method/environment you normally use.
# Editable install exposes both commands. Hydra is not needed by vipe-roomtour.
python -m pip install -e .
```

If the environment already contains ViPE's CUDA/model dependencies and only an
editable command registration is needed, use:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

`vipe-roomtour` now constructs the static pipeline directly with OmegaConf and
does not import Hydra. Hydra is no longer a default project dependency. The
original upstream compositional `vipe infer` command is retained for
compatibility; install Hydra separately only if that command is needed:

```bash
python -m pip install hydra-core
```

`python-pycg` is also optional for this workflow. The bounded-memory room-tour
path does not build ViPE's legacy diagnostic video; use the exported PLY and
top-down PNG files for QA instead.

ViPE downloads model weights on first use. The `roomtour_dav3` pipeline uses
Depth Anything 3 and is the recommended high-quality setting. Review the
third-party model licenses described by upstream VIPE before large-scale use.
`quality` uses DA3-GIANT as before. `preview` and `balanced` use the
pose-conditioned multi-view DA3-LARGE model (not DA3METRIC-LARGE), so that
checkpoint must also be downloaded/cached before an offline compute job.

## Run one complete video

`OUTPUT` must be an absolute path (or begin with `~/`). Relative outputs such
as `output` are rejected so results cannot accidentally be written inside the
current ViPE checkout.

```bash
vipe-roomtour run /data/video.mp4 /mnt/petrelfs/user/results/video_001 \
  --pipeline roomtour_dav3
```

For rapid calibration QA, stop before the expensive all-frame dense-depth
stage. This still exports original-resolution intrinsics and c2w/w2c for every
frame:

```bash
vipe-roomtour run /data/video.mp4 /mnt/petrelfs/user/results/video_001 \
  --pipeline roomtour_dav3 --pose-only
```

Later, reuse that exact SLAM checkpoint and add preview depth/map without
re-estimating pose:

```bash
vipe-roomtour run /data/video.mp4 /mnt/petrelfs/user/results/video_001 \
  --pipeline roomtour_dav3 --depth-only --depth-preset preview
```

## Long tours: overlapping chunk solve and Sim(3) stitch

For long trajectories that are less stable as one global SLAM solve, run each
overlapping block independently and align adjacent blocks through their shared
source frames:

```bash
vipe-roomtour chunked-run /data/villa_25000_frames.mp4 /mnt/petrelfs/user/results/villa_chunked \
  --pipeline roomtour_dav3 \
  --chunk-frames 5000 \
  --overlap-frames 500
```

The chunked command now runs in three explicit phases: all chunk SLAM solves,
pose-overlap validation, then dense depth/map. A fast pose-only pass is:

```bash
vipe-roomtour chunked-run /data/villa_25000_frames.mp4 /mnt/petrelfs/user/results/villa_chunked \
  --chunk-frames 5000 --overlap-frames 500 --pose-only
```

To continue with the fast preview preset on four allocated GPUs:

```bash
srun -p vcg --gres=gpu:4 \
  vipe-roomtour chunked-run /data/villa_25000_frames.mp4 /mnt/petrelfs/user/results/villa_chunked \
  --chunk-frames 5000 --overlap-frames 500 \
  --depth-only --depth-preset preview --depth-workers 4
```

Each worker owns one visible GPU and processes independent chunk DAv3 jobs.
This parallelism does not alter chunk poses or the DAv3 3/10 overlap blending.

The half-open blocks are `[0,5000)`, `[4500,9500)`, and so on. They are read
directly from the original MP4; no lossy intermediate clip is created. Every
block uses the bounded-memory SLAM/DAv3 writer described above.

Adjacent blocks have independent coordinate gauges. The stitcher therefore
does not concatenate their matrices directly. It uses the matching camera
centres and orientations in the 500 shared frames to robustly estimate a
Sim(3): rotation, translation, and scale. The scale is applied to both the
chunk point cloud and its camera translations. Duplicate poses in the overlap
are rotation/translation blended so the final per-frame trajectory is smooth.

The extra model work is approximately
`overlap / (chunk_size - overlap)` (11.1% for 500/5000). Pose-based alignment
itself is negligible and is safer than unconstrained ICP on repetitive walls.
The command rejects a stitch if the overlap has too little camera motion, an
implausible scale ratio, excessive position RMSE, or excessive rotation error.
Move the boundary or increase overlap rather than disabling those checks.

Important outputs are:

- `OUTPUT/chunks/CHUNK_NAME/`: each independently calibrated chunk map.
- `OUTPUT/vipe_artifacts/`: each chunk's RGB, depth, pose, intrinsics, and SLAM map.
- `OUTPUT/global/calibration.npz`: continuous full-video frame indices, intrinsics, and stitched c2w/w2c.
- `OUTPUT/global/global_rgb_map_world.ply`: all chunk clouds in the chunk-0 world/scale.
- `OUTPUT/global/topdown_with_trajectory.png`: global stitched QA view.
- `OUTPUT/global/chunk_transforms.json`: local-to-global Sim(3) and depth scale for every chunk.
- `OUTPUT/global/stitch_metrics.json`: overlap baseline, scale, position RMSE, and rotation error.
- `OUTPUT/global/per_frame_chunk_assignment.csv`: primary chunk plus its exact
  depth-to-global scale (and the overlap-blended scale) for every source frame.

This strategy removes long-horizon drift inside one monolithic solve, but it
cannot by itself repair a bad pose estimate inside a 5000-frame chunk. The
experimental loop-closure mode below addresses revisits inside each chunk.
Pairwise chunk alignment can still accumulate slowly across many chunks.

## Experimental intrachunk loop closure

Use `--loop-closure` when the camera sees the same stair landing, doorway, or
room again after a long delay inside one SLAM solve:

```bash
python -m vipe_roomtour.cli chunked-run /data/villa.mp4 /data/output/villa_loop \
  --chunk-frames 5000 --overlap-frames 500 \
  --pose-only --loop-closure
```

This keeps the stock frontend and backend, then runs loop-closure v3 before the
final DROID bundle adjustment:

1. Build a hybrid place descriptor from the existing DROID feature pyramid and
   a video-specific RootSIFT-VLAD vocabulary. Candidate pairs must be separated
   by at least 300 source frames and 12 keyframes.
2. Score five-keyframe sequences in both forward and reverse order. Reverse
   scoring is important when the camera climbs a stair, turns around, and later
   observes the same landing while walking in the opposite direction.
3. Verify multiple neighboring frame pairs, not only the best-looking image.
   Each pair must pass mutual RootSIFT matching, an essential-matrix test,
   forward and reverse depth-PnP, reprojection, cycle, and correction-size
   checks. At least two neighboring pairs must agree.
4. In parallel, propose temporally distant keyframes that are within 4.5 SLAM
   units of one another, then aggregate short point-cloud windows from the
   existing SLAM depth. Voxel cross-correlation plus reciprocal trimmed ICP
   handles opposite-facing revisits with no shared image pixels. The transform
   must repeat at two submap radii and pass overlap, mutual-correspondence,
   RMSE, colour, surface-normal, correction-size, and ambiguity checks. A
   transform from one anchor pair is still not enough: at least two nearby,
   independently registered keyframe pairs must recover the same world-space
   correction within 0.40 SLAM units and 4 degrees. This prevents one repeated
   white wall or floor patch from pulling the trajectory.
5. Optimize all keyframe poses with local odometry plus the verified long-range
   constraints. A second robust pass downweights mutually inconsistent loop
   edges. Image-verified pairs are inserted into the final dense DROID BA;
   opposite-view submap factors initialize the pose graph but are not forced
   into a dense-flow edge that may have no image overlap.
6. Measure the poses again after that final BA. The report therefore shows the
   correction that actually survives into exported per-frame poses, rather than
   only the intermediate pose-graph result.

The v3 retrieval and verification code uses OpenCV, NumPy/SciPy, and features
and SLAM depth already present in ViPE; it does not download an additional
neural-network checkpoint.

The mode is opt-in: omitting `--loop-closure` executes the previous solver.
Loop and non-loop checkpoints/depth are tagged separately. The algorithm
version is tagged as well, so a directory produced by loop-closure v1 is
automatically invalidated and recomputed by v3 instead of being silently reused.
For an A/B test, use different output directories and compare pose-only first:

```bash
python -m vipe_roomtour.cli chunked-run /data/villa.mp4 /data/output/villa_base \
  --chunk-frames 5000 --overlap-frames 500 --pose-only
python -m vipe_roomtour.cli chunked-run /data/villa.mp4 /data/output/villa_loop \
  --chunk-frames 5000 --overlap-frames 500 --pose-only --loop-closure
```

Each chunk writes
`vipe_artifacts/vipe/CHUNK_NAME_loop_closure.json`. It records sequence direction
and support, every per-pair rejection reason, accepted long-range edges, robust
edge weights, multiscale submap hypotheses, exact pose-graph sanity checks, and
pose-graph cost before/after. For submap edges, `cluster_support` is the number
of independent neighboring anchor pairs that recovered the same correction;
the two radii inside one registration do not count as independent support.
`final_dense_ba` reports the final camera correction and residual of every
accepted loop after DROID BA. No accepted edge is a valid outcome when the video
has no trustworthy revisit. When continuing a loop-enabled pose-only run,
repeat the flag:

```bash
python -m vipe_roomtour.cli chunked-run /data/villa.mp4 /data/output/villa_loop \
  --chunk-frames 5000 --overlap-frames 500 \
  --depth-only --depth-preset preview --loop-closure
```

This experiment targets pose drift. Rolling shutter, severe lens distortion,
or inconsistent dense DAv3 depth can still leave wall thickness even after a
correct loop. Repeated texture is why appearance similarity alone is never
allowed to modify a pose.

This accepts any input resolution supported by VIPE (720p, 1080p, and so on).
VIPE writes intrinsics in the original RGB pixel grid. If dense depth uses a
different internal grid, the mapper independently scales x/y intrinsics only
for backprojection; `calibration.npz` and CSV keep the original-resolution
values.

The depth presets affect only the post-SLAM dense-depth/map stage:

| Preset | Model | True DAv3 frame step | Resize | Saved depth grid |
| --- | --- | ---: | --- | --- |
| `preview` | DA3-LARGE | 5 | long side 504 | model grid |
| `balanced` | DA3-LARGE | 2 | short side 504 | model grid |
| `quality` | DA3-GIANT | 1 | short side 504 | original RGB grid |

`--depth-frame-step` now skips DAv3 forward passes rather than merely dropping
depth during mapping. Unless `--frame-step` is explicitly supplied, preview and
balanced fuse every generated sparse depth; quality retains the previous map
default of every fifth source frame. All-frame pose/intrinsics are unchanged.
Individual preset values can be overridden with `--dav3-model`,
`--dav3-model-path`, `--dav3-process-res`, `--dav3-resize-method`, and
`--depth-output-resolution`. The 3-frame overlap is intentionally not exposed
as a tuning option.

For offline nodes, `--dav3-model-path` accepts either the downloaded
`model.safetensors` itself or a directory containing that exact filename. The
matching architecture configuration for DA3-GIANT/LARGE/BASE/SMALL is vendored
with ViPE, so loading a local weight file does not contact Hugging Face.

Completed depth shards are kept under `depth/NAME.zip.parts/` after an
interrupted job. Rerunning the identical command resumes from the last complete
490-depth shard with one preceding window of context. After successful final
ZIP publication, that temporary shard directory is removed.

For a denser final map, override both inference and pixel sampling, for example
`--depth-frame-step 2 --frame-step 2 --pixel-stride 2`; memory and runtime rise
substantially. `--visualize-vipe` is intentionally skipped by this bounded-memory
path because retaining its full RGB-D input would reintroduce length-dependent
memory use.

## Run the uploaded example as separate floors

The reviewed complete file is approximately 251 seconds. Avoid the stair
transitions and solve each approximately planar portion independently:

```bash
vipe-roomtour run "/data/4ZNlunljHDc_review(1).mp4" /data/output/roomtour_example \
  --pipeline roomtour_dav3 \
  --segment main_floor:15:92 \
  --segment upper_floor:105:158 \
  --segment basement:191:250
```

The newer `(3)` upload is truncated near 201.5 seconds even though its MP4
header declares more frames; use the complete `(1)` file for the basement
segment. If only `(3)` is available, omit `basement` or end it before 201 s.

## Build maps from existing VIPE artifacts

If inference has already finished, do not run it again:

```bash
vipe-roomtour map /data/output/video_001/vipe_artifacts \
  /data/output/video_001/maps
```

Or resume the exact `run` layout:

```bash
vipe-roomtour run /data/video.mp4 /data/output/video_001 --skip-inference
```

`--skip-inference` now verifies that the saved depth metadata matches the
requested preset/overrides; it will not silently reuse a map made by a
different DAv3 model or frame step.

## Useful tuning

- `--max-depth 10`: suppress distant/noisy depth in small indoor rooms.
- `--voxel-size 0.02`: retain more detail; `0.04` is faster and lighter.
- `--min-voxel-observations 3`: stronger removal of one-frame artifacts.
- `--depth-edge-threshold 0`: disable depth-discontinuity filtering.
- `--floor-y VALUE`: override automatic floor detection in leveled y-down coordinates.
- `--topdown-min-height/--topdown-max-height`: control which vertical slice appears in the bird's-eye image.

The image canvas automatically reduces resolution if either side would exceed
`--max-raster-size`, preventing accidental huge allocations when a bad pose
creates an enormous trajectory.

## Reading poses correctly

ViPE poses are 4x4 OpenCV **camera-to-world** matrices:

```python
data = np.load("calibration.npz")
K = data["K"]          # [N, 3, 3], original RGB resolution
c2w = data["c2w"]      # [N, 4, 4]
w2c = data["w2c"]      # exact inverse
frame_ids = data["frame_indices"]
```

For a camera-space point `p_cam`, `p_world = R_c2w @ p_cam + t_c2w`.
Camera axes are +x right, +y down, +z forward.
