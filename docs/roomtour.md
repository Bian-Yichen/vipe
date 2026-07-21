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

The room-tour entry point writes these artifacts with bounded memory.  It does
not cache every full-resolution RGB-D frame: RGB is encoded during DAv3's
existing keyframe pre-pass, and each final full-resolution DAv3 depth is written
to the EXR zip as soon as it is produced.  Pose, intrinsics, and the SLAM map
are checkpointed before dense depth, allowing a failed post-process to resume
without repeating SLAM.  This changes storage only; SLAM, DAv3 windows,
overlap blending, depth resolution, and artifact numeric precision are unchanged.

## Install

The commands below assume Linux, an NVIDIA CUDA GPU, Git, and FFmpeg. From this
branch of the repository:

```bash
git clone https://github.com/Bian-Yichen/vipe.git
cd vipe
git checkout agent/roomtour-topdown-map

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

## Run one complete video

```bash
vipe-roomtour run /data/video.mp4 /data/output/video_001 \
  --pipeline roomtour_dav3
```

## Long tours: overlapping chunk solve and Sim(3) stitch

For long trajectories that are less stable as one global SLAM solve, run each
overlapping block independently and align adjacent blocks through their shared
source frames:

```bash
vipe-roomtour chunked-run /data/villa_25000_frames.mp4 /data/output/villa_chunked \
  --pipeline roomtour_dav3 \
  --chunk-frames 5000 \
  --overlap-frames 500
```

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
cannot repair a bad pose estimate inside a 5000-frame chunk. Pairwise alignment
also accumulates slowly over many chunks because there is no distant loop
closure. It is most suitable when the camera continuously explores new areas,
as in the intended villa tours.

This accepts any input resolution supported by VIPE (720p, 1080p, and so on).
VIPE writes intrinsics in the original RGB pixel grid. If dense depth uses a
different internal grid, the mapper independently scales x/y intrinsics only
for backprojection; `calibration.npz` and CSV keep the original-resolution
values.

For long videos, start with the defaults (`--frame-step 5 --pixel-stride 4`).
For a denser map, try `--frame-step 2 --pixel-stride 2`; memory and runtime rise
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
