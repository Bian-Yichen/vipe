# Distributed room-tour batch processing

`batch-run` scans a shared YouTube URL text file, atomically claims one video
at a time, downloads `VIDEO_ID.mp4` through rclone, creates independent chunks,
runs the existing single-video VIPE pipeline for every chunk, immediately
uploads/checkpoints/deletes that chunk, and marks the source video complete
after every planned chunk is durable remotely.

The implementation is based on `agent/roomtour-chunked-sim3`, but it does not
stitch adjacent chunks. Every chunk is a completely independent SLAM solve.

## One worker

Run from the repository root. `PROCESSING_ROOT` must be an absolute local path.
`--state-file` must be an absolute path on a filesystem shared by every worker.

```bash
srun -p vcg --gres=gpu:1 --quotatype=auto \
python -m vipe_roomtour.cli batch-run \
  /mnt/petrelfs/bianyichen/3DVision/roomtour_urls.txt \
  /mnt/hwfile/bianyichen/3DVision/vipe_batch/worker_0 \
  h:bianyichen/AnyReconProDataset_slam_chunks/ \
  --state-file /mnt/petrelfs/bianyichen/3DVision/vipe_batch_state.jsonl \
  --source-remote h:bianyichen/AnyReconProDataset_processed/ \
  --chunk-frames 5000 \
  --overlap-frames 1000 \
  --depth-preset preview \
  --ffmpeg-threads 2 \
  --rclone-transfers 200 \
  --rclone-checkers 200 \
  --rclone-extra-arg=--s3-no-check-bucket
```

By default, model inference is fully offline and rclone alone receives an
environment with `http_proxy`, `https_proxy`, and `all_proxy` (including their
uppercase forms) removed. Cached GeoCalib, DROID, DAv3 metric, and dense DAv3
weights are used without Hugging Face HEAD requests. You therefore do not need
to unset proxy variables for the complete shell command.

If a model is not cached yet, run a separate online preparation once or pass
`--online-models`. Use `--rclone-use-proxy` only for an rclone remote that
actually requires the proxy.

You may use the same absolute `PROCESSING_ROOT` for every independently
launched command. Downloads, manifests, and chunk directories are namespaced
by `VIDEO_ID`, while the shared lease prevents two workers from processing the
same video. Every command still runs one batch worker and processes its claimed
videos sequentially.

Each claimed video also gets an append-only local log at
`PROCESSING_ROOT/log/VIDEO_ID.log`. It contains the parent batch events and the
child VIPE/SLAM logger output. The video lease prevents concurrent writers for
the same `VIDEO_ID`; different workers write different files. A retry appends a
new attempt to the existing log. Successful-result cleanup deliberately keeps
the `log/` directory.

## Chunk rule

Ranges are half-open: `[start_frame, end_frame)`.

- 9000 frames with `chunk=5000, overlap=1000` becomes `[0,5000)` and
  `[4000,9000)`.
- A final remainder strictly larger than half a chunk becomes its own chunk.
- A final remainder no larger than half a chunk is merged into the preceding
  chunk, so that preceding chunk may be longer than 5000 frames.
- A source video shorter than one chunk remains one chunk.

Use `--min-tail-frames` to override the default half-chunk threshold.

## Output layout

Each uploaded directory is named
`VIDEO_ID_START_END.mp4`, where `END` is exclusive:

```text
VIDEO_ID_000000_005000.mp4/
├── video.mp4
├── RGB/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── chunk_metadata.json
├── .slam_complete.json
└── vipe/
    ├── vipe_artifacts/
    │   ├── depth/
    │   ├── intrinsics/
    │   ├── pose/
    │   └── vipe/
    ├── maps/
    └── manifest.json
```

`vipe_artifacts/rgb` is removed after map construction because `video.mp4` and
the decoded `RGB` directory already exist at the chunk root.

A video-level completion manifest is uploaded last to
`REMOTE_OUTPUT/_video_manifests/VIDEO_ID.json`.

## Concurrency and recovery

For every video, the worker holds a non-blocking POSIX `flock` next to the
shared state file. Other workers that cannot acquire that lock skip the video
and continue scanning. The kernel releases the lease immediately when the
worker exits, is killed, or is preempted; the small lock file remains as owner
metadata but does not remain locked. The shared filesystem containing
`--state-file` must support `flock`.

`--stale-lock-hours` is now used only to migrate directory/heartbeat locks left
by an older worker version. Its default is 0.1 hours (six minutes). A legacy
lock owned by a dead PID on the current host is reclaimed immediately.

Inside one worker, the splitter thread prepares `video.mp4` and `RGB/`, and the
main thread runs SLAM and upload. A completion gate prevents the splitter from
preparing the next chunk until the current chunk has been uploaded,
checkpointed, and deleted. Therefore each `VIDEO_ID` creates at most one local
chunk directory at a time (in addition to its downloaded source MP4).

The per-chunk commit sequence is:

1. prepare one chunk and run its independent VIPE/SLAM/depth/map pipeline;
2. upload that chunk to its final remote directory;
3. atomically record the successful chunk in
   `.STATE_FILE.state/chunks/VIDEO_ID.json`;
4. delete that local chunk directory;
5. only then allow the splitter to prepare the next chunk.

After all chunk checkpoints exist, the worker uploads the video manifest,
deletes the local source/temporary data, appends the terminal `success` record,
and releases the lease.

This order defines the recovery behavior:

- killed during SLAM: the one partial local chunk is resumed/rebuilt;
- killed during upload: no checkpoint exists, so the same remote path is
  uploaded again idempotently;
- killed after checkpoint but before deletion: the next owner deletes the
  leftover and skips that chunk;
- killed after several chunks: the next owner reads the per-video checkpoint
  and starts at the first chunk not recorded as successful.

The checkpoint also stores the source, destination, chunk plan, and inference
configuration. A retry with incompatible arguments fails clearly instead of
silently combining different outputs; use the original command or a new
`--state-file`.

Cleanup never deletes another video's paths, so workers sharing
`PROCESSING_ROOT` remain isolated. The common `temp/` and `result/` directories
are removed only when empty. This order also prevents a second process from
claiming the video between unlock and completion recording. Missing remote MP4s
are cleaned, recorded as `missing_remote`, and will not be scanned again.

Failed processing or upload attempts are written to
`.STATE_FILE.state/failures.jsonl`, are not marked complete, and remain
retryable in a later invocation. The current partial chunk is retained for
resume. Use `--fail-fast` when one failure should terminate the worker.

The launch command is unchanged. Workers must share the same `--state-file`;
using the same `PROCESSING_ROOT` also lets a replacement worker reuse or clean
the interrupted worker's local partial chunk.
