# Distributed room-tour batch processing

`batch-run` scans a shared YouTube URL text file, atomically claims one video
at a time, downloads `VIDEO_ID.mp4` through rclone, creates independent chunks,
runs the existing single-video VIPE pipeline for every chunk, uploads all
results, and marks the source video complete.

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
  --prepare-workers 2 \
  --ffmpeg-threads 2 \
  --rclone-transfers 200 \
  --rclone-checkers 200 \
  --rclone-extra-arg=--s3-no-check-bucket
```

Use a different local `PROCESSING_ROOT` for each concurrently launched process,
but pass the same URL file, state file, source remote, and output remote.

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

For every video, the worker atomically creates a lease directory next to the
shared state file. Other workers that cannot acquire that lease skip the video
and continue scanning. A heartbeat keeps long VIPE jobs alive; abandoned leases
can be reclaimed after `--stale-lock-hours`.

The completion sequence is:

1. upload every complete chunk;
2. upload the video manifest;
3. append a durable `success` record to the shared JSONL and create a done
   marker while still holding the lease;
4. release the lease;
5. clean the local download and chunk directories.

This order prevents a second process from claiming the video between unlock and
completion recording. Missing remote MP4s are recorded as `missing_remote` and
will not be scanned again.

Failed processing or upload attempts are written to
`.STATE_FILE.state/failures.jsonl`, are not marked complete, and remain
retryable in a later invocation. Partial local chunk outputs are retained for
resume. Use `--fail-fast` when one failure should terminate the worker.

By default, successfully uploaded local data is deleted. Add
`--keep-local-results` for debugging.
