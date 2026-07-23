# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed room-tour batch worker for videos stored behind rclone."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .depth_options import DenseDepthOptions
from .map_builder import MapOptions
from .runner import run_roomtour

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"success", "missing_remote"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def youtube_video_id(url: str) -> str:
    """Extract a filesystem-safe YouTube video id from common URL forms."""

    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().split(":", 1)[0]
    video_id = ""
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif host in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            parts = parsed.path.strip("/").split("/")
            video_id = parts[1] if len(parts) > 1 else ""
    if not video_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in video_id
    ):
        raise ValueError(f"Cannot extract a valid YouTube video id from: {url}")
    return video_id


def load_url_entries(path: Path) -> list[tuple[str, str]]:
    """Load unique ``(video_id, url)`` entries while preserving text order."""

    entries = []
    seen = set()
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), 1):
        url = raw_line.strip()
        if not url or url.startswith("#"):
            continue
        try:
            video_id = youtube_video_id(url)
        except ValueError as exc:
            logger.warning("Ignoring URL line %d: %s", line_number, exc)
            continue
        if video_id in seen:
            continue
        seen.add(video_id)
        entries.append((video_id, url))
    if not entries:
        raise ValueError(f"No valid YouTube URLs found in {path}")
    return entries


@dataclass(frozen=True)
class ChunkSpec:
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    def folder_name(self, video_id: str) -> str:
        # end_frame is exclusive everywhere in the worker.
        return (
            f"{video_id}_{self.start_frame:06d}_"
            f"{self.end_frame:06d}.mp4"
        )


def plan_independent_chunks(
    total_frames: int,
    chunk_frames: int,
    overlap_frames: int,
    *,
    min_tail_frames: int | None = None,
) -> list[ChunkSpec]:
    """Plan independent chunks and merge a short final tail into its predecessor."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if not 0 <= overlap_frames < chunk_frames:
        raise ValueError("overlap_frames must satisfy 0 <= overlap < chunk")
    min_tail_frames = (
        chunk_frames // 2
        if min_tail_frames is None
        else int(min_tail_frames)
    )
    if not 0 <= min_tail_frames < chunk_frames:
        raise ValueError(
            "min_tail_frames must satisfy 0 <= min_tail_frames < chunk_frames"
        )
    if total_frames <= chunk_frames:
        return [ChunkSpec(0, total_frames)]

    stride = chunk_frames - overlap_frames
    chunks: list[ChunkSpec] = []
    start = 0
    while start + chunk_frames <= total_frames:
        chunks.append(ChunkSpec(start, start + chunk_frames))
        if start + chunk_frames == total_frames:
            return chunks
        start += stride

    remaining = total_frames - start
    if remaining > min_tail_frames:
        chunks.append(ChunkSpec(start, total_frames))
    else:
        previous = chunks[-1]
        chunks[-1] = ChunkSpec(previous.start_frame, total_frames)
    return chunks


def remote_join(root: str, name: str) -> str:
    return f"{root.rstrip('/')}/{name.lstrip('/')}"


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@dataclass
class WorkClaim:
    store: "StateStore"
    video_id: str
    token: str
    lock_dir: Path
    heartbeat_seconds: float
    _stop: threading.Event | None = None
    _thread: threading.Thread | None = None

    def __enter__(self) -> "WorkClaim":
        self._stop = threading.Event()

        def heartbeat() -> None:
            heartbeat_path = self.lock_dir / "heartbeat"
            while not self._stop.wait(self.heartbeat_seconds):
                try:
                    heartbeat_path.touch()
                except FileNotFoundError:
                    return
                except OSError:
                    logger.exception(
                        "Could not refresh lock heartbeat for %s",
                        self.video_id,
                    )

        self._thread = threading.Thread(
            target=heartbeat,
            name=f"heartbeat-{self.video_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.heartbeat_seconds))
        self.store.release(self)


class StateStore:
    """Shared JSONL audit log plus atomic per-video lease directories."""

    def __init__(
        self,
        state_file: Path,
        *,
        worker_id: str,
        stale_lock_seconds: float,
        heartbeat_seconds: float = 60.0,
    ):
        self.state_file = Path(state_file).resolve()
        self.worker_id = worker_id
        self.stale_lock_seconds = float(stale_lock_seconds)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_root = self.state_file.parent / (
            f".{self.state_file.name}.state"
        )
        self.lock_root = self.state_root / "locks"
        self.done_root = self.state_root / "done"
        self.failure_file = self.state_root / "failures.jsonl"
        self.lock_root.mkdir(parents=True, exist_ok=True)
        self.done_root.mkdir(parents=True, exist_ok=True)
        self._completed: set[str] = set()
        self._state_signature: tuple[int, int] | None = None
        self._refresh_completed(force=True)

    def _refresh_completed(self, *, force: bool = False) -> None:
        try:
            stat = self.state_file.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            signature = (0, 0)
        if not force and signature == self._state_signature:
            return
        completed = set()
        if self.state_file.exists():
            with self.state_file.open() as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if record.get("status") in TERMINAL_STATUSES:
                        completed.add(str(record.get("video_id", "")))
        self._completed = completed
        self._state_signature = signature

    def is_done(self, video_id: str) -> bool:
        if (self.done_root / f"{video_id}.json").exists():
            return True
        self._refresh_completed()
        return video_id in self._completed

    def _lock_is_stale(self, lock_dir: Path) -> bool:
        heartbeat = lock_dir / "heartbeat"
        try:
            modified = heartbeat.stat().st_mtime
        except FileNotFoundError:
            try:
                modified = lock_dir.stat().st_mtime
            except FileNotFoundError:
                return False
        return time.time() - modified > self.stale_lock_seconds

    def try_claim(self, video_id: str, url: str) -> WorkClaim | None:
        if self.is_done(video_id):
            return None
        lock_dir = self.lock_root / f"{video_id}.lock"
        token = uuid.uuid4().hex
        while True:
            try:
                lock_dir.mkdir()
                break
            except FileExistsError:
                if not self._lock_is_stale(lock_dir):
                    return None
                stale = self.lock_root / (
                    f"{video_id}.stale.{uuid.uuid4().hex}"
                )
                try:
                    os.rename(lock_dir, stale)
                except FileNotFoundError:
                    continue
                except OSError:
                    return None
                shutil.rmtree(stale, ignore_errors=True)

        owner = {
            "video_id": video_id,
            "url": url,
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "token": token,
            "claimed_at": _utc_now(),
        }
        try:
            _atomic_write_json(lock_dir / "owner.json", owner)
            (lock_dir / "heartbeat").touch()
        except Exception:
            shutil.rmtree(lock_dir, ignore_errors=True)
            raise
        claim = WorkClaim(
            store=self,
            video_id=video_id,
            token=token,
            lock_dir=lock_dir,
            heartbeat_seconds=self.heartbeat_seconds,
        )
        # Close the race with a worker that completed immediately before mkdir.
        if self.is_done(video_id):
            self.release(claim)
            return None
        return claim

    def release(self, claim: WorkClaim) -> None:
        try:
            owner = json.loads((claim.lock_dir / "owner.json").read_text())
        except (FileNotFoundError, OSError, ValueError):
            return
        if owner.get("token") != claim.token:
            logger.error(
                "Refusing to release a lock no longer owned by this worker: %s",
                claim.lock_dir,
            )
            return
        shutil.rmtree(claim.lock_dir, ignore_errors=True)

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def mark_terminal(
        self,
        *,
        video_id: str,
        url: str,
        status: str,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"Non-terminal status passed to mark_terminal: {status}")
        record = {
            "timestamp": _utc_now(),
            "video_id": video_id,
            "url": url,
            "status": status,
            "worker_id": self.worker_id,
            **details,
        }
        # The append is durable before the done marker appears. A new worker
        # can therefore recover completion from either representation.
        self._append_jsonl(self.state_file, record)
        _atomic_write_json(self.done_root / f"{video_id}.json", record)
        self._completed.add(video_id)
        try:
            stat = self.state_file.stat()
            self._state_signature = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            self._state_signature = None
        return record

    def record_failure(
        self,
        *,
        video_id: str,
        url: str,
        error: BaseException,
    ) -> None:
        self._append_jsonl(
            self.failure_file,
            {
                "timestamp": _utc_now(),
                "video_id": video_id,
                "url": url,
                "status": "failed_attempt",
                "worker_id": self.worker_id,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )


class RcloneClient:
    def __init__(
        self,
        *,
        binary: str = "rclone",
        transfers: int = 16,
        checkers: int = 32,
        extra_args: tuple[str, ...] = (),
    ):
        self.binary = binary
        self.transfers = int(transfers)
        self.checkers = int(checkers)
        self.extra_args = tuple(extra_args)

    def _transfer_args(self) -> list[str]:
        return [
            "--transfers",
            str(self.transfers),
            "--checkers",
            str(self.checkers),
            *self.extra_args,
        ]

    def list_top_level_files(self, remote_root: str) -> set[str]:
        command = [
            self.binary,
            "lsf",
            remote_root,
            "--files-only",
            "--max-depth",
            "1",
        ]
        logger.info("Indexing remote videos: %s", " ".join(command))
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        return {
            line.strip().removeprefix("./")
            for line in result.stdout.splitlines()
            if line.strip()
        }

    def copy_to_local(self, source: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.binary,
            "copyto",
            source,
            str(destination),
            *self._transfer_args(),
        ]
        logger.info("Downloading: %s", " ".join(command))
        subprocess.run(command, check=True)

    def copy_directory(self, source: Path, destination: str) -> None:
        command = [
            self.binary,
            "copy",
            str(source),
            destination,
            *self._transfer_args(),
        ]
        logger.info("Uploading: %s", " ".join(command))
        subprocess.run(command, check=True)

    def copy_file(self, source: Path, destination: str) -> None:
        command = [
            self.binary,
            "copyto",
            str(source),
            destination,
            *self._transfer_args(),
        ]
        logger.info("Uploading manifest: %s", " ".join(command))
        subprocess.run(command, check=True)


def _ffprobe_json(video: Path, *, count_frames: bool) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        *(["-count_frames"] if count_frames else []),
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,nb_read_frames,avg_frame_rate,width,height",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video}")
    return streams[0]


def probe_video(video: Path) -> dict[str, Any]:
    stream = _ffprobe_json(video, count_frames=False)
    frame_value = stream.get("nb_frames")
    if frame_value in {None, "", "N/A", "0"}:
        stream = _ffprobe_json(video, count_frames=True)
        frame_value = stream.get("nb_read_frames")
    try:
        total_frames = int(frame_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not determine frame count for {video}") from exc
    try:
        fps = float(Fraction(str(stream["avg_frame_rate"])))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"Could not determine frame rate for {video}") from exc
    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def _count_rgb_frames(rgb_dir: Path) -> int:
    if not rgb_dir.is_dir():
        return 0
    return sum(
        entry.is_file() and entry.name.endswith(".jpg")
        for entry in os.scandir(rgb_dir)
    )


def _prepare_chunk(
    source_video: Path,
    result_root: Path,
    video_id: str,
    spec: ChunkSpec,
    video_info: dict[str, Any],
    *,
    ffmpeg_threads: int,
) -> Path:
    chunk_dir = result_root / spec.folder_name(video_id)
    chunk_video = chunk_dir / "video.mp4"
    rgb_dir = chunk_dir / "RGB"
    metadata_path = chunk_dir / "chunk_metadata.json"
    if (
        chunk_video.is_file()
        and _count_rgb_frames(rgb_dir) == spec.frame_count
        and metadata_path.is_file()
    ):
        try:
            saved = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            saved = None
        if (
            saved is not None
            and saved.get("source_start_frame") == spec.start_frame
            and saved.get("source_end_frame_exclusive") == spec.end_frame
        ):
            logger.info("Prepared chunk already exists: %s", chunk_dir.name)
            return chunk_dir

    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_video.unlink(missing_ok=True)
    shutil.rmtree(rgb_dir, ignore_errors=True)
    rgb_dir.mkdir(parents=True)
    filter_graph = (
        f"[0:v:0]trim=start_frame={spec.start_frame}:"
        f"end_frame={spec.end_frame},setpts=PTS-STARTPTS,"
        "split=2[chunkvideo][chunkframes]"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_video),
        "-filter_complex",
        filter_graph,
        "-map",
        "[chunkvideo]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        str(ffmpeg_threads),
        "-movflags",
        "+faststart",
        str(chunk_video),
        "-map",
        "[chunkframes]",
        "-vsync",
        "0",
        "-q:v",
        "2",
        "-start_number",
        "0",
        str(rgb_dir / "%06d.jpg"),
    ]
    logger.info(
        "Preparing %s frames [%d, %d)",
        chunk_dir.name,
        spec.start_frame,
        spec.end_frame,
    )
    subprocess.run(command, check=True)
    decoded_frames = _count_rgb_frames(rgb_dir)
    if decoded_frames != spec.frame_count:
        raise RuntimeError(
            f"{chunk_dir.name}: expected {spec.frame_count} RGB frames, "
            f"decoded {decoded_frames}"
        )
    chunk_video_info = probe_video(chunk_video)
    if chunk_video_info["total_frames"] != spec.frame_count:
        raise RuntimeError(
            f"{chunk_dir.name}: encoded video has "
            f"{chunk_video_info['total_frames']} frames, expected "
            f"{spec.frame_count}"
        )
    _atomic_write_json(
        metadata_path,
        {
            "video_id": video_id,
            "source_start_frame": spec.start_frame,
            "source_end_frame_exclusive": spec.end_frame,
            "frame_count": spec.frame_count,
            "source_total_frames": video_info["total_frames"],
            "fps": video_info["fps"],
            "width": video_info["width"],
            "height": video_info["height"],
            "prepared_at": _utc_now(),
        },
    )
    return chunk_dir


def _slam_config(
    pipeline: str,
    depth_options: DenseDepthOptions,
    map_options: MapOptions,
) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "depth": depth_options.to_dict(),
        "map": asdict(map_options),
    }


def _slam_outputs_exist(chunk_dir: Path) -> bool:
    artifact_root = chunk_dir / "vipe" / "vipe_artifacts"
    map_root = chunk_dir / "vipe" / "maps" / "video"
    required = (
        artifact_root / "pose" / "video.npz",
        artifact_root / "intrinsics" / "video.npz",
        artifact_root / "depth" / "video.zip",
        artifact_root / "vipe" / "video_info.pkl",
        artifact_root / "vipe" / "video_slam_map.pt",
        map_root / "global_rgb_map_world.ply",
        map_root / "quality_report.md",
    )
    return all(path.exists() for path in required)


def _run_chunk_slam(
    chunk_dir: Path,
    *,
    pipeline: str,
    depth_options: DenseDepthOptions,
    map_options: MapOptions,
) -> None:
    config = _slam_config(pipeline, depth_options, map_options)
    marker = chunk_dir / ".slam_complete.json"
    if marker.exists() and _slam_outputs_exist(chunk_dir):
        try:
            saved = json.loads(marker.read_text())
        except (OSError, ValueError):
            saved = None
        if saved is not None and saved.get("config") == config:
            logger.info("SLAM result already complete: %s", chunk_dir.name)
            return

    vipe_root = chunk_dir / "vipe"
    if vipe_root.exists():
        # A missing/mismatched marker means this output may have had its
        # artifact RGB removed already. Rebuild this exact chunk cleanly.
        shutil.rmtree(vipe_root)
    run_roomtour(
        chunk_dir / "video.mp4",
        vipe_root,
        segments=[],
        pipeline=pipeline,
        map_options=map_options,
        depth_options=depth_options,
    )
    shutil.rmtree(
        vipe_root / "vipe_artifacts" / "rgb",
        ignore_errors=True,
    )
    if not _slam_outputs_exist(chunk_dir):
        raise RuntimeError(f"Incomplete VIPE/map output for {chunk_dir.name}")
    _atomic_write_json(
        marker,
        {
            "completed_at": _utc_now(),
            "config": config,
        },
    )


def _prepare_and_process_chunks(
    source_video: Path,
    processing_root: Path,
    video_id: str,
    specs: list[ChunkSpec],
    video_info: dict[str, Any],
    *,
    ffmpeg_threads: int,
    pipeline: str,
    depth_options: DenseDepthOptions,
    map_options: MapOptions,
) -> list[Path]:
    """Pipeline one splitter thread into one sequential SLAM consumer."""

    result_root = processing_root / "result"
    result_root.mkdir(parents=True, exist_ok=True)
    prepared: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
    stop = threading.Event()
    completed: list[Path] = []

    def put_message(kind: str, payload: Any) -> bool:
        while not stop.is_set():
            try:
                prepared.put((kind, payload), timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def produce_chunks() -> None:
        try:
            for spec in specs:
                if stop.is_set():
                    return
                chunk_dir = _prepare_chunk(
                    source_video,
                    result_root,
                    video_id,
                    spec,
                    video_info,
                    ffmpeg_threads=ffmpeg_threads,
                )
                if not put_message("chunk", chunk_dir):
                    return
        except BaseException as exc:
            put_message("error", exc)
        finally:
            put_message("done", None)

    producer = threading.Thread(
        target=produce_chunks,
        name=f"split-{video_id}",
        daemon=True,
    )
    producer.start()
    try:
        while True:
            kind, payload = prepared.get()
            if kind == "done":
                break
            if kind == "error":
                raise payload
            chunk_dir = payload
            # At most one following chunk is prepared while this GPU-heavy
            # stage runs, keeping local disk growth bounded.
            _run_chunk_slam(
                chunk_dir,
                pipeline=pipeline,
                depth_options=depth_options,
                map_options=map_options,
            )
            completed.append(chunk_dir)
    finally:
        stop.set()
        producer.join()

    completed.sort()
    return completed


def _download_video(
    rclone: RcloneClient,
    source_remote: str,
    processing_root: Path,
    video_id: str,
) -> Path:
    temporary_root = processing_root / "temp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    destination = temporary_root / f"{video_id}.mp4"
    if destination.is_file() and destination.stat().st_size > 0:
        logger.info("Using existing local download: %s", destination)
        return destination
    partial = temporary_root / (
        f".{video_id}.{os.getpid()}.{uuid.uuid4().hex}.part"
    )
    try:
        rclone.copy_to_local(
            remote_join(source_remote, f"{video_id}.mp4"),
            partial,
        )
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty video for {video_id}")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def _upload_video_results(
    rclone: RcloneClient,
    *,
    chunk_dirs: list[Path],
    remote_output: str,
    processing_root: Path,
    video_id: str,
    url: str,
    video_info: dict[str, Any],
) -> Path:
    for chunk_dir in chunk_dirs:
        rclone.copy_directory(
            chunk_dir,
            remote_join(remote_output, chunk_dir.name),
        )
    manifest = processing_root / "temp" / f"{video_id}_manifest.json"
    _atomic_write_json(
        manifest,
        {
            "video_id": video_id,
            "url": url,
            "uploaded_at": _utc_now(),
            "video": video_info,
            "chunks": [chunk_dir.name for chunk_dir in chunk_dirs],
        },
    )
    rclone.copy_file(
        manifest,
        remote_join(
            remote_output,
            f"_video_manifests/{video_id}.json",
        ),
    )
    return manifest


def _cleanup_local_video(processing_root: Path, video_id: str) -> None:
    """Delete only local artifacts belonging to one successfully uploaded video."""

    processing_root = Path(processing_root)
    result_root = processing_root / "result"
    temporary_root = processing_root / "temp"
    chunk_pattern = re.compile(
        rf"{re.escape(video_id)}_\d{{6,}}_\d{{6,}}\.mp4"
    )

    if result_root.is_dir():
        for candidate in result_root.iterdir():
            if not chunk_pattern.fullmatch(candidate.name):
                continue
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate)
            else:
                candidate.unlink(missing_ok=True)

    temporary_paths = [
        temporary_root / f"{video_id}.mp4",
        temporary_root / f"{video_id}_manifest.json",
    ]
    if temporary_root.is_dir():
        temporary_paths.extend(
            temporary_root.glob(f".{video_id}.*.part")
        )
        temporary_paths.extend(
            temporary_root.glob(
                f".{video_id}_manifest.json.*.tmp"
            )
        )
    for candidate in temporary_paths:
        candidate.unlink(missing_ok=True)

    # These roots are shared by all workers. Remove them only when they are
    # empty; concurrent workers' files make rmdir fail harmlessly.
    for shared_root in (result_root, temporary_root):
        try:
            shared_root.rmdir()
        except (FileNotFoundError, OSError):
            pass


def run_batch_worker(
    urls_file: Path,
    processing_root: Path,
    remote_output: str,
    *,
    state_file: Path,
    source_remote: str,
    chunk_frames: int,
    overlap_frames: int,
    min_tail_frames: int | None,
    pipeline: str,
    depth_options: DenseDepthOptions,
    map_options: MapOptions,
    ffmpeg_threads: int = 2,
    max_videos: int = 0,
    stale_lock_hours: float = 6.0,
    worker_id: str | None = None,
    rclone_binary: str = "rclone",
    rclone_transfers: int = 16,
    rclone_checkers: int = 32,
    rclone_extra_args: tuple[str, ...] = (),
    fail_fast: bool = False,
) -> dict[str, int]:
    """Claim and process every currently available URL not owned by another worker."""

    processing_root = Path(processing_root)
    state_file = Path(state_file)
    if not processing_root.is_absolute():
        raise ValueError("PROCESSING_ROOT must be an absolute path")
    if not state_file.is_absolute():
        raise ValueError("--state-file must be an absolute shared path")
    if ffmpeg_threads < 1:
        raise ValueError("ffmpeg_threads must be >= 1")
    entries = load_url_entries(urls_file)
    worker_id = worker_id or (
        f"{socket.gethostname()}:{os.getpid()}:"
        f"{os.environ.get('SLURM_JOB_ID', 'nojob')}:{uuid.uuid4().hex[:8]}"
    )
    state = StateStore(
        state_file,
        worker_id=worker_id,
        stale_lock_seconds=stale_lock_hours * 3600.0,
    )
    rclone = RcloneClient(
        binary=rclone_binary,
        transfers=rclone_transfers,
        checkers=rclone_checkers,
        extra_args=rclone_extra_args,
    )
    remote_files = {
        name
        for name in rclone.list_top_level_files(source_remote)
        if name.lower().endswith(".mp4")
    }
    if not remote_files:
        raise RuntimeError(
            "The source remote contains no top-level MP4 files; refusing to "
            "mark the complete URL list as missing. Check --source-remote and "
            "rclone authentication."
        )
    matched_sources = sum(
        f"{video_id}.mp4" in remote_files
        for video_id, _ in entries
    )
    if matched_sources == 0:
        raise RuntimeError(
            "None of the URL-list video ids exists in --source-remote; "
            "refusing to mark every item missing. Check the URL file and "
            "remote path."
        )
    logger.info("Indexed %d source MP4 files", len(remote_files))
    logger.info(
        "Remote preflight matched %d/%d URL entries",
        matched_sources,
        len(entries),
    )
    summary = {
        "claimed": 0,
        "success": 0,
        "missing_remote": 0,
        "failed": 0,
        "skipped_done_or_locked": 0,
    }
    attempted: set[str] = set()

    for video_id, url in entries:
        if max_videos > 0 and summary["claimed"] >= max_videos:
            break
        if video_id in attempted or state.is_done(video_id):
            summary["skipped_done_or_locked"] += 1
            continue
        claim = state.try_claim(video_id, url)
        if claim is None:
            summary["skipped_done_or_locked"] += 1
            continue
        attempted.add(video_id)
        summary["claimed"] += 1
        with claim:
            if f"{video_id}.mp4" not in remote_files:
                try:
                    _cleanup_local_video(processing_root, video_id)
                    state.mark_terminal(
                        video_id=video_id,
                        url=url,
                        status="missing_remote",
                        details={
                            "source_remote": source_remote,
                            "remote_output": remote_output,
                        },
                    )
                    summary["missing_remote"] += 1
                    logger.warning(
                        "Remote video does not exist; marked missing: %s",
                        video_id,
                    )
                except Exception as exc:
                    summary["failed"] += 1
                    state.record_failure(
                        video_id=video_id,
                        url=url,
                        error=exc,
                    )
                    logger.exception(
                        "Could not clean/record missing video %s; it remains retryable",
                        video_id,
                    )
                    if fail_fast:
                        raise
                continue
            try:
                source_video = _download_video(
                    rclone,
                    source_remote,
                    processing_root,
                    video_id,
                )
                video_info = probe_video(source_video)
                specs = plan_independent_chunks(
                    video_info["total_frames"],
                    chunk_frames,
                    overlap_frames,
                    min_tail_frames=min_tail_frames,
                )
                logger.info(
                    "Processing %s: %d frames -> %d independent chunks",
                    video_id,
                    video_info["total_frames"],
                    len(specs),
                )
                chunk_dirs = _prepare_and_process_chunks(
                    source_video,
                    processing_root,
                    video_id,
                    specs,
                    video_info,
                    ffmpeg_threads=ffmpeg_threads,
                    pipeline=pipeline,
                    depth_options=depth_options,
                    map_options=map_options,
                )
                _upload_video_results(
                    rclone,
                    chunk_dirs=chunk_dirs,
                    remote_output=remote_output,
                    processing_root=processing_root,
                    video_id=video_id,
                    url=url,
                    video_info=video_info,
                )
                chunk_names = [path.name for path in chunk_dirs]
                _cleanup_local_video(processing_root, video_id)
                state.mark_terminal(
                    video_id=video_id,
                    url=url,
                    status="success",
                    details={
                        "source_remote": source_remote,
                        "remote_output": remote_output,
                        "total_frames": video_info["total_frames"],
                        "chunk_count": len(chunk_dirs),
                        "chunks": chunk_names,
                    },
                )
                summary["success"] += 1
            except Exception as exc:
                summary["failed"] += 1
                state.record_failure(
                    video_id=video_id,
                    url=url,
                    error=exc,
                )
                logger.exception("Failed video %s; it remains retryable", video_id)
                if fail_fast:
                    raise

    logger.info("Batch worker %s summary: %s", worker_id, summary)
    return summary
