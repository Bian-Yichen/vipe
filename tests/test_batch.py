from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vipe_roomtour import batch as batch_module
from vipe_roomtour.batch import (
    RcloneClient,
    StateStore,
    _cleanup_local_video,
    _prepare_and_process_chunks,
    _video_log_context,
    plan_independent_chunks,
    remote_join,
    youtube_video_id,
)


def _ranges(total: int, chunk: int = 5000, overlap: int = 1000):
    return [
        (item.start_frame, item.end_frame)
        for item in plan_independent_chunks(total, chunk, overlap)
    ]


def test_youtube_video_id() -> None:
    assert youtube_video_id(
        "https://www.youtube.com/watch?v=OKMEDoIf4uY"
    ) == "OKMEDoIf4uY"
    assert youtube_video_id(
        "https://youtu.be/z-1ERaurAjY?t=4"
    ) == "z-1ERaurAjY"
    with pytest.raises(ValueError):
        youtube_video_id("https://example.com/video")


def test_chunk_plan_exact_and_short_tail_merge() -> None:
    assert _ranges(9000) == [(0, 5000), (4000, 9000)]
    assert _ranges(9500) == [(0, 5000), (4000, 9500)]
    assert _ranges(11500) == [
        (0, 5000),
        (4000, 9000),
        (8000, 11500),
    ]
    assert _ranges(2000) == [(0, 2000)]


def test_chunk_plan_validates_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        plan_independent_chunks(10000, 5000, 5000)


def test_remote_join() -> None:
    assert remote_join("h:bucket/root/", "/abc.mp4") == (
        "h:bucket/root/abc.mp4"
    )


def test_shared_claim_and_terminal_record(tmp_path: Path) -> None:
    state_file = tmp_path / "processed.jsonl"
    first = StateStore(
        state_file,
        worker_id="worker-a",
        stale_lock_seconds=3600,
        heartbeat_seconds=0.05,
    )
    second = StateStore(
        state_file,
        worker_id="worker-b",
        stale_lock_seconds=3600,
        heartbeat_seconds=0.05,
    )
    claim = first.try_claim("video123", "https://youtu.be/video123")
    assert claim is not None
    assert second.try_claim(
        "video123",
        "https://youtu.be/video123",
    ) is None

    with claim:
        first.mark_terminal(
            video_id="video123",
            url="https://youtu.be/video123",
            status="success",
            details={"chunk_count": 2},
        )

    assert second.is_done("video123")
    records = [
        json.loads(line)
        for line in state_file.read_text().splitlines()
    ]
    assert records[-1]["status"] == "success"


def test_stale_lock_can_be_reclaimed_without_old_owner_release(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "processed.jsonl"
    first = StateStore(
        state_file,
        worker_id="worker-a",
        stale_lock_seconds=0.01,
    )
    second = StateStore(
        state_file,
        worker_id="worker-b",
        stale_lock_seconds=0.01,
    )
    old_claim = first.try_claim("video123", "https://youtu.be/video123")
    assert old_claim is not None
    old_time = time.time() - 10
    os.utime(old_claim.lock_dir / "heartbeat", (old_time, old_time))

    new_claim = second.try_claim("video123", "https://youtu.be/video123")
    assert new_claim is not None
    # The original owner must not delete the replacement lock.
    first.release(old_claim)
    assert new_claim.lock_dir.exists()
    second.release(new_claim)


def test_split_thread_and_slam_consumer_are_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_threads = []
    slam_threads = []

    def fake_prepare(
        source_video,
        result_root,
        video_id,
        spec,
        video_info,
        *,
        ffmpeg_threads,
    ):
        prepare_threads.append(threading.current_thread().name)
        chunk_dir = result_root / spec.folder_name(video_id)
        chunk_dir.mkdir(parents=True)
        return chunk_dir

    def fake_slam(chunk_dir, **kwargs):
        slam_threads.append(threading.current_thread().name)

    monkeypatch.setattr(batch_module, "_prepare_chunk", fake_prepare)
    monkeypatch.setattr(batch_module, "_run_chunk_slam", fake_slam)
    specs = [
        batch_module.ChunkSpec(0, 5000),
        batch_module.ChunkSpec(4000, 9000),
    ]
    result = _prepare_and_process_chunks(
        tmp_path / "source.mp4",
        tmp_path,
        "video123",
        specs,
        {"total_frames": 9000},
        ffmpeg_threads=2,
        pipeline="roomtour_dav3",
        depth_options=object(),
        map_options=object(),
        offline_models=True,
    )

    assert [path.name for path in result] == [
        "video123_000000_005000.mp4",
        "video123_004000_009000.mp4",
    ]
    assert prepare_threads
    assert all(name == "split-video123" for name in prepare_threads)
    assert slam_threads == [
        threading.current_thread().name,
        threading.current_thread().name,
    ]


def test_cleanup_removes_only_requested_video(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    temporary_root = tmp_path / "temp"
    result_root.mkdir()
    temporary_root.mkdir()

    target_chunk = result_root / "video123_000000_005000.mp4"
    other_chunk = result_root / "video123_extra_000000_005000.mp4"
    target_chunk.mkdir()
    other_chunk.mkdir()
    (target_chunk / "payload").write_text("target")
    (other_chunk / "payload").write_text("other")

    (temporary_root / "video123.mp4").write_text("target")
    (temporary_root / "video123_manifest.json").write_text("target")
    (temporary_root / ".video123.42.deadbeef.part").write_text("target")
    (temporary_root / "video123_extra.mp4").write_text("other")

    _cleanup_local_video(tmp_path, "video123")

    assert not target_chunk.exists()
    assert not (temporary_root / "video123.mp4").exists()
    assert not (temporary_root / "video123_manifest.json").exists()
    assert not (temporary_root / ".video123.42.deadbeef.part").exists()
    assert other_chunk.exists()
    assert (temporary_root / "video123_extra.mp4").exists()


def test_success_uploads_then_cleans_then_marks_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://www.youtube.com/watch?v=video123\n")
    processing_root = tmp_path / "processing"
    state_file = tmp_path / "state.jsonl"
    source_video = processing_root / "temp" / "video123.mp4"
    chunk_dir = (
        processing_root
        / "result"
        / "video123_000000_005000.mp4"
    )

    class FakeRclone:
        def __init__(self, **kwargs):
            pass

        def list_top_level_files(self, remote_root):
            return {"video123.mp4"}

    def fake_download(*args, **kwargs):
        source_video.parent.mkdir(parents=True)
        source_video.write_text("video")
        return source_video

    def fake_chunks(*args, **kwargs):
        chunk_dir.mkdir(parents=True)
        return [chunk_dir]

    def fake_upload(*args, **kwargs):
        events.append("upload")
        return processing_root / "temp" / "video123_manifest.json"

    def fake_cleanup(*args, **kwargs):
        events.append("cleanup")

    original_mark_terminal = StateStore.mark_terminal

    def tracked_mark_terminal(store, **kwargs):
        events.append("mark")
        return original_mark_terminal(store, **kwargs)

    monkeypatch.setattr(batch_module, "RcloneClient", FakeRclone)
    monkeypatch.setattr(batch_module, "_download_video", fake_download)
    monkeypatch.setattr(
        batch_module,
        "probe_video",
        lambda path: {
            "total_frames": 5000,
            "fps": 30.0,
            "width": 1280,
            "height": 720,
        },
    )
    monkeypatch.setattr(
        batch_module,
        "_prepare_and_process_chunks",
        fake_chunks,
    )
    monkeypatch.setattr(
        batch_module,
        "_upload_video_results",
        fake_upload,
    )
    monkeypatch.setattr(
        batch_module,
        "_cleanup_local_video",
        fake_cleanup,
    )
    monkeypatch.setattr(
        StateStore,
        "mark_terminal",
        tracked_mark_terminal,
    )

    summary = batch_module.run_batch_worker(
        urls_file,
        processing_root.resolve(),
        "h:output",
        state_file=state_file.resolve(),
        source_remote="h:input",
        chunk_frames=5000,
        overlap_frames=1000,
        min_tail_frames=None,
        pipeline="roomtour_dav3",
        depth_options=object(),
        map_options=object(),
        stale_lock_hours=1.0,
    )

    assert events == ["upload", "cleanup", "mark"]
    assert summary["success"] == 1
    assert (processing_root / "log" / "video123.log").is_file()


def test_rclone_subprocess_clears_only_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in batch_module._PROXY_ENV_NAMES:
        monkeypatch.setenv(name, f"value-{name}")
    monkeypatch.setenv("KEEP_ME", "yes")
    captured = {}

    def fake_run(command, **kwargs):
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(stdout="video123.mp4\n")

    monkeypatch.setattr(batch_module.subprocess, "run", fake_run)
    files = RcloneClient().list_top_level_files("h:input")

    assert files == {"video123.mp4"}
    assert captured["environment"]["KEEP_ME"] == "yes"
    assert all(
        name not in captured["environment"]
        for name in batch_module._PROXY_ENV_NAMES
    )


def test_model_environment_forces_offline_and_restores_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "original")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with batch_module._model_loading_environment(True):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"

    assert os.environ["HF_HUB_OFFLINE"] == "original"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "HF_DATASETS_OFFLINE" not in os.environ
    assert "HF_HUB_DISABLE_TELEMETRY" not in os.environ


def test_video_logs_are_isolated_and_retries_append(
    tmp_path: Path,
) -> None:
    first_logger = logging.getLogger("test.video.first")
    second_logger = logging.getLogger("test.video.second")
    previous_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.INFO)
    try:
        with _video_log_context(tmp_path, "video-a", "worker-1"):
            first_logger.info("first attempt payload")
        with _video_log_context(tmp_path, "video-b", "worker-2"):
            second_logger.info("second video payload")
        with _video_log_context(tmp_path, "video-a", "worker-3"):
            first_logger.info("retry payload")
    finally:
        logging.getLogger().setLevel(previous_level)

    first_log = (tmp_path / "log" / "video-a.log").read_text()
    second_log = (tmp_path / "log" / "video-b.log").read_text()
    assert "first attempt payload" in first_log
    assert "retry payload" in first_log
    assert "second video payload" not in first_log
    assert "second video payload" in second_log
    assert "first attempt payload" not in second_log


def test_child_vipe_logger_uses_inherited_video_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vipe.utils.logging import configure_logging

    log_path = tmp_path / "log" / "video-a.log"
    monkeypatch.setenv(
        batch_module._VIDEO_LOG_ENV,
        str(log_path),
    )
    vipe_logger = logging.getLogger("vipe")
    root_logger = logging.getLogger()
    original_vipe_handlers = list(vipe_logger.handlers)
    original_root_handlers = list(root_logger.handlers)
    original_propagate = vipe_logger.propagate
    original_level = vipe_logger.level
    try:
        configured = configure_logging()
        configured.info("child slam message")
        logging.getLogger("huggingface_hub").warning(
            "child hub warning"
        )
        for handler in configured.handlers + root_logger.handlers:
            handler.flush()
    finally:
        for handler in list(vipe_logger.handlers):
            if handler not in original_vipe_handlers:
                vipe_logger.removeHandler(handler)
                handler.close()
        for handler in list(root_logger.handlers):
            if handler not in original_root_handlers:
                root_logger.removeHandler(handler)
                handler.close()
        vipe_logger.propagate = original_propagate
        vipe_logger.setLevel(original_level)

    text = log_path.read_text()
    assert "child slam message" in text
    assert "child hub warning" in text
