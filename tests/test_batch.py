from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from vipe_roomtour.batch import (
    StateStore,
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
