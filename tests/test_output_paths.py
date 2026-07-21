from __future__ import annotations

from pathlib import Path

import pytest

from vipe_roomtour.output_paths import resolve_output_path


def test_absolute_output_path_is_preserved(tmp_path: Path) -> None:
    destination = tmp_path / "outside_repo" / "result"
    assert resolve_output_path(destination) == destination.resolve()


def test_relative_output_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        resolve_output_path(Path("output"))
