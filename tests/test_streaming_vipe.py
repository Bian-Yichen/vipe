from __future__ import annotations

import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch


def _install_import_stubs() -> None:
    """Let the artifact writer be tested without compiling ViPE CUDA."""

    if "vipe.ext.lietorch" not in sys.modules:
        module = types.ModuleType("vipe.ext.lietorch")
        module.SE3 = object
        sys.modules["vipe.ext.lietorch"] = module


def test_streaming_depth_writer_matches_half_precision_artifact(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("OpenEXR")
    pytest.importorskip("Imath")
    _install_import_stubs()

    # Importing the full pipeline requires optional ViPE components.  Exercise
    # the writer through a minimal module load with those imports supplied by
    # the installed project during normal CI/runtime.
    from vipe_roomtour.streaming_vipe import save_depth_artifacts_streaming
    from vipe.utils.io import read_depth_artifacts

    class Frame:
        def __init__(self, depth: torch.Tensor):
            self.metric_depth = depth

    expected = [
        torch.linspace(0.25, 4.0, 35, dtype=torch.float32).reshape(5, 7),
        torch.linspace(1.0, 8.0, 35, dtype=torch.float32).reshape(5, 7),
    ]
    output = tmp_path / "depth.zip"
    count = save_depth_artifacts_streaming(output, (Frame(depth) for depth in expected))

    assert count == 2
    assert output.exists()
    assert not (tmp_path / "depth.zip.partial").exists()
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["00000.exr", "00001.exr"]

    restored = [depth for _, depth in read_depth_artifacts(output)]
    assert len(restored) == len(expected)
    for actual, reference in zip(restored, expected, strict=True):
        np.testing.assert_array_equal(actual.numpy(), reference.half().float().numpy())


def test_streaming_depth_writer_does_not_publish_partial_archive(tmp_path: Path) -> None:
    pytest.importorskip("OpenEXR")
    pytest.importorskip("Imath")
    _install_import_stubs()
    from vipe_roomtour.streaming_vipe import save_depth_artifacts_streaming

    class Frame:
        def __init__(self, depth):
            self.metric_depth = depth

    def broken_stream():
        yield Frame(torch.ones(2, 3))
        raise RuntimeError("synthetic failure")

    output = tmp_path / "depth.zip"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        save_depth_artifacts_streaming(output, broken_stream())
    assert not output.exists()
    assert not (tmp_path / "depth.zip.partial").exists()
