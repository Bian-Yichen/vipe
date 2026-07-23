# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and normalize artifacts produced by the stock ViPE pipeline."""

from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .depth_options import DenseDepthOptions
from .geometry import intrinsics_matrix

LOOP_CLOSURE_CACHE_VERSION = 3


@dataclass(frozen=True)
class ArtifactSet:
    root: Path
    name: str

    @property
    def rgb(self) -> Path:
        return self.root / "rgb" / f"{self.name}.mp4"

    @property
    def pose(self) -> Path:
        return self.root / "pose" / f"{self.name}.npz"

    @property
    def intrinsics(self) -> Path:
        return self.root / "intrinsics" / f"{self.name}.npz"

    @property
    def camera_type(self) -> Path:
        return self.root / "intrinsics" / f"{self.name}_camera.txt"

    @property
    def depth(self) -> Path:
        return self.root / "depth" / f"{self.name}.zip"

    @property
    def info(self) -> Path:
        return self.root / "vipe" / f"{self.name}_info.pkl"

    @property
    def slam_map(self) -> Path:
        return self.root / "vipe" / f"{self.name}_slam_map.pt"

    @property
    def depth_metadata(self) -> Path:
        return self.root / "depth" / f"{self.name}_metadata.json"

    @property
    def loop_closure_report(self) -> Path:
        return self.root / "vipe" / f"{self.name}_loop_closure.json"

    def validate_calibration(self) -> None:
        missing = [
            path
            for path in (self.pose, self.intrinsics, self.camera_type, self.info, self.slam_map)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing VIPE calibration checkpoint: " + ", ".join(str(p) for p in missing))

    def validate(self) -> None:
        missing = [p for p in (self.rgb, self.pose, self.intrinsics, self.depth) if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing VIPE artifacts: " + ", ".join(str(p) for p in missing))

    def slam_matches(self, loop_closure: bool) -> bool:
        try:
            self.validate_calibration()
            with self.info.open("rb") as handle:
                metadata = pickle.load(handle)
            enabled_matches = bool(metadata.get("loop_closure_enabled", False)) == bool(loop_closure)
            version_matches = (
                not loop_closure
                or int(metadata.get("loop_closure_version", 0)) == LOOP_CLOSURE_CACHE_VERSION
            )
            return enabled_matches and version_matches
        except (FileNotFoundError, OSError, EOFError, pickle.UnpicklingError, TypeError):
            return False

    def depth_matches(self, options: DenseDepthOptions, *, loop_closure: bool = False) -> bool:
        if not all(path.exists() for path in (self.rgb, self.depth, self.pose, self.intrinsics)):
            return False
        if not self.depth_metadata.exists():
            # Artifacts produced by the preceding streaming branch had exactly
            # the quality preset but no sidecar. Reuse those without an
            # expensive, numerically redundant DAv3 rerun.
            return (
                not loop_closure
                and options.inference_config()
                == DenseDepthOptions.from_preset("quality").inference_config()
            )
        try:
            saved = json.loads(self.depth_metadata.read_text())
            expected = {
                **options.inference_config(),
                "slam_loop_closure": bool(loop_closure),
                "slam_loop_closure_version": LOOP_CLOSURE_CACHE_VERSION if loop_closure else 0,
            }
            saved.setdefault("slam_loop_closure", False)
            saved.setdefault("slam_loop_closure_version", 0)
            return saved == expected
        except (OSError, ValueError, TypeError):
            return False


@dataclass(frozen=True)
class Calibration:
    indices: np.ndarray
    c2w: np.ndarray
    w2c: np.ndarray
    intrinsics: np.ndarray
    K: np.ndarray
    camera_types: tuple[str, ...]


def discover_artifacts(root: Path, name: str | None = None) -> list[ArtifactSet]:
    root = Path(root)
    if name is not None:
        result = [ArtifactSet(root, name)]
    else:
        result = [ArtifactSet(root, path.stem) for path in sorted((root / "pose").glob("*.npz"))]
    if not result:
        raise FileNotFoundError(f"No pose/*.npz artifacts found below {root}")
    for artifact in result:
        artifact.validate()
    return result


def _camera_type_by_index(path: Path, indices: np.ndarray) -> tuple[str, ...]:
    if not path.exists():
        return tuple("PINHOLE" for _ in indices)
    mapping: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        index, camera_type = line.split(":", 1)
        mapping[int(index.strip())] = camera_type.strip().upper()
    return tuple(mapping.get(int(index), "PINHOLE") for index in indices)


def load_calibration(artifact: ArtifactSet) -> Calibration:
    """Join pose and intrinsics artifacts by frame index.

    ViPE stores OpenCV camera-to-world matrices and original-resolution pixel
    intrinsics.  Both may technically be sparse, so they are explicitly joined
    instead of being assumed to have identical row order.
    """

    pose_npz = np.load(artifact.pose)
    intr_npz = np.load(artifact.intrinsics)
    pose_by_index = {int(i): p for i, p in zip(pose_npz["inds"], pose_npz["data"], strict=True)}
    intr_by_index = {int(i): k for i, k in zip(intr_npz["inds"], intr_npz["data"], strict=True)}
    indices = np.array(sorted(pose_by_index.keys() & intr_by_index.keys()), dtype=np.int64)
    if len(indices) == 0:
        raise ValueError(f"Pose and intrinsics have no common frames for {artifact.name}")
    c2w = np.stack([pose_by_index[int(i)] for i in indices]).astype(np.float64)
    intrinsics = np.stack([intr_by_index[int(i)] for i in indices]).astype(np.float64)
    if intrinsics.shape[1] != 4:
        raise ValueError(f"Expected [fx,fy,cx,cy], got {intrinsics.shape}")
    camera_types = _camera_type_by_index(artifact.camera_type, indices)
    unsupported = sorted(set(camera_types) - {"PINHOLE"})
    if unsupported:
        raise NotImplementedError(
            "The RGB-D map exporter currently supports PINHOLE artifacts only; "
            f"found {unsupported}. Pose/intrinsics artifacts are still valid."
        )
    w2c = np.linalg.inv(c2w)
    K = np.stack([intrinsics_matrix(row) for row in intrinsics])
    return Calibration(indices, c2w, w2c, intrinsics, K, camera_types)


def export_calibration(
    calibration: Calibration,
    output_dir: Path,
    *,
    fps: float,
    source_time_offset: float = 0.0,
    image_wh: tuple[int, int] | None = None,
) -> None:
    """Write convenient per-frame matrices alongside the native VIPE files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = calibration.indices.astype(np.float64) / fps
    np.savez_compressed(
        output_dir / "calibration.npz",
        frame_indices=calibration.indices,
        timestamps_seconds=timestamps,
        source_timestamps_seconds=timestamps + source_time_offset,
        intrinsics=calibration.intrinsics,
        K=calibration.K,
        c2w=calibration.c2w,
        w2c=calibration.w2c,
    )

    c2w_names = [f"c2w_{r}{c}" for r in range(4) for c in range(4)]
    w2c_names = [f"w2c_{r}{c}" for r in range(4) for c in range(4)]
    header = [
        "frame_index",
        "timestamp_seconds",
        "source_timestamp_seconds",
        "fx",
        "fy",
        "cx",
        "cy",
        *c2w_names,
        *w2c_names,
    ]
    with (output_dir / "per_frame_calibration.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, frame_index in enumerate(calibration.indices):
            writer.writerow(
                [
                    int(frame_index),
                    float(timestamps[i]),
                    float(timestamps[i] + source_time_offset),
                    *calibration.intrinsics[i].tolist(),
                    *calibration.c2w[i].reshape(-1).tolist(),
                    *calibration.w2c[i].reshape(-1).tolist(),
                ]
            )

    convention = {
        "pose_convention": "OpenCV camera-to-world (c2w)",
        "camera_axes": {"+x": "right", "+y": "down", "+z": "forward"},
        "intrinsics_order": ["fx", "fy", "cx", "cy"],
        "intrinsics_coordinate_system": "original RGB artifact pixel coordinates",
        "image_width": image_wh[0] if image_wh else None,
        "image_height": image_wh[1] if image_wh else None,
        "fps": fps,
        "source_time_offset_seconds": source_time_offset,
        "notes": "w2c is the exact matrix inverse of c2w; K is reconstructed from [fx,fy,cx,cy].",
    }
    (output_dir / "coordinate_convention.json").write_text(json.dumps(convention, indent=2) + "\n")


def write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    observations: np.ndarray | None = None,
) -> None:
    """Write a compact binary little-endian colored PLY."""

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and colors must both be Nx3")
    if observations is None:
        observations = np.ones(len(points), dtype=np.uint16)
    observations = np.clip(np.asarray(observations), 0, 65535).astype(np.uint16)
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("observations", "<u2"),
        ]
    )
    vertices = np.empty(len(points), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["observations"] = observations
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property ushort observations\nend_header\n"
    ).encode("ascii")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
