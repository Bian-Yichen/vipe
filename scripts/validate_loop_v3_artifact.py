#!/usr/bin/env python3
"""Offline validation of loop-closure v3 against saved ViPE artifacts.

This utility intentionally does not import Torch.  It only accepts the small
set of tensor/storage opcodes emitted by ``SLAMMap.save`` and reconstructs the
three packed NumPy arrays needed by the v3 geometric verifier.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np


class _Storage:
    def __init__(self, key: str, dtype: np.dtype, size: int) -> None:
        self.key = key
        self.dtype = dtype
        self.size = size


class _TorchArchiveUnpickler(pickle.Unpickler):
    """Restricted reader for CPU tensors saved by ``SLAMMap.save``."""

    def __init__(
        self,
        handle: io.BytesIO,
        archive: zipfile.ZipFile,
        root: str,
    ) -> None:
        super().__init__(handle)
        self.archive = archive
        self.root = root

    def find_class(self, module: str, name: str):  # noqa: ANN001
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):

            def rebuild(storage, offset, shape, stride, *_):  # noqa: ANN001
                raw = self.archive.read(f"{self.root}/data/{storage.key}")
                base = np.frombuffer(
                    raw,
                    dtype=storage.dtype,
                    count=storage.size,
                )
                view = np.lib.stride_tricks.as_strided(
                    base[offset:],
                    shape=tuple(shape),
                    strides=tuple(
                        int(value) * base.dtype.itemsize for value in stride
                    ),
                )
                return np.array(view, copy=True)

            return rebuild
        if module == "torch" and name in (
            "FloatStorage",
            "HalfStorage",
            "LongStorage",
        ):
            return {
                "FloatStorage": np.dtype("<f4"),
                "HalfStorage": np.dtype("<f2"),
                "LongStorage": np.dtype("<i8"),
            }[name]
        if (module, name) == ("torch", "device"):
            return lambda *values: values
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        raise pickle.UnpicklingError(
            f"Unsupported global in SLAM-map archive: {module}.{name}"
        )

    def persistent_load(self, persistent_id):  # noqa: ANN001
        kind, dtype, key, _location, size = persistent_id
        if kind != "storage":
            raise pickle.UnpicklingError(
                f"Unsupported persistent object: {persistent_id!r}"
            )
        return _Storage(key, dtype, size)


def _load_slam_map(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        pickle_name = next(
            name for name in archive.namelist() if name.endswith("/data.pkl")
        )
        root = pickle_name.rsplit("/", 1)[0]
        return _TorchArchiveUnpickler(
            io.BytesIO(archive.read(pickle_name)),
            archive,
            root,
        ).load()


def _load_registration_module(repository_root: Path):
    path = (
        repository_root
        / "vipe"
        / "slam"
        / "components"
        / "submap_registration.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_vipe_submap_registration",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _spatial_candidates(
    c2w: np.ndarray,
    frame_indices: np.ndarray,
    *,
    min_frame_gap: int,
    min_keyframe_gap: int,
    radius: float,
    top_k: int,
    nms: int,
    maximum: int,
) -> list[tuple[float, int, int]]:
    centers = c2w[frame_indices, :3, 3]
    raw = []
    for target in range(len(frame_indices)):
        per_target = []
        for source in range(target):
            if target - source < min_keyframe_gap:
                continue
            if frame_indices[target] - frame_indices[source] < min_frame_gap:
                continue
            distance = float(np.linalg.norm(centers[target] - centers[source]))
            if distance <= radius:
                per_target.append((distance, source, target))
        raw.extend(sorted(per_target)[:top_k])
    selected = []
    for candidate in sorted(raw):
        _, source, target = candidate
        if any(
            abs(source - old_source) <= nms
            and abs(target - old_target) <= nms
            for _, old_source, old_target in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= maximum:
            break
    return selected


def _parse_candidate(text: str) -> tuple[int, int]:
    source, separator, target = text.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError(
            "Candidate must be SOURCE_KEYFRAME:TARGET_KEYFRAME"
        )
    return int(source), int(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slam_map", type=Path)
    parser.add_argument("poses", type=Path)
    parser.add_argument("--candidate", action="append", type=_parse_candidate)
    parser.add_argument("--auto-candidates", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    registration = _load_registration_module(repository_root)
    saved_map = _load_slam_map(args.slam_map)
    poses = np.load(args.poses, allow_pickle=False)["data"]
    frame_indices = np.asarray(
        saved_map["dense_disp_frame_inds"],
        dtype=np.int64,
    )
    requested = list(args.candidate or [])
    if args.auto_candidates:
        automatic = _spatial_candidates(
            poses,
            frame_indices,
            min_frame_gap=300,
            min_keyframe_gap=12,
            radius=4.5,
            top_k=24,
            nms=6,
            maximum=args.auto_candidates,
        )
        requested.extend((source, target) for _, source, target in automatic)
    requested = list(dict.fromkeys(requested))
    if not requested:
        raise SystemExit("Pass --candidate S:T or --auto-candidates N")

    options = registration.SubmapRegistrationOptions()
    records = []
    for source, target in requested:
        result = registration.register_packed_submaps(
            saved_map["dense_disp_xyz"],
            saved_map["dense_disp_rgb"],
            saved_map["dense_disp_packinfo"],
            source,
            target,
            options,
        )
        record = {
            "source_keyframe": source,
            "target_keyframe": target,
            "source_frame": int(frame_indices[source]),
            "target_frame": int(frame_indices[target]),
            **result.to_dict(),
        }
        records.append(record)
        translation = record["target_world_to_source_world"]
        translation = (
            None
            if translation is None
            else np.round(np.asarray(translation)[:3, 3], 4).tolist()
        )
        print(
            f"{source:04d}/f{frame_indices[source]:06d} -> "
            f"{target:04d}/f{frame_indices[target]:06d}: "
            f"{result.reason}; translation={translation}; "
            f"ambiguity={result.ambiguity_ratio:.3f}"
        )

    accepted = [
        record
        for record in records
        if record["accepted"]
        and record["target_world_to_source_world"] is not None
    ]
    support = registration.independent_transform_support(
        [
            (
                record["source_keyframe"],
                record["target_keyframe"],
                np.asarray(
                    record["target_world_to_source_world"],
                    dtype=np.float64,
                ),
            )
            for record in accepted
        ],
        index_radius=8,
        max_translation=options.translation_consistency,
        max_rotation_deg=options.rotation_consistency_deg,
    )
    for record, record_support in zip(accepted, support, strict=True):
        record_support = int(record_support)
        record["independent_consistency_support"] = record_support
        record["retained_by_v3_consistency_filter"] = bool(
            record_support >= 2
        )
    retained = [
        record
        for record in accepted
        if record["retained_by_v3_consistency_filter"]
    ]
    print(
        "Independent-consistency filter: "
        f"{len(retained)}/{len(accepted)} accepted registrations retained"
    )

    payload = {
        "slam_map": str(args.slam_map),
        "poses": str(args.poses),
        "keyframes": int(len(frame_indices)),
        "options": {
            field: getattr(options, field)
            for field in options.__dataclass_fields__
        },
        "accepted_registrations": len(accepted),
        "retained_consistent_registrations": len(retained),
        "results": records,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
