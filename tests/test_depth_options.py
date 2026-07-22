from __future__ import annotations

import importlib
import json
import pickle
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vipe_roomtour.artifacts import ArtifactSet
from vipe_roomtour.depth_options import DenseDepthOptions


def test_presets_keep_existing_overlap() -> None:
    preview = DenseDepthOptions.from_preset("preview")
    balanced = DenseDepthOptions.from_preset("balanced")
    quality = DenseDepthOptions.from_preset("quality")

    assert (preview.model, preview.frame_step, preview.process_res_method) == (
        "large",
        5,
        "upper_bound_resize",
    )
    assert (balanced.model, balanced.frame_step) == ("large", 2)
    assert (quality.model, quality.frame_step, quality.output_resolution) == (
        "giant",
        1,
        "original",
    )
    assert {preview.overlap_size, balanced.overlap_size, quality.overlap_size} == {3}


def test_all_advertised_dav3_architecture_configs_are_vendored() -> None:
    config_dir = Path(__file__).parents[1] / "vipe" / "priors" / "depth" / "dav3" / "configs"
    for model in ("giant", "large", "base", "small"):
        config = config_dir / f"da3-{model}.yaml"
        assert config.is_file(), f"Missing DAv3 architecture config: {config}"
        loaded = OmegaConf.load(config)
        for component in ("net", "head", "cam_enc", "cam_dec"):
            target = loaded[component]["__object__"]
            module = importlib.import_module(str(target["path"]))
            assert getattr(module, str(target["name"]), None) is not None


def test_explicit_depth_overrides_and_sparse_indices() -> None:
    options = DenseDepthOptions.from_preset(
        "preview",
        model="base",
        frame_step=4,
        process_res=672,
        process_res_method="lower_bound_resize",
        output_resolution="original",
    )
    assert options.expected_frame_indices(14) == [0, 4, 8, 12]
    assert options.model == "base"
    assert options.process_res == 672


def test_resume_shard_must_align_with_unchanged_window_stride() -> None:
    options = DenseDepthOptions.from_preset("quality")
    assert options.resume_inference_ordinal(0) == 0
    assert options.resume_inference_ordinal(490) == 483
    with pytest.raises(ValueError, match="multiple"):
        DenseDepthOptions.from_preset("quality", shard_size=500)


def test_artifact_depth_config_match(tmp_path: Path) -> None:
    options = DenseDepthOptions.from_preset("preview")
    artifact = ArtifactSet(tmp_path, "tour")
    artifact.rgb.parent.mkdir(parents=True)
    artifact.depth.parent.mkdir(parents=True, exist_ok=True)
    artifact.rgb.touch()
    artifact.depth.touch()
    artifact.depth_metadata.write_text(json.dumps(options.inference_config()))
    assert artifact.depth_matches(options)
    assert not artifact.depth_matches(DenseDepthOptions.from_preset("quality"))


def test_artifacts_do_not_cross_reuse_loop_closure_variants(tmp_path: Path) -> None:
    options = DenseDepthOptions.from_preset("preview")
    artifact = ArtifactSet(tmp_path, "tour")
    for path in (
        artifact.rgb,
        artifact.depth,
        artifact.pose,
        artifact.intrinsics,
        artifact.camera_type,
        artifact.info,
        artifact.slam_map,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    with artifact.info.open("wb") as handle:
        pickle.dump({"loop_closure_enabled": True}, handle)
    artifact.depth_metadata.write_text(
        json.dumps({**options.inference_config(), "slam_loop_closure": True})
    )

    assert artifact.slam_matches(True)
    assert not artifact.slam_matches(False)
    assert artifact.depth_matches(options, loop_closure=True)
    assert not artifact.depth_matches(options, loop_closure=False)
