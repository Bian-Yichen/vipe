# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration schemas with an optional Hydra compatibility layer.

The room-tour entry point builds its small configuration directly with
OmegaConf and does not need Hydra.  Parsing the original compositional VIPE
configs remains available when the ``hydra`` package extra is installed.
"""

from __future__ import annotations

import importlib
from typing import Any

from vipe.config.base_schema import BaseConfigSchema, Field, config_to_primitive
from vipe.config.pipeline import DefaultPipelineConfig, PanoramaPipelineConfig, PipelineConfig
from vipe.config.slam import BAConfig, SLAMConfig, SparseTracksConfig
from vipe.config.streams import FrameDirStreamListConfig, RawMP4StreamListConfig, StreamsConfig
from vipe.config.vipe import ViPEConfig

_PARSE_EXPORTS = {
    "parse_typed_config",
    "parse_untyped_config",
    "register_config_resolvers",
    "validate_typed_config",
}

__all__ = [
    "BAConfig",
    "BaseConfigSchema",
    "DefaultPipelineConfig",
    "Field",
    "FrameDirStreamListConfig",
    "PanoramaPipelineConfig",
    "PipelineConfig",
    "RawMP4StreamListConfig",
    "SLAMConfig",
    "SparseTracksConfig",
    "StreamsConfig",
    "ViPEConfig",
    "config_to_primitive",
    "parse_typed_config",
    "parse_untyped_config",
    "register_config_resolvers",
    "validate_typed_config",
]


def __getattr__(name: str) -> Any:
    if name not in _PARSE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        parse = importlib.import_module("vipe.config.parse")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("hydra"):
            raise ModuleNotFoundError(
                "The original Hydra-based `vipe infer` CLI requires the optional dependency. "
                "Install it separately with `python -m pip install hydra-core`. "
                "The `vipe-roomtour` command does not require Hydra."
            ) from exc
        raise
    return getattr(parse, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
