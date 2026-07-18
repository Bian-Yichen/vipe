# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


def test_basic_vipe_does_not_import_hydra():
    source_path = Path(__file__).parents[1] / "vipe_roomtour" / "basic_vipe.py"
    tree = ast.parse(source_path.read_text())
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules |= {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(module == "hydra" or module.startswith("hydra.") for module in imported_modules)

