# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


def test_pycg_is_not_imported_at_visualization_module_scope():
    source = (Path(__file__).parents[1] / "vipe" / "utils" / "visualization.py").read_text()
    tree = ast.parse(source)
    module_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_imports.append(node.module or "")
    assert "pycg" not in module_imports

