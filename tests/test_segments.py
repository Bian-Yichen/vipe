# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vipe_roomtour.segments import parse_segment


def test_parse_segment():
    segment = parse_segment("upper_floor:105:158.5")
    assert segment.name == "upper_floor"
    assert segment.start_seconds == 105
    assert segment.duration_seconds == 53.5


@pytest.mark.parametrize("spec", ["bad", "x:2:1", "../x:0:1", "x:-1:2"])
def test_invalid_segment(spec):
    with pytest.raises(ValueError):
        parse_segment(spec)
