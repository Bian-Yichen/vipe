#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the room-tour CLI without relying on an installed console-script shim."""

from vipe_roomtour.cli import main


if __name__ == "__main__":
    main()
