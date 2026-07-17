# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static top-down RGB/occupancy renderings for rapid SLAM QA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RasterInfo:
    min_x: float
    max_z: float
    meters_per_pixel: float
    width: int
    height: int

    def project(self, xz: np.ndarray) -> np.ndarray:
        xz = np.asarray(xz)
        px = np.rint((xz[:, 0] - self.min_x) / self.meters_per_pixel)
        py = np.rint((self.max_z - xz[:, 1]) / self.meters_per_pixel)
        return np.column_stack((px, py)).astype(np.int32)


def _make_raster_info(
    points_xz: np.ndarray,
    trajectory_xz: np.ndarray,
    resolution: float,
    margin: float,
    max_size: int,
) -> RasterInfo:
    all_xz = np.concatenate((points_xz, trajectory_xz), axis=0)
    low = np.nanmin(all_xz, axis=0) - margin
    high = np.nanmax(all_xz, axis=0) + margin
    resolution = max(float(resolution), float(np.max(high - low)) / max(max_size - 1, 1))
    width = max(1, int(math.ceil((high[0] - low[0]) / resolution)) + 1)
    height = max(1, int(math.ceil((high[1] - low[1]) / resolution)) + 1)
    return RasterInfo(float(low[0]), float(high[1]), resolution, width, height)


def _draw_grid(draw: ImageDraw.ImageDraw, info: RasterInfo) -> None:
    grid_m = 1.0
    x_start = math.ceil(info.min_x / grid_m) * grid_m
    x_end = info.min_x + (info.width - 1) * info.meters_per_pixel
    z_min = info.max_z - (info.height - 1) * info.meters_per_pixel
    z_start = math.ceil(z_min / grid_m) * grid_m
    for x in np.arange(x_start, x_end + 1e-6, grid_m):
        px = int(round((x - info.min_x) / info.meters_per_pixel))
        draw.line((px, 0, px, info.height - 1), fill=(180, 180, 180), width=1)
    for z in np.arange(z_start, info.max_z + 1e-6, grid_m):
        py = int(round((info.max_z - z) / info.meters_per_pixel))
        draw.line((0, py, info.width - 1, py), fill=(180, 180, 180), width=1)


def render_topdown(
    output_dir: Path,
    points_level: np.ndarray,
    colors: np.ndarray,
    trajectory_level: np.ndarray,
    *,
    floor_y: float,
    min_height: float = -0.10,
    max_height: float = 2.20,
    resolution: float = 0.025,
    margin: float = 0.5,
    max_size: int = 4096,
) -> RasterInfo:
    """Render RGB, occupancy, and trajectory-overlay bird's-eye images."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points_level = np.asarray(points_level)
    colors = np.asarray(colors, dtype=np.uint8)
    trajectory_level = np.asarray(trajectory_level)
    height_above_floor = floor_y - points_level[:, 1]
    keep = (
        np.isfinite(points_level).all(axis=1)
        & (height_above_floor >= min_height)
        & (height_above_floor <= max_height)
    )
    points = points_level[keep]
    colors = colors[keep]
    if len(points) == 0:
        raise ValueError("No points survive the top-down height filter")

    info = _make_raster_info(points[:, [0, 2]], trajectory_level[:, [0, 2]], resolution, margin, max_size)
    pixels = info.project(points[:, [0, 2]])
    flat = pixels[:, 1].astype(np.int64) * info.width + pixels[:, 0]
    count = np.bincount(flat, minlength=info.width * info.height).astype(np.int64)
    sums = np.stack(
        [np.bincount(flat, weights=colors[:, channel], minlength=len(count)) for channel in range(3)], axis=1
    )
    occupied = count > 0
    rgb_flat = np.full((len(count), 3), 245, dtype=np.uint8)
    rgb_flat[occupied] = np.clip(np.rint(sums[occupied] / count[occupied, None]), 0, 255).astype(np.uint8)
    rgb = rgb_flat.reshape(info.height, info.width, 3)
    Image.fromarray(rgb).save(output_dir / "topdown_rgb.png")

    occupancy = np.zeros_like(count, dtype=np.uint8)
    if occupied.any():
        occupancy[occupied] = np.clip(
            np.rint(255 * np.log1p(count[occupied]) / np.log1p(count[occupied].max())), 1, 255
        ).astype(np.uint8)
    Image.fromarray(occupancy.reshape(info.height, info.width), mode="L").save(output_dir / "topdown_occupancy.png")

    overlay = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(overlay, "RGB")
    _draw_grid(draw, info)
    trajectory_px = info.project(trajectory_level[:, [0, 2]])
    if len(trajectory_px) > 1:
        for i in range(len(trajectory_px) - 1):
            alpha = i / max(len(trajectory_px) - 2, 1)
            color = (int(30 + 210 * alpha), int(80 + 100 * (1 - alpha)), int(230 - 180 * alpha))
            draw.line((*trajectory_px[i], *trajectory_px[i + 1]), fill=color, width=max(2, info.width // 900))
    marker = max(4, info.width // 350)
    for point, color, label in (
        (trajectory_px[0], (0, 190, 0), "START"),
        (trajectory_px[-1], (220, 30, 30), "END"),
    ):
        x, y = map(int, point)
        draw.ellipse((x - marker, y - marker, x + marker, y + marker), fill=color, outline=(0, 0, 0), width=1)
        draw.text((x + marker + 2, y - marker), label, fill=(0, 0, 0), font=ImageFont.load_default())
    draw.rectangle((6, 6, 254, 40), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.text(
        (12, 12),
        f"1 m grid | {info.meters_per_pixel:.3f} m/px | trajectory: start green, end red",
        fill=(0, 0, 0),
        font=ImageFont.load_default(),
    )
    overlay.save(output_dir / "topdown_with_trajectory.png")
    return info
