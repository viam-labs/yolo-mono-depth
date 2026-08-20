"""Depth map → XYZ point cloud (camera → ROS camera_link convention)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def intrinsics_from_fov(
    width: int,
    height: int,
    *,
    hfov_deg: float = 70.0,
) -> Tuple[float, float, float, float]:
    """Approximate pinhole intrinsics from image size and horizontal FOV."""
    w = max(int(width), 1)
    h = max(int(height), 1)
    fx = (0.5 * w) / math.tan(math.radians(float(hfov_deg)) * 0.5)
    fy = fx
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    return float(fx), float(fy), float(cx), float(cy)


def depth_to_points(
    depth_m: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale: float = 1.0,
    stride: int = 4,
    min_depth_m: float = 0.2,
    max_depth_m: float = 12.0,
) -> np.ndarray:
    """Backproject a ``(H, W)`` depth map to ``(N, 3)`` XYZ meters.

    Frame is ROS ``camera_link`` style: **X forward**, Y left, Z up — suitable
    as a forward-facing lidar for nav-stack mount TFs.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth map must be 2D, got shape {depth.shape}")
    step = max(int(stride), 1)
    s = float(scale)
    z_min = float(min_depth_m)
    z_max = float(max_depth_m)

    rows = np.arange(0, depth.shape[0], step, dtype=np.float32)
    cols = np.arange(0, depth.shape[1], step, dtype=np.float32)
    uu, vv = np.meshgrid(cols, rows)
    d = depth[::step, ::step] * s
    valid = np.isfinite(d) & (d >= z_min) & (d <= z_max)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    uu = uu[valid]
    vv = vv[valid]
    d = d[valid]
    # Optical (x right, y down, z forward) → camera_link (x forward, y left, z up).
    x_opt = (uu - float(cx)) * d / float(fx)
    y_opt = (vv - float(cy)) * d / float(fy)
    x = d
    y = -x_opt
    z = -y_opt
    return np.stack([x, y, z], axis=1).astype(np.float32, copy=False)


def resolve_intrinsics(
    width: int,
    height: int,
    *,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    hfov_deg: float = 70.0,
) -> Tuple[float, float, float, float]:
    """Fill missing intrinsics from FOV defaults."""
    dfx, dfy, dcx, dcy = intrinsics_from_fov(width, height, hfov_deg=hfov_deg)
    return (
        float(fx) if fx is not None else dfx,
        float(fy) if fy is not None else dfy,
        float(cx) if cx is not None else dcx,
        float(cy) if cy is not None else dcy,
    )
