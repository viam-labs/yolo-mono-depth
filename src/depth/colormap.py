"""Depth map → RGB visualization (turbo-style LUT, no matplotlib)."""

from __future__ import annotations

import numpy as np

# Compact turbo-like stops (near → far). Interpolated to 256 entries once.
_TURBO_STOPS = np.asarray(
    [
        [48, 18, 59],
        [70, 70, 160],
        [45, 145, 200],
        [40, 190, 140],
        [160, 220, 60],
        [250, 180, 40],
        [245, 80, 30],
        [180, 10, 20],
    ],
    dtype=np.float32,
)


def _build_lut() -> np.ndarray:
    n = len(_TURBO_STOPS)
    xs = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.clip(np.interp(t, xs, _TURBO_STOPS[:, c]), 0, 255).astype(np.uint8)
    return lut


_LUT = _build_lut()


def depth_to_color_rgb(
    depth_m: np.ndarray,
    *,
    scale: float = 1.0,
    min_depth_m: float = 0.2,
    max_depth_m: float = 12.0,
) -> np.ndarray:
    """Map metric depth to an HxWx3 uint8 RGB image (invalid → black)."""
    z = np.asarray(depth_m, dtype=np.float32) * float(scale)
    lo = float(min_depth_m)
    hi = float(max_depth_m)
    if hi <= lo:
        hi = lo + 1.0
    valid = np.isfinite(z) & (z >= lo) & (z <= hi)
    t = np.zeros(z.shape, dtype=np.float32)
    t[valid] = (z[valid] - lo) / (hi - lo)
    idx = np.clip((t * 255.0).astype(np.int32), 0, 255)
    out = _LUT[idx]
    out = np.ascontiguousarray(out, dtype=np.uint8)
    out[~valid] = 0
    return out
