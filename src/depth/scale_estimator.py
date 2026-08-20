"""Online metric scale from odom translation vs depth scan-match."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..util import scan as scan_util


@dataclass
class ScaleEstimatorConfig:
    min_translation_m: float = 0.4
    max_yaw_rate_rad: float = 0.4
    ema: float = 0.2
    scale_min: float = 0.3
    scale_max: float = 3.0
    min_match_fraction: float = 0.15


@dataclass
class ScaleUpdate:
    accepted: bool
    scale: float
    ratio: Optional[float] = None
    odom_dx: float = 0.0
    depth_dx: float = 0.0
    reason: str = ""


@dataclass
class ScaleEstimator:
    """EMA scale so ``scaled_depth ≈ true_meters``.

    Compares integrated forward odom (wheel ``linear_velocity``) to forward
    range-flow on consecutive LaserScans projected from the depth cloud.
    Gyro gates high-yaw windows; accel-only sensors never update (no trustworthy
    translation).
    """

    scale: float = 1.0
    cfg: ScaleEstimatorConfig = field(default_factory=ScaleEstimatorConfig)
    updates: int = 0
    last_ratio: Optional[float] = None
    last_odom_dx: float = 0.0
    last_depth_dx: float = 0.0
    last_reason: str = "init"
    _prev_scan: Optional[scan_util.LaserScan2D] = None
    _odom_accum_m: float = 0.0
    _has_wheel_twist: bool = False

    def note_wheel_twist(self, available: bool) -> None:
        self._has_wheel_twist = bool(available)

    def integrate_odom(self, vx: float, vy: float, dt_s: float) -> None:
        if dt_s <= 0:
            return
        forward = float(vx)
        if abs(forward) < 1e-4:
            forward = math.copysign(math.hypot(float(vx), float(vy)), float(vx) or 1.0)
        self._odom_accum_m += forward * float(dt_s)

    def reset_window(self) -> None:
        self._odom_accum_m = 0.0
        self._prev_scan = None

    def observe_scan(
        self,
        scan: scan_util.LaserScan2D,
        *,
        yaw_rate_rad: float,
    ) -> ScaleUpdate:
        cfg = self.cfg
        if not self._has_wheel_twist:
            self._prev_scan = scan
            self.last_reason = "no_linear_velocity"
            return ScaleUpdate(False, self.scale, reason=self.last_reason)

        if abs(float(yaw_rate_rad)) > cfg.max_yaw_rate_rad:
            self.reset_window()
            self._prev_scan = scan
            self.last_reason = "yaw_rate_high"
            return ScaleUpdate(False, self.scale, reason=self.last_reason)

        prev = self._prev_scan
        self._prev_scan = scan
        if prev is None:
            self.last_reason = "need_prev_scan"
            return ScaleUpdate(False, self.scale, reason=self.last_reason)

        odom_dx = float(self._odom_accum_m)
        self._odom_accum_m = 0.0
        if abs(odom_dx) < cfg.min_translation_m:
            self.last_reason = "odom_translation_small"
            self.last_odom_dx = odom_dx
            return ScaleUpdate(False, self.scale, odom_dx=odom_dx, reason=self.last_reason)

        motion = scan_util.estimate_forward_range_flow(
            prev, scan, dtheta=0.0, min_abs_dx=0.02
        )
        if motion is None or motion.match_fraction < cfg.min_match_fraction:
            self.last_reason = "scan_match_weak"
            self.last_odom_dx = odom_dx
            return ScaleUpdate(False, self.scale, odom_dx=odom_dx, reason=self.last_reason)

        depth_dx = float(motion.dx)
        if abs(depth_dx) < 0.02:
            self.last_reason = "depth_dx_small"
            return ScaleUpdate(
                False, self.scale, odom_dx=odom_dx, depth_dx=depth_dx, reason=self.last_reason
            )

        ratio = odom_dx / depth_dx
        if ratio <= 0:
            self.last_reason = "ratio_nonpositive"
            return ScaleUpdate(
                False,
                self.scale,
                ratio=ratio,
                odom_dx=odom_dx,
                depth_dx=depth_dx,
                reason=self.last_reason,
            )

        new_scale = self.scale * float(ratio)
        new_scale = float(np.clip(new_scale, cfg.scale_min, cfg.scale_max))
        alpha = float(np.clip(cfg.ema, 0.0, 1.0))
        self.scale = (1.0 - alpha) * self.scale + alpha * new_scale
        self.updates += 1
        self.last_ratio = float(ratio)
        self.last_odom_dx = odom_dx
        self.last_depth_dx = depth_dx
        self.last_reason = "updated"
        return ScaleUpdate(
            True,
            self.scale,
            ratio=self.last_ratio,
            odom_dx=odom_dx,
            depth_dx=depth_dx,
            reason=self.last_reason,
        )

    def status(self) -> dict:
        return {
            "scale": round(self.scale, 4),
            "updates": self.updates,
            "last_ratio": None if self.last_ratio is None else round(self.last_ratio, 4),
            "last_odom_dx": round(self.last_odom_dx, 4),
            "last_depth_dx": round(self.last_depth_dx, 4),
            "last_reason": self.last_reason,
            "has_wheel_twist": self._has_wheel_twist,
        }
