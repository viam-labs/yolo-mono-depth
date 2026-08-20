"""2D laser-scan projection + forward range-flow (for online scale)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float

    def to_matrix(self) -> np.ndarray:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([[c, -s, self.x], [s, c, self.y], [0.0, 0.0, 1.0]])


@dataclass
class LaserScan2D:
    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float = 0.0
    range_max: float = float("inf")
    sensor_pose: Pose2D = Pose2D(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ScanMotionResult:
    dx: float
    dy: float
    dtheta: float
    residual: float
    improvement: float
    match_fraction: float
    method: str = "grid"


def filter_points_by_z(points: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0 or points.shape[1] < 3:
        return points
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask]


def points_to_scan(
    points: np.ndarray,
    angle_min: float = -math.pi,
    angle_max: float = math.pi,
    num_bins: int = 720,
    range_min: float = 0.05,
    range_max: float = 25.0,
) -> LaserScan2D:
    points = np.asarray(points, dtype=float)
    angle_increment = (angle_max - angle_min) / num_bins
    ranges = np.full(num_bins, np.inf)
    if points.size == 0:
        return LaserScan2D(ranges, angle_min, angle_increment, range_min, range_max)

    xy = points[:, :2]
    r = np.hypot(xy[:, 0], xy[:, 1])
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    valid = (r >= range_min) & (r <= range_max) & (ang >= angle_min) & (ang < angle_max)
    r = r[valid]
    ang = ang[valid]
    bins = ((ang - angle_min) / angle_increment).astype(int)
    bins = np.clip(bins, 0, num_bins - 1)
    for b, rng in zip(bins, r):
        if rng < ranges[b]:
            ranges[b] = rng
    return LaserScan2D(ranges, angle_min, angle_increment, range_min, range_max)


def pointcloud_to_scan(
    points: np.ndarray,
    z_min: float = -0.2,
    z_max: float = 2.0,
    sensor_pose: Pose2D = Pose2D(0.0, 0.0, 0.0),
    **scan_kwargs,
) -> LaserScan2D:
    points = np.asarray(points, dtype=float)
    if points.size and points.shape[1] >= 3:
        points = filter_points_by_z(points, z_min, z_max)
    xy = points[:, :2] if points.size else np.empty((0, 2))
    if xy.size:
        homog = np.stack([xy[:, 0], xy[:, 1], np.ones(len(xy))], axis=1)
        xy = (sensor_pose.to_matrix() @ homog.T).T[:, :2]
    return points_to_scan(xy, **scan_kwargs)


def _interpolate_range_at_angle(
    angles: np.ndarray, ranges: np.ndarray, angle: float
) -> Optional[float]:
    if angles.size == 0:
        return None
    wrapped = ((angles + math.pi) % (2 * math.pi)) - math.pi
    target = ((angle + math.pi) % (2 * math.pi)) - math.pi
    if target <= wrapped[0] or target >= wrapped[-1]:
        return None
    j = int(np.searchsorted(wrapped, target))
    if j <= 0 or j >= wrapped.size:
        return None
    a0, a1 = float(wrapped[j - 1]), float(wrapped[j])
    r0, r1 = float(ranges[j - 1]), float(ranges[j])
    if a1 == a0:
        return r0
    t = (target - a0) / (a1 - a0)
    return r0 + t * (r1 - r0)


def estimate_forward_range_flow(
    prev: LaserScan2D,
    curr: LaserScan2D,
    *,
    dtheta: float = 0.0,
    forward_window_rad: float = math.pi / 3,
    min_beams: int = 25,
    max_median_deviation_m: float = 0.12,
    min_abs_dx: float = 0.005,
) -> Optional[ScanMotionResult]:
    prev_r = np.asarray(prev.ranges, dtype=float)
    curr_r = np.asarray(curr.ranges, dtype=float)
    if prev_r.size < 20 or curr_r.size < 20:
        return None

    prev_angles = prev.angle_min + np.arange(prev_r.size, dtype=float) * prev.angle_increment
    curr_angles = curr.angle_min + np.arange(curr_r.size, dtype=float) * curr.angle_increment
    valid_prev = (
        np.isfinite(prev_r)
        & (prev_r >= float(prev.range_min))
        & (prev_r <= float(prev.range_max))
    )
    valid_curr = (
        np.isfinite(curr_r)
        & (curr_r >= float(curr.range_min))
        & (curr_r <= float(curr.range_max))
    )
    curr_angles_v = curr_angles[valid_curr]
    curr_r_v = curr_r[valid_curr]
    if curr_angles_v.size < 20:
        return None
    order = np.argsort(curr_angles_v)
    curr_angles_v = curr_angles_v[order]
    curr_r_v = curr_r_v[order]

    estimates: List[float] = []
    for angle, r0 in zip(prev_angles[valid_prev], prev_r[valid_prev]):
        if abs(angle) > forward_window_rad:
            continue
        cos_a = math.cos(angle)
        if abs(cos_a) < 0.25:
            continue
        r1 = _interpolate_range_at_angle(curr_angles_v, curr_r_v, angle - dtheta)
        if r1 is None:
            continue
        estimates.append((float(r0) - float(r1)) / cos_a)

    if len(estimates) < min_beams:
        return None
    arr = np.asarray(estimates, dtype=float)
    dx = float(np.median(arr))
    mad = float(np.median(np.abs(arr - dx)))
    if mad > max_median_deviation_m:
        return None
    if abs(dx) < min_abs_dx:
        return None
    return ScanMotionResult(
        dx,
        0.0,
        float(dtheta),
        mad,
        mad,
        len(estimates) / max(int(np.sum(valid_prev)), 1),
        method="range_flow",
    )
