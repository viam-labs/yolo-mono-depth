"""Tests for yolo-mono-depth helpers and camera (no real YOLO weights)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.depth.backproject import depth_to_points, intrinsics_from_fov, resolve_intrinsics
from src.depth.scale_estimator import ScaleEstimator, ScaleEstimatorConfig
from src.depth.yolo_depth import looks_like_ncnn, resolve_backend
from src.util import pcd as pcd_util
from src.util import pcshm
from src.util import scan as scan_util


def test_intrinsics_from_fov_sensible():
    fx, fy, cx, cy = intrinsics_from_fov(640, 480, hfov_deg=70.0)
    assert fx == pytest.approx(fy)
    assert fx > 200
    assert cx == pytest.approx(319.5)
    assert cy == pytest.approx(239.5)


def test_depth_to_points_forward_plane():
    h, w = 20, 20
    depth = np.full((h, w), 2.0, dtype=np.float32)
    fx, fy, cx, cy = resolve_intrinsics(w, h, hfov_deg=90.0)
    pts = depth_to_points(depth, fx=fx, fy=fy, cx=cx, cy=cy, stride=1, scale=1.0)
    assert pts.shape[1] == 3
    assert pts.shape[0] > 10
    assert float(np.median(pts[:, 0])) == pytest.approx(2.0, abs=0.05)


def test_depth_to_points_scale_and_filters():
    depth = np.full((10, 10), 1.0, dtype=np.float32)
    fx, fy, cx, cy = 100.0, 100.0, 4.5, 4.5
    pts = depth_to_points(
        depth, fx=fx, fy=fy, cx=cx, cy=cy, scale=2.0, stride=1, min_depth_m=0.5, max_depth_m=3.0
    )
    assert float(np.median(pts[:, 0])) == pytest.approx(2.0, abs=0.05)
    empty = depth_to_points(
        depth, fx=fx, fy=fy, cx=cx, cy=cy, scale=1.0, stride=1, min_depth_m=5.0, max_depth_m=10.0
    )
    assert empty.shape[0] == 0


def test_resolve_backend_auto_ncnn():
    assert resolve_backend("auto", "yolo26n-depth_ncnn_model") == "ncnn"
    assert resolve_backend("auto", "yolo26n-depth.pt") == "pt"
    assert resolve_backend("ncnn", "/tmp/foo_ncnn_model") == "ncnn"
    assert looks_like_ncnn("foo_ncnn_model")


def _wall_scan(dist: float) -> scan_util.LaserScan2D:
    n = 720
    ranges = np.full(n, np.inf)
    for i in range(n):
        ang = -np.pi + i * (2 * np.pi / n)
        if abs(ang) < np.radians(40):
            ranges[i] = dist / max(np.cos(ang), 0.2)
    return scan_util.LaserScan2D(ranges, -np.pi, 2 * np.pi / n, 0.05, 25.0)


def test_scale_estimator_converges():
    est = ScaleEstimator(
        scale=1.0,
        cfg=ScaleEstimatorConfig(ema=1.0, min_translation_m=0.1, min_match_fraction=0.05),
    )
    est.note_wheel_twist(True)

    prev = _wall_scan(3.0)
    curr = _wall_scan(2.0)
    motion = scan_util.estimate_forward_range_flow(prev, curr, dtheta=0.0, min_abs_dx=0.05)
    assert motion is not None
    assert motion.dx == pytest.approx(1.0, abs=0.15)

    est._prev_scan = prev
    est._odom_accum_m = float(motion.dx) * 2.0
    upd = est.observe_scan(curr, yaw_rate_rad=0.0)
    assert upd.accepted
    assert est.scale == pytest.approx(2.0, rel=0.2)


def test_scale_estimator_skips_without_wheel_twist():
    est = ScaleEstimator(scale=1.5)
    est.note_wheel_twist(False)
    scan = scan_util.LaserScan2D(np.full(100, 2.0), -np.pi, 2 * np.pi / 100, 0.05, 25.0)
    upd = est.observe_scan(scan, yaw_rate_rad=0.0)
    assert not upd.accepted
    assert est.scale == 1.5
    assert upd.reason == "no_linear_velocity"


@pytest.mark.asyncio
async def test_mono_depth_publishes_pcd_and_shm():
    from src.camera import MonoDepth

    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[:, :] = (40, 80, 120)
    depth = np.full((48, 64), 1.5, dtype=np.float32)

    fake_est = MagicMock()
    fake_est.backend = "pt"
    fake_est.predict.return_value = depth
    fake_est.load = MagicMock()

    class _Img:
        data = rgb

    source = MagicMock()
    source.get_images = AsyncMock(return_value=([_Img()], None))

    shm_name = f"/viam-test-mono-{int(time.time() * 1000) % 100000}"
    cam = MonoDepth("mono")
    cam._source = source
    cam._source_name = "cam"
    cam._estimator = fake_est
    cam._produce_hz = 0
    cam._timeout_s = 1.0
    cam._stride = 2
    cam._shm_name = shm_name
    cam._shm = pcshm.open_writer(shm_name, pcshm.DEFAULT_REGION_SIZE)
    cam._stop.clear()

    try:
        await cam._publish_once_async()
        pcd, mime = await cam.get_point_cloud(timeout=1.0)
        assert mime == "pointcloud/pcd"
        assert len(pcd) > 100
        assert cam._last_points > 0
        pts = pcd_util.parse_pcd(pcd)
        assert pts.shape[0] == cam._last_points
        with pcshm.open_reader(shm_name, pcshm.DEFAULT_REGION_SIZE) as reader:
            raw, _ts = reader.read()
            assert len(raw) > 0
        status = await cam.do_command({})
        assert status["frames"] >= 1
        assert status["scale"]["scale"] == pytest.approx(1.0)
    finally:
        cam.close_sync()


@pytest.mark.asyncio
async def test_mono_depth_scale_static_without_ms():
    from src.camera import MonoDepth

    depth = np.full((32, 32), 1.0, dtype=np.float32)
    fake_est = MagicMock()
    fake_est.backend = "pt"
    fake_est.predict.return_value = depth
    fake_est.load = MagicMock()

    class _Img:
        data = np.zeros((32, 32, 3), dtype=np.uint8)

    source = MagicMock()
    source.get_images = AsyncMock(return_value=([_Img()], None))

    cam = MonoDepth("mono")
    cam._source = source
    cam._estimator = fake_est
    cam._scale_est.scale = 1.25
    cam._online_scale = False
    cam._produce_hz = 0
    cam._stop.clear()
    await cam._publish_once_async()
    assert cam._scale_est.scale == pytest.approx(1.25)
    set_out = await cam.do_command({"command": "set_scale", "scale": 0.8})
    assert set_out["scale"] == pytest.approx(0.8)
    cam.close_sync()
