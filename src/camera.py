"""Monocular YOLO depth camera that publishes PCD (+ optional shm).

``viam-labs:yolo-mono-depth:camera`` reads RGB from a source camera, runs
Ultralytics YOLO26 depth (``.pt`` or NCNN), backprojects to a point cloud, and
exposes it as a lidar-compatible ``rdk:component:camera``. Optional
``movement_sensor`` enables online scale from wheel odometry vs scan-match.
"""
from __future__ import annotations

import asyncio
import io
import threading
import time
from typing import ClassVar, Mapping, Optional, Sequence

import numpy as np
from typing_extensions import Self

from viam.components.camera import Camera
from viam.components.movement_sensor import MovementSensor
from viam.logging import getLogger
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

from .depth.backproject import depth_to_points, resolve_intrinsics
from .depth.colormap import depth_to_color_rgb
from .depth.scale_estimator import ScaleEstimator, ScaleEstimatorConfig
from .depth.yolo_depth import YoloDepthEstimator
from .util import pcshm
from .util import pcd as pcd_util
from .util import scan as scan_util
from .util.odom import TypedMovementSensorOdom, TypedOdomConfig

LOGGER = getLogger(__name__)


def _rgb_from_viam_image(image) -> np.ndarray:
    """Decode a Viam ``ViamImage`` / bytes payload to HxWx3 uint8 RGB."""
    data = getattr(image, "data", image)
    if isinstance(data, np.ndarray):
        arr = data
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
    raw = bytes(data) if not isinstance(data, (bytes, bytearray)) else bytes(data)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow required to decode camera images") from exc
    try:
        from viam.media.utils.pil import viam_to_pil_image

        pil = viam_to_pil_image(image)
    except Exception:
        pil = Image.open(io.BytesIO(raw))
    return np.asarray(pil.convert("RGB"), dtype=np.uint8)


def _rgb_to_jpeg(rgb: np.ndarray, *, quality: int = 85) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
        buf, format="JPEG", quality=int(quality)
    )
    return buf.getvalue()


COLOR_SOURCE_NAME = "color"
DEPTH_SOURCE_NAME = "depth"


class MonoDepth(Camera):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "yolo-mono-depth"), "camera")

    def __init__(self, name: str):
        super().__init__(name)
        self._source: Optional[Camera] = None
        self._source_name: Optional[str] = None
        self._ms: Optional[MovementSensor] = None
        self._ms_name: Optional[str] = None
        self._odom: Optional[TypedMovementSensorOdom] = None
        self._estimator: Optional[YoloDepthEstimator] = None
        self._scale_est = ScaleEstimator()
        self._shm: Optional[pcshm.Writer] = None
        self._shm_name: Optional[str] = None
        self._stop = threading.Event()
        self._produce_task: Optional[asyncio.Task] = None
        self._module_loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._latest = b""
        self._latest_jpeg = b""
        self._latest_depth_jpeg = b""
        self._latest_points: Optional[np.ndarray] = None
        self._produce_hz = 10.0
        self._timeout_s = 2.0
        self._frames = 0
        self._errors = 0
        self._last_error: Optional[str] = None
        self._last_points = 0
        self._last_infer_ms: Optional[float] = None
        self._last_grab_ms: Optional[float] = None
        self._last_decode_ms: Optional[float] = None
        self._last_cycle_ms: Optional[float] = None
        self._measured_hz: Optional[float] = None
        self._last_publish_wall: Optional[float] = None
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._hfov_deg = 70.0
        self._stride = 4
        self._min_depth_m = 0.2
        self._max_depth_m = 12.0
        self._backend = "auto"
        self._model_path = "yolo26n-depth.pt"
        self._imgsz = 416
        self._online_scale = False
        self._last_odom_wall: Optional[float] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        cam = cls(config.name)
        cam.reconfigure(config, dependencies)
        return cam

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        source = str(attrs.get("source") or attrs.get("camera") or "").strip()
        if not source:
            raise ValueError("yolo-mono-depth requires attributes.source")
        optional: list[str] = []
        ms = str(attrs.get("movement_sensor") or "").strip()
        if ms:
            optional.append(ms)
        return [source], optional

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        self.close_sync()
        attrs = struct_to_dict(config.attributes)
        source_name = str(attrs.get("source") or attrs.get("camera") or "").strip()
        self._source_name = source_name
        source = dependencies.get(Camera.get_resource_name(source_name))
        if source is None:
            raise RuntimeError(f"yolo-mono-depth source camera {source_name!r} missing")
        self._source = source  # type: ignore[assignment]

        ms_name = str(attrs.get("movement_sensor") or "").strip() or None
        self._ms_name = ms_name
        self._ms = None
        self._odom = None
        self._online_scale = False
        if ms_name:
            ms = dependencies.get(MovementSensor.get_resource_name(ms_name))
            if ms is None:
                raise RuntimeError(f"yolo-mono-depth movement_sensor {ms_name!r} missing")
            self._ms = ms  # type: ignore[assignment]
            self._odom = TypedMovementSensorOdom(
                self._ms,
                TypedOdomConfig(use_linear_velocity=True),
                logger=LOGGER,
            )
            self._online_scale = True

        self._model_path = str(attrs.get("model") or "yolo26n-depth.pt").strip()
        self._backend = str(attrs.get("backend") or "auto").strip().lower()
        self._imgsz = int(attrs.get("imgsz", 416))
        self._produce_hz = float(attrs.get("produce_hz", 10.0))
        self._timeout_s = max(float(attrs.get("timeout_s", 2.0)), 0.5)
        self._stride = max(int(attrs.get("stride", 4)), 1)
        self._min_depth_m = float(attrs.get("min_depth_m", 0.2))
        self._max_depth_m = float(attrs.get("max_depth_m", 12.0))
        self._hfov_deg = float(attrs.get("hfov_deg", 70.0))
        self._fx = float(attrs["fx"]) if attrs.get("fx") is not None else None
        self._fy = float(attrs["fy"]) if attrs.get("fy") is not None else None
        self._cx = float(attrs["cx"]) if attrs.get("cx") is not None else None
        self._cy = float(attrs["cy"]) if attrs.get("cy") is not None else None

        scale0 = float(attrs.get("scale", 1.0))
        scale_cfg = ScaleEstimatorConfig(
            min_translation_m=float(attrs.get("scale_min_translation_m", 0.4)),
            max_yaw_rate_rad=float(attrs.get("scale_max_yaw_rate_rad", 0.4)),
            ema=float(attrs.get("scale_ema", 0.2)),
            scale_min=float(attrs.get("scale_min", 0.3)),
            scale_max=float(attrs.get("scale_max", 3.0)),
        )
        self._scale_est = ScaleEstimator(scale=scale0, cfg=scale_cfg)

        self._estimator = YoloDepthEstimator(
            self._model_path,
            backend=self._backend,
            imgsz=self._imgsz,
            device=str(attrs["device"]) if attrs.get("device") is not None else None,
        )
        self._estimator.load()

        shm_name = str(attrs.get("shm_name") or "").strip() or None
        self._shm_name = shm_name
        region = int(attrs.get("shm_region_size", pcshm.DEFAULT_REGION_SIZE))
        if shm_name:
            self._shm = pcshm.open_writer(shm_name, region)

        self._frames = 0
        self._errors = 0
        self._last_error = None
        self._last_odom_wall = None
        self._measured_hz = None
        self._last_grab_ms = None
        self._last_decode_ms = None
        self._last_cycle_ms = None
        self._stop.clear()
        # CameraClient RPCs must run on the module event loop. A background
        # thread with a new loop made get_images hang until timeout (~2s).
        self._start_producer_if_possible()
        LOGGER.info(
            "yolo-mono-depth %r source=%s backend=%s model=%s imgsz=%d hz=%.1f shm=%s ms=%s",
            self.name,
            source_name,
            self._estimator.backend,
            self._model_path,
            self._imgsz,
            self._produce_hz,
            shm_name,
            ms_name,
        )

    def _start_producer_if_possible(self) -> None:
        if self._produce_hz <= 0 or self._stop.is_set():
            return
        task = self._produce_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._module_loop = loop
        self._produce_task = loop.create_task(
            self._produce_loop_async(), name=f"{self.name}-yolo-mono-depth"
        )

    def _ensure_producer(self) -> None:
        self._start_producer_if_possible()

    async def _produce_loop_async(self) -> None:
        period = 1.0 / max(self._produce_hz, 0.1)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                await self._publish_once_async()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._record_error(exc)
            remaining = period - (time.monotonic() - t0)
            if remaining <= 0:
                continue
            try:
                await asyncio.wait_for(self._stop_wait_async(), timeout=remaining)
                break
            except asyncio.TimeoutError:
                continue

    async def _stop_wait_async(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.05)

    def _record_error(self, exc: BaseException) -> None:
        self._errors += 1
        self._last_error = repr(exc)
        LOGGER.warning("yolo-mono-depth %r produce failed: %s", self.name, exc)

    async def _maybe_update_odom(self) -> tuple[float, float]:
        if self._odom is None:
            return 0.0, 0.0
        now = time.monotonic()
        dt = 0.0
        if self._last_odom_wall is not None:
            dt = max(0.0, now - self._last_odom_wall)
        self._last_odom_wall = now
        reading = await self._odom.read()
        props = await self._odom.properties()
        has_lv = bool(getattr(props, "linear_velocity_supported", False))
        self._scale_est.note_wheel_twist(has_lv)
        if has_lv and dt > 0:
            self._scale_est.integrate_odom(reading.vx, reading.vy, dt)
        return float(reading.vtheta), dt

    async def _publish_once_async(self) -> None:
        if self._stop.is_set():
            return
        source = self._source
        est = self._estimator
        if source is None or est is None:
            return

        cycle_t0 = time.monotonic()
        yaw_rate, _dt = await self._maybe_update_odom()
        grab_t0 = time.monotonic()
        images, _meta = await source.get_images(timeout=self._timeout_s)
        grab_ms = (time.monotonic() - grab_t0) * 1000.0
        if not images:
            raise RuntimeError(f"source camera {self._source_name!r} returned no images")
        if self._stop.is_set():
            return
        decode_t0 = time.monotonic()
        rgb = _rgb_from_viam_image(images[0])
        decode_ms = (time.monotonic() - decode_t0) * 1000.0
        infer_t0 = time.monotonic()
        depth = await asyncio.to_thread(est.predict, rgb)
        infer_ms = (time.monotonic() - infer_t0) * 1000.0
        h, w = rgb.shape[:2]
        fx, fy, cx, cy = resolve_intrinsics(
            w,
            h,
            fx=self._fx,
            fy=self._fy,
            cx=self._cx,
            cy=self._cy,
            hfov_deg=self._hfov_deg,
        )
        points = depth_to_points(
            depth,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            scale=self._scale_est.scale,
            stride=self._stride,
            min_depth_m=self._min_depth_m,
            max_depth_m=self._max_depth_m,
        )
        if self._online_scale and points.size:
            scan = scan_util.pointcloud_to_scan(
                points,
                z_min=-2.0,
                z_max=2.0,
                range_min=self._min_depth_m,
                range_max=self._max_depth_m,
            )
            self._scale_est.observe_scan(scan, yaw_rate_rad=yaw_rate)
            if self._scale_est.last_reason == "updated":
                points = depth_to_points(
                    depth,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    scale=self._scale_est.scale,
                    stride=self._stride,
                    min_depth_m=self._min_depth_m,
                    max_depth_m=self._max_depth_m,
                )

        pcd = pcd_util.points_to_pcd(points)
        color_jpeg = _rgb_to_jpeg(rgb)
        depth_viz = depth_to_color_rgb(
            depth,
            scale=self._scale_est.scale,
            min_depth_m=self._min_depth_m,
            max_depth_m=self._max_depth_m,
        )
        depth_jpeg = _rgb_to_jpeg(depth_viz)
        with self._write_lock:
            shm = self._shm
            if shm is not None and not self._stop.is_set():
                shm.write(pcd)
        now = time.monotonic()
        cycle_ms = (now - cycle_t0) * 1000.0
        if self._last_publish_wall is not None:
            dt = now - self._last_publish_wall
            if dt > 1e-3:
                inst_hz = 1.0 / dt
                prev = self._measured_hz
                self._measured_hz = (
                    inst_hz if prev is None else (0.7 * prev + 0.3 * inst_hz)
                )
        with self._lock:
            self._latest = pcd
            self._latest_jpeg = color_jpeg
            self._latest_depth_jpeg = depth_jpeg
            self._latest_points = points
        self._frames += 1
        self._last_points = int(points.shape[0])
        self._last_infer_ms = round(infer_ms, 2)
        self._last_grab_ms = round(grab_ms, 2)
        self._last_decode_ms = round(decode_ms, 2)
        self._last_cycle_ms = round(cycle_ms, 2)
        self._last_publish_wall = now
        self._last_error = None

    async def _wait_for_frame(
        self, timeout: Optional[float]
    ) -> tuple[bytes, bytes, bytes]:
        """Return ``(pcd, color_jpeg, depth_jpeg)`` once a frame is available."""
        wait_s = float(timeout) if timeout is not None else self._timeout_s
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            with self._lock:
                pcd = self._latest
                color = self._latest_jpeg
                depth = self._latest_depth_jpeg
            if pcd and color and depth:
                return pcd, color, depth
            if self._produce_hz > 0:
                await asyncio.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                continue
            await self._publish_once_async()
            with self._lock:
                pcd = self._latest
                color = self._latest_jpeg
                depth = self._latest_depth_jpeg
            if pcd and color and depth:
                return pcd, color, depth
            break
        raise RuntimeError("yolo-mono-depth has no frame yet")

    async def get_images(
        self,
        *,
        filter_source_names: Optional[Sequence[str]] = None,
        extra=None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        self._ensure_producer()
        _pcd, color_jpeg, depth_jpeg = await self._wait_for_frame(timeout)
        want = None
        if filter_source_names:
            want = {str(n).strip().lower() for n in filter_source_names if str(n).strip()}
        images: list[NamedImage] = []
        if want is None or COLOR_SOURCE_NAME in want or "rgb" in want:
            images.append(NamedImage(COLOR_SOURCE_NAME, color_jpeg, CameraMimeType.JPEG))
        if want is None or DEPTH_SOURCE_NAME in want:
            images.append(NamedImage(DEPTH_SOURCE_NAME, depth_jpeg, CameraMimeType.JPEG))
        return images, ResponseMetadata()

    async def get_point_cloud(
        self, *, extra=None, timeout: Optional[float] = None, **kwargs
    ) -> tuple[bytes, str]:
        self._ensure_producer()
        pcd, _color, _depth = await self._wait_for_frame(timeout)
        return pcd, CameraMimeType.PCD

    async def get_properties(self, *, timeout=None, **kwargs) -> Camera.Properties:
        return Camera.Properties(
            supports_pcd=True,
            mime_types=[CameraMimeType.JPEG, CameraMimeType.PCD],
        )

    async def do_command(
        self, command: Mapping[str, object], *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, object]:
        self._ensure_producer()
        cmd = command.get("command") if isinstance(command, Mapping) else None
        if cmd == "set_scale":
            s = float(command.get("scale", self._scale_est.scale))
            self._scale_est.scale = float(
                np.clip(s, self._scale_est.cfg.scale_min, self._scale_est.cfg.scale_max)
            )
            self._scale_est.reset_window()
            return {"scale": self._scale_est.scale}
        last_age = None
        if self._last_publish_wall is not None:
            last_age = round(time.monotonic() - self._last_publish_wall, 3)
        est = self._estimator
        return {
            "source": self._source_name,
            "movement_sensor": self._ms_name,
            "online_scale": self._online_scale,
            "backend": est.backend if est else self._backend,
            "model": (
                getattr(est, "model_path", None) if est is not None else None
            )
            or self._model_path,
            "imgsz": self._imgsz,
            "produce_hz": self._produce_hz,
            "shm_name": self._shm_name,
            "frames": self._frames,
            "errors": self._errors,
            "last_points": self._last_points,
            "last_infer_ms": self._last_infer_ms,
            "last_grab_ms": self._last_grab_ms,
            "last_decode_ms": self._last_decode_ms,
            "last_cycle_ms": self._last_cycle_ms,
            "measured_hz": (
                None if self._measured_hz is None else round(self._measured_hz, 2)
            ),
            "last_publish_age_s": last_age,
            "last_error": self._last_error,
            "latest_bytes": len(self._latest),
            "scale": self._scale_est.status(),
            "producer": (
                "running"
                if self._produce_task is not None and not self._produce_task.done()
                else "stopped"
            ),
        }

    def close_sync(self) -> None:
        self._stop.set()
        task = self._produce_task
        loop = self._module_loop
        if task is not None and loop is not None and loop.is_running():
            loop.call_soon_threadsafe(task.cancel)
        self._produce_task = None
        self._source = None
        self._ms = None
        self._odom = None
        self._estimator = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    async def close(self):
        self._stop.set()
        task = self._produce_task
        self._produce_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._source = None
        self._ms = None
        self._odom = None
        self._estimator = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None
