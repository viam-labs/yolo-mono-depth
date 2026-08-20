"""Ultralytics YOLO26 depth inference (``.pt`` or NCNN), lazy-imported."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, Optional

import numpy as np


class YoloDepthError(RuntimeError):
    pass


def _ensure_lzma_importable() -> None:
    """Torchvision imports ``lzma`` at load time; some pyenv builds lack ``_lzma``.

    Depth inference does not need LZMA. Install a minimal stub so
    ``import torchvision`` succeeds on macOS Pythons built without xz.
    """
    try:
        import lzma  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class _LZMAError(Exception):
        pass

    class _LZMADecompressor:
        def __init__(self, *args, **kwargs):
            raise _LZMAError("lzma/_lzma unavailable in this Python build")

        def decompress(self, data):  # pragma: no cover
            raise _LZMAError("lzma/_lzma unavailable in this Python build")

    stub = types.ModuleType("lzma")
    stub.LZMAError = _LZMAError
    stub.LZMADecompressor = _LZMADecompressor
    stub.FORMAT_AUTO = 0
    stub.FORMAT_XZ = 1
    stub.FORMAT_ALONE = 2
    stub.FORMAT_RAW = 3
    stub.CHECK_NONE = 0
    stub.CHECK_CRC32 = 1
    stub.CHECK_CRC64 = 4
    stub.CHECK_SHA256 = 10

    def _unavailable(*_a, **_k):
        raise _LZMAError("lzma/_lzma unavailable in this Python build")

    stub.open = _unavailable
    stub.compress = _unavailable
    stub.decompress = _unavailable
    sys.modules["lzma"] = stub


def looks_like_ncnn(model: str) -> bool:
    """True when ``model`` names or is an NCNN export dir (may not exist yet)."""
    p = Path(str(model)).expanduser()
    name = p.name
    if name.endswith("_ncnn_model") or name.endswith(".ncnn"):
        return True
    return ncnn_export_ready(p)


def ncnn_export_ready(model: str | Path) -> bool:
    """True when ``model`` is an existing directory with NCNN param weights."""
    p = Path(str(model)).expanduser()
    if not p.is_dir():
        return False
    if (p / "model.ncnn.param").is_file() or (p / "model.param").is_file():
        return True
    for child in p.iterdir():
        if child.suffix == ".param" or child.name.endswith(".ncnn.param"):
            return True
    return False


def resolve_backend(backend: str, model: str) -> str:
    b = (backend or "auto").strip().lower()
    if b in ("pt", "pytorch", "torch"):
        return "pt"
    if b == "ncnn":
        return "ncnn"
    if b != "auto":
        raise YoloDepthError(f"unknown depth backend {backend!r} (use auto|pt|ncnn)")
    return "ncnn" if looks_like_ncnn(model) else "pt"


def ncnn_dir_for_pt(pt_path: str | Path) -> Path:
    """Ultralytics default export dir next to a ``.pt`` weights file."""
    p = Path(str(pt_path)).expanduser()
    return p.with_name(f"{p.stem}_ncnn_model")


def ensure_ncnn_model(pt_path: str, *, imgsz: int, yolo_cls: Any = None) -> str:
    """Return an NCNN model directory, exporting from ``pt_path`` if needed.

    Reuses ``{stem}_ncnn_model`` beside the weights when present. First export on a
    Pi can take several minutes (download ``.pt`` + convert).
    """
    pt = Path(str(pt_path)).expanduser()
    out = ncnn_dir_for_pt(pt)
    if ncnn_export_ready(out):
        return str(out.resolve())

    if yolo_cls is None:
        from ultralytics import YOLO as yolo_cls  # type: ignore

    try:
        model = yolo_cls(str(pt))
        exported = model.export(format="ncnn", imgsz=int(imgsz))
    except Exception as exc:  # pragma: no cover - ultralytics errors vary
        raise YoloDepthError(
            f"failed to export {pt!s} to NCNN (imgsz={imgsz}): {exc}"
        ) from exc

    # Ultralytics may return a path string or Path to the export dir / zip.
    candidate = Path(str(exported)).expanduser() if exported else out
    if candidate.is_file() and candidate.suffix == ".zip":
        candidate = candidate.with_suffix("")
    for path in (candidate, out):
        if ncnn_export_ready(path):
            return str(path.resolve())
    raise YoloDepthError(
        f"NCNN export of {pt!s} did not produce a usable model dir "
        f"(tried {candidate!s} and {out!s})"
    )


class YoloDepthEstimator:
    """Thin wrapper: RGB ndarray ``(H,W,3)`` uint8 → depth meters ``(H,W)``."""

    def __init__(
        self,
        model: str = "yolo26n-depth.pt",
        *,
        backend: str = "auto",
        imgsz: int = 416,
        device: Optional[str] = None,
    ):
        self.model_path = str(model)
        self.backend = resolve_backend(backend, self.model_path)
        self.imgsz = int(imgsz)
        self.device = device
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        _ensure_lzma_importable()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional dep
            raise YoloDepthError(
                "ultralytics is required for yolo-mono-depth; "
                "install with: pip install ultralytics"
            ) from exc
        path = self.model_path
        if self.backend == "ncnn" and not looks_like_ncnn(path):
            # ``backend=ncnn`` with a ``.pt`` (including the default model name):
            # export once beside the weights, then load the NCNN directory.
            if path.endswith(".pt") and not os.path.isdir(path):
                path = ensure_ncnn_model(path, imgsz=self.imgsz, yolo_cls=YOLO)
                self.model_path = path
            else:
                raise YoloDepthError(
                    f"backend=ncnn requires an NCNN model directory or .pt weights, "
                    f"got {path!r}"
                )
        self._model = YOLO(path)
        self._model.overrides["task"] = "depth"

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Return float32 depth map in meters, shape ``(H, W)`` matching ``rgb``."""
        self.load()
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            raise YoloDepthError(f"expected HxWx3 RGB, got shape {img.shape}")
        kwargs = {"imgsz": self.imgsz, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        results = self._model.predict(source=img, **kwargs)
        if not results:
            raise YoloDepthError("YOLO depth returned no results")
        result = results[0]
        depth_obj = getattr(result, "depth", None)
        if depth_obj is None:
            raise YoloDepthError(
                "YOLO result has no .depth — is this a *-depth model / task=depth?"
            )
        data = depth_obj.data
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        depth = np.asarray(data, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim != 2:
            raise YoloDepthError(f"unexpected depth shape {depth.shape}")
        # Resize to source resolution when the network letterboxed.
        h, w = img.shape[:2]
        if depth.shape != (h, w):
            try:
                from PIL import Image
            except ImportError as exc:  # pragma: no cover
                raise YoloDepthError("Pillow required to resize depth maps") from exc
            depth = np.asarray(
                Image.fromarray(depth, mode="F").resize((w, h), Image.BILINEAR),
                dtype=np.float32,
            )
        return depth
