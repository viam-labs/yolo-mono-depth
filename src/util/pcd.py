"""Binary PCD encode/decode for XYZ point clouds."""

from __future__ import annotations

from typing import List

import numpy as np


def points_to_pcd(points: np.ndarray) -> bytes:
    """Serialize an ``(N, 3)`` float array into a binary PCD."""
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    n = points.shape[0]
    header = (
        "VERSION .7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")
    return header + points.tobytes()


def parse_pcd(raw: bytes) -> np.ndarray:
    """Parse a PCD (ASCII or uncompressed binary) into ``(N, 3)`` XYZ."""
    if not raw:
        return np.empty((0, 3))
    idx = raw.find(b"DATA ")
    if idx < 0:
        return np.empty((0, 3))
    nl = raw.find(b"\n", idx)
    header_text = raw[:nl].decode("ascii", errors="replace")
    data_fmt = raw[idx + 5 : nl].decode("ascii").strip()
    body = raw[nl + 1 :]

    fields: List[str] = []
    sizes: List[int] = []
    types: List[str] = []
    counts: List[int] = []
    npoints = 0
    for line in header_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(x) for x in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(x) for x in parts[1:]]
        elif key == "POINTS":
            npoints = int(parts[1])
        elif key == "WIDTH" and npoints == 0:
            npoints = int(parts[1])

    if not counts:
        counts = [1] * len(fields)

    if not fields or not {"x", "y", "z"} <= set(fields):
        return np.empty((0, 3))

    if data_fmt == "ascii":
        rows = [r.split() for r in body.decode("ascii").splitlines() if r.strip()]
        arr = np.array(rows, dtype=float) if rows else np.empty((0, len(fields)))
        col = {f: i for i, f in enumerate(fields)}
        if not {"x", "y", "z"} <= set(col) or arr.size == 0:
            return np.empty((0, 3))
        return arr[:, [col["x"], col["y"], col["z"]]].astype(float)

    type_map = {
        ("F", 4): "f4",
        ("F", 8): "f8",
        ("U", 1): "u1",
        ("U", 2): "u2",
        ("U", 4): "u4",
        ("I", 1): "i1",
        ("I", 2): "i2",
        ("I", 4): "i4",
    }
    dtype_fields = []
    for f, s, t, c in zip(fields, sizes, types, counts):
        np_t = type_map.get((t.upper(), s), f"V{s}")
        for k in range(c):
            name = f if c == 1 else f"{f}_{k}"
            dtype_fields.append((name, np.dtype(np_t)))
    if not dtype_fields:
        return np.empty((0, 3))
    record = np.dtype(dtype_fields)
    if npoints == 0:
        npoints = len(body) // record.itemsize
    structured = np.frombuffer(body[: npoints * record.itemsize], dtype=record)
    if not {"x", "y", "z"} <= set(structured.dtype.names or ()):
        return np.empty((0, 3))
    return np.stack(
        [structured["x"], structured["y"], structured["z"]], axis=1
    ).astype(float)
