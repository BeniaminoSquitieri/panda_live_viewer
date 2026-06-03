"""Geometry utilities for RGB-D perception."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def stamp_to_float(stamp: Any) -> float | None:
    """Return a ROS-like stamp as seconds, or None when unavailable."""
    if stamp is None:
        return None
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    if hasattr(stamp, "stamp"):
        return stamp_to_float(stamp.stamp)
    try:
        return float(stamp)
    except (TypeError, ValueError):
        return None


def camera_info_to_intrinsics(msg: Any) -> CameraIntrinsics:
    """Convert sensor_msgs/CameraInfo-like data into intrinsics."""
    k = list(getattr(msg, "k", []) or getattr(msg, "K", []))
    if len(k) != 9:
        raise ValueError("CameraInfo.k must contain 9 values.")
    return CameraIntrinsics(
        width=int(getattr(msg, "width")),
        height=int(getattr(msg, "height")),
        fx=float(k[0]),
        fy=float(k[4]),
        cx=float(k[2]),
        cy=float(k[5]),
    )


def decode_depth_image(msg: Any) -> np.ndarray:
    """Decode a ROS Image-like depth message into a 2D numpy array."""
    encoding = str(getattr(msg, "encoding", "")).upper()
    width = int(getattr(msg, "width"))
    height = int(getattr(msg, "height"))
    data = getattr(msg, "data")

    if encoding in {"16UC1", "MONO16"}:
        dtype = np.uint16
    elif encoding == "32FC1":
        dtype = np.float32
    else:
        raise ValueError(f"Unsupported depth encoding {encoding!r}; expected 16UC1 or 32FC1.")

    array = np.frombuffer(bytes(data), dtype=dtype)
    expected = width * height
    if array.size < expected:
        raise ValueError(f"Depth image has {array.size} values; expected at least {expected}.")
    return array[:expected].reshape((height, width))


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    """Convert supported depth arrays to meters."""
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) * 0.001
    return depth.astype(np.float32)


def back_project_mask(
    *,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_points: int = 4096,
) -> np.ndarray:
    """Back-project valid masked depth pixels into camera-frame XYZ points."""
    if depth.shape != mask.shape:
        raise ValueError(f"Depth shape {depth.shape} must match mask shape {mask.shape}.")
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"Depth shape {depth.shape} does not match intrinsics "
            f"{intrinsics.width}x{intrinsics.height}."
        )

    depth_m = depth_to_meters(depth)
    valid = mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0.0)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if rows.size > max_points:
        indices = np.linspace(0, rows.size - 1, num=max_points, dtype=np.int64)
        rows = rows[indices]
        cols = cols[indices]

    z = depth_m[rows, cols]
    x = (cols.astype(np.float32) - intrinsics.cx) * z / intrinsics.fx
    y = (rows.astype(np.float32) - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def centroid_pose(points: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate a conservative object pose from the masked point cloud centroid."""
    if points.size == 0:
        raise ValueError("Cannot estimate pose from an empty point cloud.")
    centroid = np.mean(points, axis=0)
    translation = {"x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2])}
    quaternion = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    return translation, quaternion


def diagonal_covariance(points: np.ndarray, *, orientation_std_rad: float = 3.14) -> list[float]:
    """Return a flattened 6x6 covariance from point spread and unknown orientation."""
    if points.size == 0:
        xyz_var = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    elif points.shape[0] == 1:
        xyz_var = np.array([0.0025, 0.0025, 0.0025], dtype=np.float32)
    else:
        xyz_var = np.maximum(np.var(points, axis=0), 1e-6)

    diag = [
        float(xyz_var[0]),
        float(xyz_var[1]),
        float(xyz_var[2]),
        orientation_std_rad * orientation_std_rad,
        orientation_std_rad * orientation_std_rad,
        orientation_std_rad * orientation_std_rad,
    ]
    covariance = [0.0] * 36
    for index, value in enumerate(diag):
        covariance[index * 6 + index] = value
    return covariance


def quaternion_multiply_xyzw(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Multiply two quaternions represented as x/y/z/w dictionaries."""
    lx, ly, lz, lw = left["x"], left["y"], left["z"], left["w"]
    rx, ry, rz, rw = right["x"], right["y"], right["z"], right["w"]
    return {
        "x": lw * rx + lx * rw + ly * rz - lz * ry,
        "y": lw * ry - lx * rz + ly * rw + lz * rx,
        "z": lw * rz + lx * ry - ly * rx + lz * rw,
        "w": lw * rw - lx * rx - ly * ry - lz * rz,
    }


def quaternion_conjugate_xyzw(quaternion: dict[str, float]) -> dict[str, float]:
    return {
        "x": -quaternion["x"],
        "y": -quaternion["y"],
        "z": -quaternion["z"],
        "w": quaternion["w"],
    }


def rotate_vector_xyzw(vector: dict[str, float], quaternion: dict[str, float]) -> dict[str, float]:
    """Rotate a vector by a quaternion represented as x/y/z/w dictionaries."""
    vector_quaternion = {"x": vector["x"], "y": vector["y"], "z": vector["z"], "w": 0.0}
    rotated = quaternion_multiply_xyzw(
        quaternion_multiply_xyzw(quaternion, vector_quaternion),
        quaternion_conjugate_xyzw(quaternion),
    )
    return {"x": rotated["x"], "y": rotated["y"], "z": rotated["z"]}
