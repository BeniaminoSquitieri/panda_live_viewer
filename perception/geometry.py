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


@dataclass(frozen=True)
class PoseEstimate:
    translation: dict[str, float]
    quaternion_xyzw: dict[str, float]
    covariance: list[float]
    confidence: float
    warnings: list[str]
    residual_m: float
    inlier_ratio: float


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
    raw_k = getattr(msg, "k", None)
    if raw_k is None or len(raw_k) == 0:
        raw_k = getattr(msg, "K", None)
    k = list(raw_k) if raw_k is not None else []
    if len(k) != 9:
        raise ValueError("CameraInfo.k must contain 9 values.")
    return CameraIntrinsics(
        width=int(msg.width),
        height=int(msg.height),
        fx=float(k[0]),
        fy=float(k[4]),
        cx=float(k[2]),
        cy=float(k[5]),
    )


def decode_depth_image(msg: Any) -> np.ndarray:
    """Decode a ROS Image-like depth message into a 2D numpy array."""
    encoding = str(getattr(msg, "encoding", "")).upper()
    width = int(msg.width)
    height = int(msg.height)
    data = msg.data

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


def depth_to_meters(depth: np.ndarray, scale_m: float = 0.001) -> np.ndarray:
    """Convert supported depth arrays to meters.

    ``scale_m`` is the meters-per-raw-unit factor applied to integer depth maps.
    The ROS convention for 16UC1 is millimeters (0.001), but some RealSense
    devices (e.g. D405) report depth in 0.1 mm units (0.0001). float32 depth is
    assumed to already be in meters.
    """
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) * float(scale_m)
    return depth.astype(np.float32)


def back_project_mask(
    *,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_points: int = 4096,
    max_depth_m: float = 0.0,
    depth_band_m: float = 0.0,
    depth_scale_m: float = 0.001,
) -> np.ndarray:
    """Back-project valid masked depth pixels into camera-frame XYZ points.

    Optional depth gating removes background contamination before pose fitting:
    - ``max_depth_m`` (>0): drop pixels farther than this absolute range. For
      tabletop manipulation this discards far walls/floor captured by a loose
      segmentation mask.
    - ``depth_band_m`` (>0): keep only the nearest cluster, i.e. pixels whose
      depth is within ``depth_band_m`` of a robust near-depth estimate (10th
      percentile). This isolates the foreground object from residual background.
    """
    if depth.shape != mask.shape:
        raise ValueError(f"Depth shape {depth.shape} must match mask shape {mask.shape}.")
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"Depth shape {depth.shape} does not match intrinsics "
            f"{intrinsics.width}x{intrinsics.height}."
        )

    depth_m = depth_to_meters(depth, scale_m=depth_scale_m)
    valid = mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0.0)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth_m[rows, cols]

    if max_depth_m > 0.0:
        keep = z <= max_depth_m
        rows, cols, z = rows[keep], cols[keep], z[keep]
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float32)

    if depth_band_m > 0.0:
        near = float(np.percentile(z, 10.0))
        keep = np.abs(z - near) <= depth_band_m
        rows, cols, z = rows[keep], cols[keep], z[keep]
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float32)

    if rows.size > max_points:
        indices = np.linspace(0, rows.size - 1, num=max_points, dtype=np.int64)
        rows = rows[indices]
        cols = cols[indices]
        z = z[indices]

    x = (cols.astype(np.float32) - intrinsics.cx) * z / intrinsics.fx
    y = (rows.astype(np.float32) - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def robust_filter_points(
    points: np.ndarray,
    *,
    mad_scale: float = 4.5,
    minimum_radius_m: float = 0.003,
) -> np.ndarray:
    """Remove spatial outliers using a median/MAD radial filter.

    The median center is resistant to background pixels left by an imperfect
    mask. A small metric floor prevents a nearly rigid depth patch from being
    reduced to too few points by numerical noise.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}.")
    if points.shape[0] < 4:
        return points
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    robust_sigma = 1.4826 * mad
    radius = max(median_distance + mad_scale * robust_sigma, float(minimum_radius_m))
    filtered = points[distances <= radius]
    return filtered if filtered.shape[0] >= 3 else points


def _normalize_quaternion_xyzw(quaternion: dict[str, float]) -> dict[str, float]:
    norm = float(
        np.sqrt(
            quaternion["x"] * quaternion["x"]
            + quaternion["y"] * quaternion["y"]
            + quaternion["z"] * quaternion["z"]
            + quaternion["w"] * quaternion["w"]
        )
    )
    if norm <= 0.0:
        return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    return {key: float(value / norm) for key, value in quaternion.items()}


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> dict[str, float]:
    """Convert a 3x3 rotation matrix to a normalized x/y/z/w quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix.")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = float(np.sqrt(trace + 1.0) * 2.0)
        quaternion = {
            "x": (matrix[2, 1] - matrix[1, 2]) / scale,
            "y": (matrix[0, 2] - matrix[2, 0]) / scale,
            "z": (matrix[1, 0] - matrix[0, 1]) / scale,
            "w": 0.25 * scale,
        }
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = float(np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0)
            quaternion = {
                "x": 0.25 * scale,
                "y": (matrix[0, 1] + matrix[1, 0]) / scale,
                "z": (matrix[0, 2] + matrix[2, 0]) / scale,
                "w": (matrix[2, 1] - matrix[1, 2]) / scale,
            }
        elif index == 1:
            scale = float(np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0)
            quaternion = {
                "x": (matrix[0, 1] + matrix[1, 0]) / scale,
                "y": 0.25 * scale,
                "z": (matrix[1, 2] + matrix[2, 1]) / scale,
                "w": (matrix[0, 2] - matrix[2, 0]) / scale,
            }
        else:
            scale = float(np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0)
            quaternion = {
                "x": (matrix[0, 2] + matrix[2, 0]) / scale,
                "y": (matrix[1, 2] + matrix[2, 1]) / scale,
                "z": 0.25 * scale,
                "w": (matrix[1, 0] - matrix[0, 1]) / scale,
            }

    return _normalize_quaternion_xyzw(quaternion)


def quaternion_to_rotation_matrix_xyzw(quaternion: dict[str, float]) -> np.ndarray:
    """Convert an x/y/z/w quaternion into a 3x3 rotation matrix."""
    q = _normalize_quaternion_xyzw(quaternion)
    x, y, z, w = q["x"], q["y"], q["z"], q["w"]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _right_handed_principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = points.astype(np.float64) - np.mean(points.astype(np.float64), axis=0)
    if points.shape[0] < 2:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), centered
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    axes = eigenvectors[:, order]

    for axis_index in range(3):
        axis = axes[:, axis_index]
        dominant_index = int(np.argmax(np.abs(axis)))
        if axis[dominant_index] < 0.0:
            axes[:, axis_index] = -axis

    if np.linalg.det(axes) < 0.0:
        axes[:, 2] = -axes[:, 2]
    return axes, eigenvalues, centered


def centroid_pose(points: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate a pose from the masked point cloud centroid and PCA axes."""
    if points.size == 0:
        raise ValueError("Cannot estimate pose from an empty point cloud.")
    # Median is robust to residual table/background pixels left by box-driven
    # segmentation. PCA still uses all retained points for shape orientation.
    centroid = np.median(points, axis=0)
    axes, _, _ = _right_handed_principal_axes(points)
    translation = {"x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2])}
    quaternion = rotation_matrix_to_quaternion_xyzw(axes)
    return translation, quaternion


def _orientation_stds_from_eigenvalues(eigenvalues: np.ndarray) -> tuple[np.ndarray, list[str], float]:
    total = float(np.sum(eigenvalues)) + 1e-12
    linearity = float((eigenvalues[0] - eigenvalues[1]) / total)
    planarity = float((eigenvalues[1] - eigenvalues[2]) / total)
    surface_thickness = float(eigenvalues[2] / total)
    warnings: list[str] = []
    if linearity < 0.08:
        warnings.append("pca_primary_axis_ambiguous")
    if planarity < 0.04:
        warnings.append("pca_secondary_axis_ambiguous")

    ambiguity = np.array(
        [
            1.0 - min(planarity * 8.0, 1.0),
            1.0 - min(linearity * 6.0, 1.0),
            1.0 - min((linearity + planarity) * 4.0, 1.0),
        ],
        dtype=np.float64,
    )
    orientation_stds = 0.08 + 1.2 * ambiguity + min(surface_thickness * 8.0, 0.5)
    orientation_stds = np.clip(orientation_stds, 0.08, 1.57)
    confidence = max(0.1, min((linearity + planarity) * 2.0, 1.0))
    return orientation_stds, warnings, confidence


def diagonal_covariance(
    points: np.ndarray,
    *,
    orientation_stds_rad: np.ndarray | None = None,
) -> list[float]:
    """Return a flattened 6x6 covariance for translation and orientation."""
    if points.size == 0:
        xyz_var = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    elif points.shape[0] == 1:
        xyz_var = np.array([0.0025, 0.0025, 0.0025], dtype=np.float32)
    else:
        xyz_var = np.maximum(np.var(points, axis=0) / max(points.shape[0], 1), 1e-6)
    if orientation_stds_rad is None:
        orientation_stds_rad = np.array([1.57, 1.57, 1.57], dtype=np.float64)

    diag = [
        float(xyz_var[0]),
        float(xyz_var[1]),
        float(xyz_var[2]),
        float(orientation_stds_rad[0] * orientation_stds_rad[0]),
        float(orientation_stds_rad[1] * orientation_stds_rad[1]),
        float(orientation_stds_rad[2] * orientation_stds_rad[2]),
    ]
    covariance = [0.0] * 36
    for index, value in enumerate(diag):
        covariance[index * 6 + index] = value
    return covariance


def estimate_pose_pca(points: np.ndarray) -> PoseEstimate:
    """Estimate 6D pose from masked point cloud principal axes."""
    if points.size == 0:
        raise ValueError("Cannot estimate pose from an empty point cloud.")
    translation, quaternion = centroid_pose(points)
    _, eigenvalues, centered = _right_handed_principal_axes(points)
    orientation_stds, warnings, shape_confidence = _orientation_stds_from_eigenvalues(eigenvalues)
    covariance = diagonal_covariance(points, orientation_stds_rad=orientation_stds)
    distances = np.linalg.norm(centered, axis=1)
    median_distance = float(np.median(distances)) if distances.size else 0.0
    inlier_ratio = (
        1.0
        if median_distance <= 0.0
        else float(np.mean(distances <= 3.0 * median_distance))
    )
    residual_m = float(np.mean(distances)) if distances.size else 0.0
    confidence = max(0.0, min(shape_confidence * inlier_ratio, 1.0))
    return PoseEstimate(
        translation=translation,
        quaternion_xyzw=quaternion,
        covariance=covariance,
        confidence=confidence,
        warnings=warnings,
        residual_m=residual_m,
        inlier_ratio=inlier_ratio,
    )


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


def rotate_covariance_6x6_xyzw(covariance: list[float], quaternion: dict[str, float]) -> list[float]:
    """Rotate translation and rotation covariance blocks into a new frame."""
    if len(covariance) != 36:
        raise ValueError("6D covariance must contain 36 values.")
    rotation = quaternion_to_rotation_matrix_xyzw(quaternion)
    transform = np.zeros((6, 6), dtype=np.float64)
    transform[:3, :3] = rotation
    transform[3:, 3:] = rotation
    cov = np.asarray(covariance, dtype=np.float64).reshape((6, 6))
    rotated = transform @ cov @ transform.T
    return [float(value) for value in rotated.reshape(-1)]
