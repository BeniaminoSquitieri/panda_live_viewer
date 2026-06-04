"""Compute a hand-eye ``camera_static_tf_map`` entry from 3D correspondences.

This bridges the perception node (which reports object poses in the CAMERA
frame) and the lerobot executor YAML (which broadcasts a static base->camera
transform via ``camera_static_tf_map``). The output is a ready-to-paste YAML
block in the exact schema consumed by
``camera_publisher._build_static_transform_msg``.

Workflow for a statically-mounted (eye-to-hand) camera:
  1. Launch perception with ``target_frame_id`` set to the *camera* frame so the
     query service returns object poses in the camera frame, e.g.::

       -p target_frame_id:=panda_front_camera

  2. For several physical points, record the SAME point in two frames:
       - CAMERA frame: the translation returned by /perception/query_pose.
       - BASE frame:  the robot tool-tip position (from TF / robot state) when
         the tool touches that point.
     Use 4+ non-coplanar points for a robust fit.

  3. Put the pairs in a JSON file (see ``--input`` format) and run this script.
     Paste the printed ``camera_static_tf_map`` block into the executor YAML and
     restart the skill server so it broadcasts the calibrated TF. Perception then
     lifts poses to base_link and the spatial-prior gate stops abstaining with
     ``frame_mismatch``.

Input JSON format::

    {
      "camera_name": "left",                 # BT camera key in the executor YAML
      "parent_frame_id": "base_link",
      "child_frame_id": "panda_front_camera",
      "correspondences": [
        {"camera": [x, y, z], "base": [X, Y, Z]},
        ...
      ]
    }

The solver is a rigid Kabsch/Umeyama alignment (rotation + translation, no
scaling). It DOES NOT invent calibration: it only fits the pairs you provide and
reports the residual RMS so you can judge whether the fit is trustworthy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def solve_rigid_transform(
    camera: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (R, t, rms) such that base ~= R @ camera + t (no scaling)."""
    if camera.shape != base.shape or camera.shape[0] < 3 or camera.shape[1] != 3:
        raise ValueError("Need matching Nx3 arrays with N >= 3 correspondences.")

    mu_cam = camera.mean(axis=0)
    mu_base = base.mean(axis=0)
    cam_c = camera - mu_cam
    base_c = base - mu_base

    covariance = cam_c.T @ base_c
    u, _, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = vt.T @ correction @ u.T
    translation = mu_base - rotation @ mu_cam

    transformed = (rotation @ camera.T).T + translation
    rms = float(np.sqrt(np.mean(np.sum((transformed - base) ** 2, axis=1))))
    return rotation, translation, rms


def rotation_matrix_to_quaternion_xyzw(matrix: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to a normalized [x, y, z, w] quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm > 0.0:
        quat = quat / norm
    return [float(v) for v in quat]


def _format_yaml_block(
    camera_name: str,
    parent_frame_id: str,
    child_frame_id: str,
    translation: np.ndarray,
    quaternion_xyzw: list[float],
) -> str:
    tx, ty, tz = (float(translation[0]), float(translation[1]), float(translation[2]))
    qx, qy, qz, qw = quaternion_xyzw
    return (
        "camera_static_tf_map:\n"
        f"  {camera_name}:\n"
        f"    parent_frame_id: {parent_frame_id}\n"
        f"    child_frame_id: {child_frame_id}\n"
        f"    translation: [{tx:.6f}, {ty:.6f}, {tz:.6f}]\n"
        f"    rotation_xyzw: [{qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}]\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Path to the correspondences JSON file.")
    parser.add_argument("--output", default="", help="Optional path to write the YAML block.")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).expanduser().read_text(encoding="utf-8"))
    camera_name = str(payload.get("camera_name", "left"))
    parent_frame_id = str(payload.get("parent_frame_id", "base_link"))
    child_frame_id = str(payload.get("child_frame_id", "panda_front_camera"))
    pairs = payload.get("correspondences", [])
    if not isinstance(pairs, list) or len(pairs) < 3:
        raise ValueError("'correspondences' must be a list with at least 3 entries.")

    camera = np.array([p["camera"] for p in pairs], dtype=np.float64)
    base = np.array([p["base"] for p in pairs], dtype=np.float64)

    rotation, translation, rms = solve_rigid_transform(camera, base)
    quaternion = rotation_matrix_to_quaternion_xyzw(rotation)
    block = _format_yaml_block(
        camera_name, parent_frame_id, child_frame_id, translation, quaternion
    )

    print(f"[calib] fit RMS residual = {rms * 1000.0:.2f} mm over {len(pairs)} points")
    if rms > 0.02:
        print(
            "[calib] WARNING: residual > 20 mm. Re-check your correspondences; "
            "do NOT enforce the gate with an untrustworthy calibration."
        )
    print(block)
    if args.output:
        Path(args.output).expanduser().write_text(block, encoding="utf-8")
        print(f"[calib] wrote YAML block -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
