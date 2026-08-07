from types import SimpleNamespace

import numpy as np
import pytest

from perception.geometry import (
    CameraIntrinsics,
    back_project_mask,
    decode_depth_image,
    estimate_pose_pca,
    quaternion_multiply_xyzw,
    rotate_covariance_6x6_xyzw,
    rotate_vector_xyzw,
)


def _depth_msg(data, *, encoding="16UC1", width=2, height=1, step=4, big_endian=False):
    return SimpleNamespace(encoding=encoding, width=width, height=height, step=step, is_bigendian=big_endian, data=data)


def test_decode_depth_image_handles_row_padding():
    data = np.array([1, 2], dtype="<u2").tobytes() + b"\xaa\xbb" + np.array([3, 4], dtype="<u2").tobytes() + b"\xcc\xdd"
    np.testing.assert_array_equal(decode_depth_image(_depth_msg(data, height=2, step=6)), [[1, 2], [3, 4]])


def test_decode_depth_image_handles_big_endian_values():
    expected = np.array([[1.25, 2.5]], dtype=np.float32)
    msg = _depth_msg(expected.astype(">f4").tobytes(), encoding="32FC1", step=8, big_endian=True)
    np.testing.assert_allclose(decode_depth_image(msg), expected)


def test_decode_depth_image_rejects_short_rows():
    with pytest.raises(ValueError, match="step"):
        decode_depth_image(_depth_msg(b"\x00\x01", step=2))


def test_back_project_mask_uses_intrinsics_and_depth_scale():
    depth = np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)
    mask = np.array([[True, True], [False, True]])
    intrinsics = CameraIntrinsics(width=2, height=2, fx=1.0, fy=1.0, cx=0.0, cy=0.0)

    points = back_project_mask(depth=depth, mask=mask, intrinsics=intrinsics)

    np.testing.assert_allclose(
        points,
        np.array(
            [
                [0.0, 0.0, 1.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        ),
    )


def test_back_project_mask_rejects_mask_shape_mismatch():
    depth = np.ones((2, 2), dtype=np.uint16)
    mask = np.ones((2, 3), dtype=bool)
    intrinsics = CameraIntrinsics(width=2, height=2, fx=1.0, fy=1.0, cx=0.0, cy=0.0)

    with pytest.raises(ValueError, match="must match mask shape"):
        back_project_mask(depth=depth, mask=mask, intrinsics=intrinsics)


def test_back_project_mask_rejects_intrinsics_shape_mismatch():
    depth = np.ones((2, 2), dtype=np.uint16)
    mask = np.ones((2, 2), dtype=bool)
    intrinsics = CameraIntrinsics(width=3, height=2, fx=1.0, fy=1.0, cx=0.0, cy=0.0)

    with pytest.raises(ValueError, match="does not match intrinsics"):
        back_project_mask(depth=depth, mask=mask, intrinsics=intrinsics)


def test_quaternion_rotate_vector_identity():
    vector = {"x": 1.0, "y": 2.0, "z": 3.0}
    identity = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

    assert rotate_vector_xyzw(vector, identity) == vector
    assert quaternion_multiply_xyzw(identity, identity) == identity


def test_estimate_pose_pca_recovers_primary_axis_orientation():
    points = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    pose = estimate_pose_pca(points)

    assert pose.translation == {"x": 0.5, "y": 0.0, "z": 0.0}
    rotated_x = rotate_vector_xyzw({"x": 1.0, "y": 0.0, "z": 0.0}, pose.quaternion_xyzw)
    np.testing.assert_allclose(
        [rotated_x["x"], rotated_x["y"], rotated_x["z"]],
        [1.0, 0.0, 0.0],
        atol=1e-6,
    )
    assert len(pose.covariance) == 36
    assert pose.inlier_ratio == 1.0
    assert "pca_secondary_axis_ambiguous" in pose.warnings


def test_rotate_covariance_6x6_preserves_shape_and_rotates_blocks():
    covariance = [0.0] * 36
    covariance[0] = 1.0
    covariance[7] = 2.0
    covariance[14] = 3.0
    covariance[21] = 4.0
    covariance[28] = 5.0
    covariance[35] = 6.0
    identity = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

    rotated = rotate_covariance_6x6_xyzw(covariance, identity)

    assert rotated == covariance
