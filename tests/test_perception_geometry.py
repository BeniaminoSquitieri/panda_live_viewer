import numpy as np

from perception.geometry import (
    CameraIntrinsics,
    back_project_mask,
    estimate_pose_pca,
    quaternion_multiply_xyzw,
    rotate_covariance_6x6_xyzw,
    rotate_vector_xyzw,
)


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
