import numpy as np

from perception.geometry import (
    CameraIntrinsics,
    back_project_mask,
    quaternion_multiply_xyzw,
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

