import numpy as np

from perception.geometry import CameraIntrinsics
from perception.pipeline import Detection, PerceptionPipeline


class FixedSegmenter:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, rgb):
        return self._detections


def test_pipeline_maps_registry_label_and_estimates_centroid_pose():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    mask = np.ones((2, 2), dtype=bool)
    pipeline = PerceptionPipeline(
        segmenter=FixedSegmenter([Detection(label="toast_slice", mask=mask, score=0.8)]),
        min_depth_points=1,
    )

    result = pipeline.run(
        rgb=rgb,
        depth=depth,
        intrinsics=CameraIntrinsics(width=2, height=2, fx=1.0, fy=1.0, cx=0.0, cy=0.0),
        registry={
            "objects": [
                {
                    "canonical_name": "toast",
                    "aliases": ["toast_slice"],
                }
            ]
        },
        frame_id="front_camera",
        stamp=12.5,
    )

    fact = result.facts["toast"]
    assert fact["present"] is True
    assert fact["frame_id"] == "front_camera"
    assert fact["pose"]["translation"] == {"x": 0.5, "y": 0.5, "z": 1.0}
    assert 0.0 < fact["pose_confidence"] <= 0.8
    assert len(fact["covariance"]) == 36
    assert "orientation_estimated_pca" in fact["warnings"]
    assert "pose_residual_m" in fact
    assert "inlier_ratio" in fact


def test_pipeline_fails_closed_without_segmenter():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    result = PerceptionPipeline().run(
        rgb=rgb,
        depth=None,
        intrinsics=None,
        registry={"objects": ["toast"]},
        frame_id="front_camera",
        stamp=1.0,
    )

    assert result.facts == {}
    assert "segmenter_no_detections" in result.warnings
