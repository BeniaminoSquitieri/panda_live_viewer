import numpy as np
from perception.contracts import ObjectEstimate, ObservationStatus, enrich_fact_contract
from perception.fusion import fuse_object_facts
from perception.synchronization import ApproximateRgbdSynchronizer
from perception.tracking import TemporalObjectTracker


def _covariance(xyz_variance: float) -> list[float]:
    covariance = np.eye(6, dtype=float)
    covariance[:3, :3] *= xyz_variance
    return covariance.reshape(-1).tolist()


def _fact(x: float, confidence: float, camera: str, variance: float = 0.01) -> dict:
    return enrich_fact_contract(
        {
            "name": "cup",
            "present": True,
            "frame_id": "base_link",
            "stamp": 1.0,
            "pose": {
                "translation": {"x": x, "y": 0.0, "z": 0.2},
                "quaternion_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "covariance": _covariance(variance),
            "pose_confidence": confidence,
            "warnings": [],
        },
        source_camera=camera,
    )


def test_approximate_rgbd_synchronizer_selects_nearest_pair():
    sync = ApproximateRgbdSynchronizer(max_delta_s=0.05)
    assert sync.add_rgb(stamp=1.0, value="rgb-old", frame_id="cam") is None
    assert sync.add_rgb(stamp=1.1, value="rgb-near", frame_id="cam") is None
    pair = sync.add_depth(stamp=1.12, value="depth", frame_id="cam")
    assert pair is not None
    assert pair.rgb == "rgb-near"
    assert pair.depth == "depth"


def test_covariance_weighted_fusion_prefers_precise_camera():
    precise = _fact(1.0, 0.8, "front", variance=0.001)
    noisy = _fact(2.0, 0.7, "wrist", variance=0.1)
    fused = fuse_object_facts([precise, noisy])
    x = fused["pose"]["translation"]["x"]
    assert 1.0 < x < 1.02
    assert fused["source_cameras"] == ["front", "wrist"]
    assert "fused_multi_camera" in fused["warnings"]


def test_tracker_requires_confirmation_and_preserves_track_id():
    tracker = TemporalObjectTracker(min_confirmations=2, alpha=0.5)
    first = tracker.update({"cup": _fact(1.0, 0.8, "front")}, stamp=1.0)["cup"]
    second = tracker.update({"cup": _fact(1.02, 0.8, "front")}, stamp=1.1)["cup"]
    assert first["observation_status"] == ObservationStatus.TENTATIVE
    assert second["observation_status"] == ObservationStatus.DETECTED_WITH_POSE
    assert first["track_id"] == second["track_id"]
    assert second["pose"]["translation"]["x"] == 1.01


def test_tracker_marks_short_dropout_temporarily_lost():
    tracker = TemporalObjectTracker(min_confirmations=1, max_lost_age_s=1.0)
    detected = tracker.update({"cup": _fact(1.0, 0.8, "front")}, stamp=1.0)["cup"]
    missing = tracker.update(
        {
            "cup": {
                "name": "cup",
                "present": False,
                "frame_id": "base_link",
                "stamp": 1.2,
                "warnings": [],
            }
        },
        stamp=1.2,
    )["cup"]
    assert missing["observation_status"] == ObservationStatus.TEMPORARILY_LOST
    assert missing["track_id"] == detected["track_id"]
    assert "pose" not in missing


def test_object_estimate_contract_round_trip():
    estimate = ObjectEstimate.from_fact(_fact(1.0, 0.8, "front"))
    assert estimate.object_id == "cup"
    assert estimate.status == ObservationStatus.DETECTED_WITH_POSE
    assert estimate.source_cameras == ("front",)
