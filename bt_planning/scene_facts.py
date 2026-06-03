"""Scene facts helpers shared by planning and perception."""

from typing import Any, Dict, Iterable, Mapping, Optional


SCHEMA_VERSION = 1


def _camera_frame(
    available: bool,
    stamp: Optional[Any],
    *,
    frame_id: Optional[str] = None,
    depth_available: Optional[bool] = None,
    camera_info_available: Optional[bool] = None,
    tf_available: Optional[bool] = None,
) -> Dict[str, Any]:
    frame = {"available": bool(available)}
    if stamp is not None:
        frame["stamp"] = stamp
    if frame_id:
        frame["frame_id"] = frame_id
    if depth_available is not None:
        frame["depth_available"] = bool(depth_available)
    if camera_info_available is not None:
        frame["camera_info_available"] = bool(camera_info_available)
    if tf_available is not None:
        frame["tf_available"] = bool(tf_available)
    return frame


def build_object_pose_fact(
    *,
    name: str,
    present: bool,
    frame_id: str,
    stamp: Any,
    translation: Mapping[str, float] | None = None,
    quaternion_xyzw: Mapping[str, float] | None = None,
    pose_confidence: float = 0.0,
    covariance: Iterable[float] | None = None,
    seg_score: float | None = None,
    pose_residual_m: float | None = None,
    inlier_ratio: float | None = None,
    warnings: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """Build one per-object scene fact with explicit uncertainty fields."""
    fact: Dict[str, Any] = {
        "name": name,
        "present": bool(present),
        "frame_id": frame_id,
        "stamp": stamp,
        "pose_confidence": max(0.0, min(float(pose_confidence), 1.0)),
        "warnings": list(warnings or []),
    }
    if translation is not None and quaternion_xyzw is not None:
        fact["pose"] = {
            "translation": {key: float(value) for key, value in translation.items()},
            "quaternion_xyzw": {key: float(value) for key, value in quaternion_xyzw.items()},
        }
    if covariance is not None:
        fact["covariance"] = [float(value) for value in covariance]
    if seg_score is not None:
        fact["seg_score"] = float(seg_score)
    if pose_residual_m is not None:
        fact["pose_residual_m"] = float(pose_residual_m)
    if inlier_ratio is not None:
        fact["inlier_ratio"] = float(inlier_ratio)
    return fact


def build_scene_facts_stub(
    front_available: bool,
    wrist_available: bool,
    timestamps: Optional[Mapping[str, Any]] = None,
    *,
    camera_details: Optional[Mapping[str, Mapping[str, Any]]] = None,
    facts: Optional[Mapping[str, Any]] = None,
    warnings: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return a stable envelope for future scene extraction."""
    timestamps = timestamps or {}
    camera_details = camera_details or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "camera_frames": {
            "front": _camera_frame(
                front_available,
                timestamps.get("front"),
                **dict(camera_details.get("front", {})),
            ),
            "wrist": _camera_frame(
                wrist_available,
                timestamps.get("wrist"),
                **dict(camera_details.get("wrist", {})),
            ),
        },
        "facts": dict(facts or {}),
        "warnings": list(warnings or []),
    }
