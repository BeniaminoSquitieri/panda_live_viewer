"""Scene facts scaffolding for future VLM scene extraction."""

from typing import Any, Dict, Mapping, Optional


def _camera_frame(available: bool, stamp: Optional[Any]) -> Dict[str, Any]:
    frame = {"available": bool(available)}
    if stamp is not None:
        frame["stamp"] = stamp
    return frame


def build_scene_facts_stub(
    front_available: bool,
    wrist_available: bool,
    timestamps: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the stable scene facts envelope without deriving facts yet."""
    timestamps = timestamps or {}
    return {
        "schema_version": 1,
        "camera_frames": {
            "front": _camera_frame(front_available, timestamps.get("front")),
            "wrist": _camera_frame(wrist_available, timestamps.get("wrist")),
        },
        "facts": {},
        "warnings": [],
    }
