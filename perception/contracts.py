"""Typed, ROS-agnostic contracts for metric perception outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ObservationStatus(StrEnum):
    """Quality/state vocabulary owned by metric perception."""

    DETECTED_WITH_POSE = "DETECTED_WITH_POSE"
    DETECTED_NO_POSE = "DETECTED_NO_POSE"
    TENTATIVE = "TENTATIVE"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    NOT_DETECTED = "NOT_DETECTED"
    STALE = "STALE"


@dataclass(frozen=True)
class ObjectEstimate:
    """Normalized metric estimate independent from task semantics."""

    object_id: str
    status: ObservationStatus
    frame_id: str
    stamp: float | None
    pose: Mapping[str, Any] | None
    covariance: tuple[float, ...]
    confidence: float
    track_id: int | None
    confirmation_count: int
    source_cameras: tuple[str, ...]
    warnings: tuple[str, ...]

    @classmethod
    def from_fact(cls, fact: Mapping[str, Any]) -> ObjectEstimate:
        raw_status = str(fact.get("observation_status") or "")
        if not raw_status:
            raw_status = (
                ObservationStatus.DETECTED_WITH_POSE
                if fact.get("pose")
                else ObservationStatus.DETECTED_NO_POSE
                if fact.get("present")
                else ObservationStatus.NOT_DETECTED
            )
        try:
            status = ObservationStatus(raw_status)
        except ValueError:
            status = ObservationStatus.DETECTED_NO_POSE
        stamp = fact.get("stamp")
        try:
            normalized_stamp = None if stamp is None else float(stamp)
        except (TypeError, ValueError):
            normalized_stamp = None
        track_id = fact.get("track_id")
        try:
            normalized_track_id = None if track_id is None else int(track_id)
        except (TypeError, ValueError):
            normalized_track_id = None
        return cls(
            object_id=str(fact.get("name") or ""),
            status=status,
            frame_id=str(fact.get("frame_id") or ""),
            stamp=normalized_stamp,
            pose=fact.get("pose") if isinstance(fact.get("pose"), Mapping) else None,
            covariance=tuple(float(value) for value in fact.get("covariance", []) or []),
            confidence=max(0.0, min(float(fact.get("pose_confidence", 0.0)), 1.0)),
            track_id=normalized_track_id,
            confirmation_count=max(0, int(fact.get("confirmation_count", 0))),
            source_cameras=tuple(str(value) for value in fact.get("source_cameras", []) or []),
            warnings=tuple(str(value) for value in fact.get("warnings", []) or []),
        )


def enrich_fact_contract(
    fact: dict[str, Any],
    *,
    source_camera: str | None = None,
) -> dict[str, Any]:
    """Add stable contract fields while preserving the legacy scene-fact schema."""
    has_pose = isinstance(fact.get("pose"), Mapping)
    present = bool(fact.get("present"))
    fact.setdefault(
        "observation_status",
        ObservationStatus.DETECTED_WITH_POSE
        if has_pose
        else ObservationStatus.DETECTED_NO_POSE
        if present
        else ObservationStatus.NOT_DETECTED,
    )
    fact.setdefault("geometry_quality", float(fact.get("pose_confidence", 0.0)))
    fact.setdefault("confirmation_count", 1 if present else 0)
    fact.setdefault("track_id", None)
    sources = list(fact.get("source_cameras", []) or [])
    if source_camera and source_camera not in sources:
        sources.append(source_camera)
    fact["source_cameras"] = sources
    return fact
