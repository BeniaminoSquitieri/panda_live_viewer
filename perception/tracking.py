"""Temporal filtering and stable track identifiers for metric object facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import ObservationStatus, enrich_fact_contract


def _position(fact: Mapping[str, Any]) -> np.ndarray | None:
    try:
        raw = fact["pose"]["translation"]
        value = np.array([raw["x"], raw["y"], raw["z"]], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.all(np.isfinite(value)) else None


@dataclass
class _Track:
    track_id: int
    frame_id: str
    position: np.ndarray
    last_stamp: float
    confirmations: int


class TemporalObjectTracker:
    """Per-canonical-object EMA tracker with explicit tentative/lost states."""

    def __init__(
        self,
        *,
        alpha: float = 0.65,
        association_distance_m: float = 0.15,
        min_confirmations: int = 2,
        max_lost_age_s: float = 1.0,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = float(alpha)
        self.association_distance_m = float(association_distance_m)
        self.min_confirmations = max(1, int(min_confirmations))
        self.max_lost_age_s = max(0.0, float(max_lost_age_s))
        self._tracks: dict[str, _Track] = {}
        self._next_track_id = 1

    def update(self, facts: Mapping[str, dict[str, Any]], *, stamp: float) -> dict[str, Any]:
        output: dict[str, Any] = {}
        names = set(facts) | set(self._tracks)
        for name in names:
            fact = dict(facts.get(name, {}))
            position = _position(fact)
            track = self._tracks.get(name)
            if position is None:
                if track is not None and stamp - track.last_stamp <= self.max_lost_age_s:
                    fact.setdefault("name", name)
                    fact["present"] = False
                    fact.pop("pose", None)
                    fact.pop("covariance", None)
                    fact["track_id"] = track.track_id
                    fact["confirmation_count"] = track.confirmations
                    fact["observation_status"] = ObservationStatus.TEMPORARILY_LOST
                    fact.setdefault("warnings", []).append("temporarily_lost")
                    output[name] = enrich_fact_contract(fact)
                elif fact:
                    if track is not None:
                        self._tracks.pop(name, None)
                    output[name] = enrich_fact_contract(fact)
                continue

            frame_id = str(fact.get("frame_id") or "")
            if (
                track is None
                or track.frame_id != frame_id
                or float(np.linalg.norm(position - track.position)) > self.association_distance_m
            ):
                track = _Track(
                    track_id=self._next_track_id,
                    frame_id=frame_id,
                    position=position,
                    last_stamp=stamp,
                    confirmations=1,
                )
                self._next_track_id += 1
                self._tracks[name] = track
            else:
                track.position = self.alpha * position + (1.0 - self.alpha) * track.position
                track.last_stamp = stamp
                track.confirmations += 1

            fact["pose"] = dict(fact["pose"])
            fact["pose"]["translation"] = {
                "x": float(track.position[0]),
                "y": float(track.position[1]),
                "z": float(track.position[2]),
            }
            fact["track_id"] = track.track_id
            fact["confirmation_count"] = track.confirmations
            fact["observation_status"] = (
                ObservationStatus.DETECTED_WITH_POSE
                if track.confirmations >= self.min_confirmations
                else ObservationStatus.TENTATIVE
            )
            if track.confirmations < self.min_confirmations:
                fact.setdefault("warnings", []).append("track_tentative")
            output[name] = enrich_fact_contract(fact)
        return output
