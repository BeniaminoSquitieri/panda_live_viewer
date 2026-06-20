"""Append-only perception JSONL logging."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EVENT_TYPE_PERCEPTION = "perception"


def build_perception_event(
    *,
    camera_name: str,
    frame_id: str,
    stamp: Any,
    facts: Mapping[str, Any],
    warnings: list[str],
    detections_seen: int,
    latency_s: float | None = None,
) -> OrderedDict[str, object]:
    event: OrderedDict[str, object] = OrderedDict()
    event["schema_version"] = SCHEMA_VERSION
    event["event_type"] = EVENT_TYPE_PERCEPTION
    event["timestamp_unix_s"] = time.time()
    event["camera_name"] = camera_name
    event["frame_id"] = frame_id
    event["stamp"] = stamp
    event["detections_seen"] = int(detections_seen)
    event["object_names"] = sorted(facts)
    event["object_estimates"] = {
        name: {
            "observation_status": fact.get("observation_status"),
            "pose_confidence": fact.get("pose_confidence"),
            "track_id": fact.get("track_id"),
            "confirmation_count": fact.get("confirmation_count"),
            "source_cameras": fact.get("source_cameras", []),
            "calibration_id": fact.get("calibration_id"),
        }
        for name, fact in sorted(facts.items())
    }
    event["warnings"] = list(warnings)
    event["latency_s"] = latency_s
    return event


def append_perception_event(path, event: Mapping[str, Any]) -> None:
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
