"""Append-only verifier event logging for the VLM node.

Records one JSONL event per completed VLM verification attempt for offline
analysis of the runtime BT experiments. This logging is provenance only: it does
not change the VLM result topic payload, does not alter status decisions, and
never saves camera images. When the log path is empty, logging is disabled.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1
EVENT_TYPE_VERIFIER = "verifier"


def build_verifier_event(
    *,
    skill_name: str,
    attempt_id,
    allowed_statuses,
    raw_status: str,
    published_status: str,
    reason: str,
    was_wait_human_coerced: bool,
    front_frame_available: bool,
    wrist_frame_available: bool,
    duration_s: Optional[float] = None,
    model_path: str = "",
    dry_run_planner: bool = False,
    error_message: Optional[str] = None,
    timestamp_unix_s: Optional[float] = None,
) -> "OrderedDict[str, object]":
    """Build a verifier event dict with a stable field order."""

    event: "OrderedDict[str, object]" = OrderedDict()
    event["schema_version"] = SCHEMA_VERSION
    event["event_type"] = EVENT_TYPE_VERIFIER
    event["timestamp_unix_s"] = time.time() if timestamp_unix_s is None else timestamp_unix_s
    event["skill_name"] = skill_name
    event["attempt_id"] = attempt_id
    event["allowed_statuses"] = list(allowed_statuses) if allowed_statuses else []
    event["raw_status"] = raw_status
    event["published_status"] = published_status
    event["reason"] = reason
    event["was_wait_human_coerced"] = bool(was_wait_human_coerced)
    event["front_frame_available"] = bool(front_frame_available)
    event["wrist_frame_available"] = bool(wrist_frame_available)
    event["duration_s"] = duration_s
    event["model_path"] = model_path
    event["dry_run_planner"] = bool(dry_run_planner)
    event["error_message"] = error_message
    return event


def append_verifier_event(path, event) -> None:
    """Append a single verifier event as one JSON line (append-only)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
