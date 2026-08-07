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
    duration_s: float | None = None,
    model_path: str = "",
    dry_run_planner: bool = False,
    error_message: str | None = None,
    timestamp_unix_s: float | None = None,
) -> OrderedDict[str, object]:
    """Build a verifier event dict with a stable field order."""
    return OrderedDict(
        schema_version=SCHEMA_VERSION,
        event_type=EVENT_TYPE_VERIFIER,
        timestamp_unix_s=time.time() if timestamp_unix_s is None else timestamp_unix_s,
        skill_name=skill_name,
        attempt_id=attempt_id,
        allowed_statuses=list(allowed_statuses) if allowed_statuses else [],
        raw_status=raw_status,
        published_status=published_status,
        reason=reason,
        was_wait_human_coerced=bool(was_wait_human_coerced),
        front_frame_available=bool(front_frame_available),
        wrist_frame_available=bool(wrist_frame_available),
        duration_s=duration_s,
        model_path=model_path,
        dry_run_planner=bool(dry_run_planner),
        error_message=error_message,
    )


def append_verifier_event(path, event) -> None:
    """Append a single verifier event as one JSON line (append-only)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
