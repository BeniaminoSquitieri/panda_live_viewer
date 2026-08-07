"""Protocol parsing and payload helpers for VLM request/result ROS messages."""

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from std_msgs.msg import String

from .const import (
    DEFAULT_ALLOWED_STATUSES,
    STATUS_RUNNING,
    STATUS_WAIT_HUMAN,
    SUPPORTED_STATUSES,
)
from .contracts import semantic_contract_for_status
from .protocol_spec import load_protocol_spec

PROTOCOL_SCHEMA_VERSION = int(load_protocol_spec()["schema_version"])


def _normalize_statuses(value: Any) -> list[str]:
    """Normalize a list-like value into supported uppercase status tokens."""
    if not isinstance(value, list):
        return DEFAULT_ALLOWED_STATUSES.copy()
    cleaned = list(dict.fromkeys(token for item in value if (token := str(item).strip().upper()) in SUPPORTED_STATUSES))
    return cleaned or DEFAULT_ALLOWED_STATUSES.copy()


def clean_statuses(value: Any) -> list[str]:
    """Return a normalized list of accepted status values."""
    return _normalize_statuses(value)


def parse_request(msg: "String", logger) -> dict[str, Any] | None:
    """Validate and normalize the incoming JSON request message."""
    raw = msg.data.strip()
    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring malformed JSON on VLM request topic")
        return None

    if not isinstance(payload, dict):
        logger.warning("Ignoring VLM request that is not a JSON object")
        return None

    raw_version = payload.get("protocol_schema_version")
    if raw_version is not None:
        try:
            version = int(raw_version)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring VLM request with invalid protocol schema {raw_version!r}")
            return None
        if version != PROTOCOL_SCHEMA_VERSION:
            logger.warning(f"Ignoring VLM protocol schema {version}; expected {PROTOCOL_SCHEMA_VERSION}")
            return None

    skill_name = str(payload.get("skill_name", "")).strip()
    if not skill_name:
        logger.warning("Ignoring VLM request with empty skill_name")
        return None

    attempt_raw = payload.get("attempt_id", 0)
    try:
        attempt_id = int(attempt_raw)
    except (TypeError, ValueError):
        attempt_id = 0
    if attempt_id < 0:
        attempt_id = 0

    message = str(payload.get("message", "")).strip()
    task_text = payload.get("task")
    task_text = str(task_text).strip() if task_text is not None else ""

    check_period_s = None
    check_period_raw = payload.get("check_period_s")
    if check_period_raw is not None:
        try:
            check_period_s = max(0.0, float(check_period_raw))
        except (TypeError, ValueError):
            check_period_s = None

    return {
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "skill_name": skill_name,
        "attempt_id": attempt_id,
        "message": message,
        "task": task_text,
        "allowed_statuses": clean_statuses(payload.get("allowed_statuses")),
        "check_period_s": check_period_s,
    }


def build_result_payload(request: dict[str, Any], status: str, reason: str) -> dict[str, Any] | None:
    """Build an outgoing VLM status payload from a normalized request."""
    skill_name = request.get("skill_name", "").strip()
    if not skill_name:
        return None
    status, reason = coerce_result_for_request(request, status, reason)
    semantic_status, control_action = semantic_contract_for_status(status)

    payload = {
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "skill_name": skill_name,
        "attempt_id": request.get("attempt_id", 0),
        "status": status,
        "semantic_status": str(semantic_status),
        "control_action": str(control_action),
    }
    if reason:
        payload["message"] = reason
    return payload


def coerce_result_for_request(request: dict[str, Any], status: str, reason: str) -> tuple[str, str]:
    """Map WAIT_HUMAN to RUNNING with a prefixed reason unless explicitly allowed."""
    allowed = clean_statuses(request.get("allowed_statuses", DEFAULT_ALLOWED_STATUSES))
    return (STATUS_RUNNING, _wait_human_reason(reason)) if status == STATUS_WAIT_HUMAN and status not in allowed else (status, reason)


def _wait_human_reason(reason: str) -> str:
    text = str(reason or "").strip()
    return f"WAIT_HUMAN: {text or 'human intervention requested'}"
