"""Protocol parsing and payload helpers for VLM request/result ROS messages."""

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from std_msgs.msg import String

from .const import (
    DEFAULT_ALLOWED_STATUSES,
    STATUS_RUNNING,
    STATUS_WAIT_HUMAN,
    SUPPORTED_STATUSES,
)


def _normalize_statuses(value: Any) -> List[str]:
    """Normalize a list-like value into supported uppercase status tokens."""
    if not isinstance(value, list):
        return DEFAULT_ALLOWED_STATUSES.copy()

    allowed_set = set(SUPPORTED_STATUSES)
    cleaned: List[str] = []
    for item in value:
        token = str(item).strip().upper()
        if not token or token in cleaned:
            continue
        if token not in allowed_set:
            continue
        cleaned.append(token)
    return cleaned or DEFAULT_ALLOWED_STATUSES.copy()


def clean_statuses(value: Any) -> List[str]:
    """Return a normalized list of accepted status values."""
    return _normalize_statuses(value)


def parse_request(msg: "String", logger) -> Optional[Dict[str, Any]]:
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

    return {
        "skill_name": skill_name,
        "attempt_id": attempt_id,
        "message": message,
        "task": task_text,
        "allowed_statuses": clean_statuses(payload.get("allowed_statuses")),
    }


def build_result_payload(request: Dict[str, Any], status: str, reason: str) -> Optional[Dict[str, Any]]:
    """Build an outgoing VLM status payload from a normalized request."""
    skill_name = request.get("skill_name", "").strip()
    if not skill_name:
        return None
    status, reason = coerce_result_for_request(request, status, reason)

    payload = {
        "skill_name": skill_name,
        "attempt_id": request.get("attempt_id", 0),
        "status": status,
    }
    if reason:
        payload["message"] = reason
    return payload


def coerce_result_for_request(request: Dict[str, Any], status: str, reason: str) -> tuple[str, str]:
    """Map WAIT_HUMAN to RUNNING with a prefixed reason unless explicitly allowed."""
    allowed_statuses = request.get("allowed_statuses", DEFAULT_ALLOWED_STATUSES)
    allowed_statuses = clean_statuses(allowed_statuses)
    if status == STATUS_WAIT_HUMAN and STATUS_WAIT_HUMAN not in allowed_statuses:
        return STATUS_RUNNING, _wait_human_reason(reason)
    return status, reason


def _wait_human_reason(reason: str) -> str:
    text = str(reason or "").strip()
    if text:
        return f"WAIT_HUMAN: {text}"
    return "WAIT_HUMAN: human intervention requested"
