"""Prompt construction and status normalization helpers."""

import re
from typing import Any, Dict, List

from .const import (
    DEFAULT_ALLOWED_STATUSES,
    STATUS_FAILURE,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_WAIT_HUMAN,
    SUPPORTED_STATUSES,
)

STATUS_PATTERN = "|".join((STATUS_RUNNING, STATUS_SUCCESS, STATUS_FAILURE, STATUS_WAIT_HUMAN))

def build_prompt(request: Dict[str, Any], reasoning: bool) -> str:
    """Build the textual instruction for the VLM."""
    message = request.get("message") or "No additional context."
    task_text = request.get("task") or ""
    allowed_statuses = _allowed_statuses_for_prompt(request)
    status_spec = "|".join(allowed_statuses)
    status_text = ", ".join(allowed_statuses)
    wait_human_instruction = ""
    if STATUS_WAIT_HUMAN in allowed_statuses:
        wait_human_instruction = (
            "Choose WAIT_HUMAN when explicit human intervention, a human action, or a human decision is required.\n"
        )
    task_line = f"Task description: {task_text}\n" if task_text else ""
    common_header = (
        "You are a robotic task verifier. Inspect the live camera scene and decide whether the requested condition is satisfied.\n"
        "The image shows two camera views side by side: the LEFT half is the fixed 'Front camera' and the RIGHT half is the moving 'Wrist camera' mounted on the gripper.\n"
        "The two cameras see the same workspace from different angles, so each object is usually visible in only one view. Consider an object PRESENT if it is visible in AT LEAST ONE of the two views; do NOT require it to appear in both.\n"
        f"Check name: {request['skill_name']}\n"
        f"Attempt id: {request['attempt_id']}\n"
        f"{task_line}"
        f"BT message: {message}\n"
    )

    if reasoning:
        return (
            common_header
            + "\nReturn two fields only:\n"
            + f"STATUS=<{status_spec}>\n"
            + "REASON=<short explanation; may span multiple lines>\n"
            + wait_human_instruction
            + "Choose SUCCESS only when the condition is fully satisfied.\n"
            + "Choose RUNNING when evidence is insufficient or the condition is not yet satisfied.\n"
            + "Choose FAILURE when the condition clearly failed and retrying the action is appropriate."
        )

    return (
        common_header
        + f"\nReturn exactly one token: {status_text}.\n"
        + wait_human_instruction
        + "Choose SUCCESS only when the condition is fully satisfied.\n"
        + "Choose RUNNING when evidence is insufficient or the condition is not yet satisfied.\n"
        + "Choose FAILURE when the condition clearly failed and retrying the action is appropriate."
    )


def extract_reason(text: str) -> str:
    """Return only the explanatory part of the model output."""
    stripped = text.strip()
    if not stripped:
        return ""

    match = re.search(r"REASON\s*[:=]\s*(.*)", stripped, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    without_status = re.sub(
        rf"^\s*STATUS\s*[:=]\s*(?:{STATUS_PATTERN})\s*",
        "",
        stripped,
        flags=re.IGNORECASE,
    ).strip()
    if without_status:
        return without_status

    if re.fullmatch(rf"\s*STATUS\s*[:=]\s*(?:{STATUS_PATTERN})\s*", stripped, flags=re.IGNORECASE):
        return ""

    if re.fullmatch(STATUS_PATTERN, stripped, flags=re.IGNORECASE):
        return ""

    return stripped


def parse_status(text: str) -> str:
    """Normalize the raw model output into a supported status value."""
    upper = text.upper()
    match = re.search(
        rf"STATUS\s*[:=]\s*({STATUS_PATTERN})",
        upper,
    )
    if match:
        return match.group(1)
    if STATUS_WAIT_HUMAN in upper or "WAIT HUMAN" in upper:
        return STATUS_WAIT_HUMAN
    if STATUS_SUCCESS in upper:
        return STATUS_SUCCESS
    if STATUS_FAILURE in upper or "FAILED" in upper:
        return STATUS_FAILURE
    if STATUS_RUNNING in upper or "STILL_RUNNING" in upper or "STILL RUNNING" in upper:
        return STATUS_RUNNING
    return STATUS_RUNNING


def fit_status(status: str, allowed_statuses: List[str]) -> str:
    """Map the model verdict into one of the requested allowed statuses."""
    if status in allowed_statuses:
        return status
    if STATUS_RUNNING in allowed_statuses:
        return STATUS_RUNNING
    if STATUS_FAILURE in allowed_statuses:
        return STATUS_FAILURE
    if STATUS_SUCCESS in allowed_statuses:
        return STATUS_SUCCESS
    if STATUS_WAIT_HUMAN in allowed_statuses:
        return STATUS_WAIT_HUMAN
    if allowed_statuses:
        return allowed_statuses[0]
    return STATUS_RUNNING


def _allowed_statuses_for_prompt(request: Dict[str, Any]) -> List[str]:
    raw_statuses = request.get("allowed_statuses")
    if not isinstance(raw_statuses, list):
        return DEFAULT_ALLOWED_STATUSES.copy()
    normalized = {str(status).strip().upper() for status in raw_statuses}
    allowed = [status for status in SUPPORTED_STATUSES if status in normalized]
    return allowed or DEFAULT_ALLOWED_STATUSES.copy()
