"""Prompt construction and status normalization helpers."""

import re
from typing import Any, Dict, List

from .const import (
    STATUS_FAILURE,
    STATUS_MANUAL_INTERVENTION_REQUIRED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_WAIT_HUMAN,
)

def build_prompt(request: Dict[str, Any], reasoning: bool) -> str:
    """Build the textual instruction for the VLM."""
    message = request.get("message") or "No additional context."
    task_text = request.get("task") or ""
    task_line = f"Task description: {task_text}\n" if task_text else ""
    common_header = (
        "You are a robotic task verifier. Inspect the live camera scene and decide whether the requested condition is satisfied.\n"
        f"Check name: {request['skill_name']}\n"
        f"Attempt id: {request['attempt_id']}\n"
        f"{task_line}"
        f"BT message: {message}\n"
    )

    if reasoning:
        return (
            common_header
            + "\nReturn two fields only:\n"
            + "STATUS=<PENDING|RUNNING|WAIT_HUMAN|MANUAL_INTERVENTION_REQUIRED|SUCCESS|FAILURE>\n"
            + "REASON=<short explanation; may span multiple lines>\n"
            + "Choose PENDING or RUNNING when the condition is not yet satisfied or evidence is insufficient.\n"
            + "Choose FAILURE when the condition is clearly not satisfied in this attempt.\n"
            + "Choose SUCCESS only when the condition is fully satisfied."
        )

    return (
        common_header
        + "\nReturn exactly one token: PENDING, RUNNING, WAIT_HUMAN, MANUAL_INTERVENTION_REQUIRED, SUCCESS, or FAILURE.\n"
        + "Choose PENDING or RUNNING when the condition is not yet satisfied or evidence is insufficient.\n"
        + "Choose FAILURE when the condition is clearly not satisfied in this attempt.\n"
        + "Choose SUCCESS only when the condition is fully satisfied."
    )


def extract_reason(text: str) -> str:
    """Return only the explanatory part of the model output."""
    stripped = text.strip()
    if not stripped:
        return ""

    match = re.search(r"REASON\s*[:=]\s*(.*)", stripped, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    status_pattern = (
        r"PENDING|RUNNING|WAIT_HUMAN|MANUAL_INTERVENTION_REQUIRED|SUCCESS|FAILURE"
    )
    without_status = re.sub(
        rf"^\s*STATUS\s*[:=]\s*(?:{status_pattern})\s*",
        "",
        stripped,
        flags=re.IGNORECASE,
    ).strip()
    if without_status:
        return without_status

    if re.fullmatch(status_pattern, stripped, flags=re.IGNORECASE):
        return ""

    return stripped


def parse_status(text: str) -> str:
    """Normalize the raw model output into a supported status value."""
    upper = text.upper()
    match = re.search(
        r"STATUS\s*[:=]\s*(PENDING|RUNNING|WAIT_HUMAN|MANUAL_INTERVENTION_REQUIRED|SUCCESS|FAILURE)",
        upper,
    )
    if match:
        return match.group(1)
    if STATUS_SUCCESS in upper:
        return STATUS_SUCCESS
    if STATUS_FAILURE in upper or "FAILED" in upper:
        return STATUS_FAILURE
    if STATUS_MANUAL_INTERVENTION_REQUIRED in upper or "MANUAL_INTERVENTION" in upper:
        return STATUS_MANUAL_INTERVENTION_REQUIRED
    if STATUS_WAIT_HUMAN in upper or "WAIT HUMAN" in upper or "HUMAN_HELP" in upper:
        return STATUS_WAIT_HUMAN
    if STATUS_PENDING in upper:
        return STATUS_PENDING
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
    if STATUS_MANUAL_INTERVENTION_REQUIRED in allowed_statuses:
        return STATUS_MANUAL_INTERVENTION_REQUIRED
    if STATUS_PENDING in allowed_statuses:
        return STATUS_PENDING
    return STATUS_RUNNING
