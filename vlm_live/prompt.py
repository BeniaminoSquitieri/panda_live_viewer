"""Prompt construction and status normalization helpers."""

import json
import re
from typing import Any

from .const import (
    DEFAULT_ALLOWED_STATUSES,
    STATUS_FAILURE,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_WAIT_HUMAN,
    SUPPORTED_STATUSES,
)

STATUS_PATTERN = "|".join((STATUS_RUNNING, STATUS_SUCCESS, STATUS_FAILURE, STATUS_WAIT_HUMAN))


def format_scene_context(scene_facts_json: str) -> str:
    """Render perception scene facts as a concise, read-only prompt context.

    One-way perception -> VLM enrichment: the VLM may *read* metric object poses
    to ground its judgement, but it never produces coordinates. Returns an empty
    string when there are no usable facts, so the prompt is unchanged (no
    regression) whenever perception is absent.
    """
    raw = (scene_facts_json or "").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    facts = payload.get("facts")
    if not isinstance(facts, dict) or not facts:
        return ""

    lines: list[str] = []
    for name, fact in sorted(facts.items()):
        if not isinstance(fact, dict) or not fact.get("present"):
            continue
        frame_id = str(fact.get("frame_id") or "camera")
        pose = fact.get("pose")
        if isinstance(pose, dict) and isinstance(pose.get("translation"), dict):
            t = pose["translation"]
            try:
                coords = f"[{float(t['x']):.3f}, {float(t['y']):.3f}, {float(t['z']):.3f}] m"
            except (KeyError, TypeError, ValueError):
                coords = "(pose unavailable)"
            confidence = fact.get("pose_confidence")
            conf_text = ""
            try:
                if confidence is not None:
                    conf_text = f", confidence {float(confidence):.2f}"
            except (TypeError, ValueError):
                conf_text = ""
            lines.append(f"- {name}: {coords} in {frame_id}{conf_text}")
        else:
            lines.append(f"- {name}: present (no metric pose)")

    return "\n".join(lines)


def build_prompt(request: dict[str, Any], reasoning: bool) -> str:
    """Build the textual instruction for the VLM."""
    message = request.get("message") or "No additional context."
    task_text = request.get("task") or ""
    scene_context = (request.get("scene_context") or "").strip()
    allowed_statuses = _allowed_statuses_for_prompt(request)
    camera_view = str(request.get("camera_view") or "front").strip().lower()
    status_spec = "|".join(allowed_statuses)
    status_text = ", ".join(allowed_statuses)
    wait_human_instruction = "Choose WAIT_HUMAN when explicit human intervention, a human action, or a human decision is required.\n" if STATUS_WAIT_HUMAN in allowed_statuses else ""
    task_line = f"Task description: {task_text}\n" if task_text else ""
    scene_context_block = (
        "Perception scene facts (metric object poses measured by the depth pipeline; use only as spatial context, do not invent coordinates):\n"
        f"{scene_context}\n"
        if scene_context
        else ""
    )
    if camera_view == "both":
        camera_instruction = (
            "The image shows two camera views side by side: the LEFT half is the fixed 'Front camera' and the RIGHT half is the moving 'Wrist camera' mounted on the gripper.\n"
            "The two cameras see the same workspace from different angles, so each object is usually visible in only one view. Consider an object PRESENT if it is visible in AT LEAST ONE of the two views; do NOT require it to appear in both.\n"
            "Inspect EACH half of the image separately and exhaustively before concluding anything is missing: first scan the LEFT (Front) half, then scan the RIGHT (Wrist) half. An object counts as present if it appears in EITHER scan.\n"
            "Some objects are easy to miss: cups and mugs may be transparent, white, reflective, empty, small, tilted, partially occluded by other objects, or only partially inside the frame (e.g. at an edge or in the gripper). Do NOT report such an object as absent unless you have carefully searched BOTH halves and still cannot find it.\n"
        )
    else:
        camera_instruction = (
            "The image shows only the fixed Front camera view. Base your decision exclusively on this Front camera image.\n"
            "Do not mention, require, infer, or wait for any additional camera view. If the requested condition is not visible in the Front camera, judge only from the available Front camera evidence.\n"
            "Some objects are easy to miss: cups and mugs may be transparent, white, reflective, empty, small, tilted, partially occluded by other objects, or only partially inside the frame. Do NOT report such an object as absent unless you have carefully searched the Front camera image.\n"
        )
    common_header = (
        "You are an object-state verifier. Inspect the live camera scene and decide whether the requested object condition is satisfied.\n"
        "Focus on task-relevant objects, their presence, and their final locations. Do not use robot arm pose, gripper contact, grasping, lifting, or motion as evidence unless the request explicitly asks about the robot itself.\n"
        f"{camera_instruction}"
        f"Check name: {request['skill_name']}\n"
        f"Attempt id: {request['attempt_id']}\n"
        f"{task_line}"
        f"{scene_context_block}"
        f"BT message: {message}\n"
    )
    decision_rules = (
        wait_human_instruction
        + "Choose SUCCESS only when the condition is fully satisfied.\n"
        + "Choose RUNNING when evidence is insufficient or the condition is not yet satisfied.\n"
        + "Choose FAILURE only for an explicit, irreversible wrong outcome. If an object is unclear, occluded, hard to see, or temporarily not visible, choose RUNNING."
    )

    if reasoning:
        return (
            common_header
            + "\nReturn two fields only:\n"
            + f"STATUS=<{status_spec}>\n"
            + "REASON=<short explanation; may span multiple lines>\n"
            + decision_rules
        )

    return common_header + f"\nReturn exactly one token: {status_text}.\n" + decision_rules


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


def fit_status(status: str, allowed_statuses: list[str]) -> str:
    """Map the model verdict into one of the requested allowed statuses."""
    fallback = allowed_statuses[0] if allowed_statuses else STATUS_RUNNING
    return next((value for value in (status, STATUS_RUNNING, STATUS_FAILURE, STATUS_SUCCESS, STATUS_WAIT_HUMAN) if value in allowed_statuses), fallback)


def _allowed_statuses_for_prompt(request: dict[str, Any]) -> list[str]:
    raw_statuses = request.get("allowed_statuses")
    if not isinstance(raw_statuses, list):
        return DEFAULT_ALLOWED_STATUSES.copy()
    normalized = {str(status).strip().upper() for status in raw_statuses}
    allowed = [status for status in SUPPORTED_STATUSES if status in normalized]
    return allowed or DEFAULT_ALLOWED_STATUSES.copy()


def parse_verdict(text: str) -> tuple[str, str]:
    """Return a normalized status and reason."""
    return parse_status(text), extract_reason(text) if text else "Empty model output."


def log_response(text: str, logger) -> None:
    """Log a normalized model response."""
    status, reason = parse_status(text), extract_reason(text)
    logger.info(f"VLM response: STATUS={status}" + (f" REASON={reason}" if reason else ""))
