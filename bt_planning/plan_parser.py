"""
Lightweight parser for Linear IR JSON plans.
- Rejects XML-like or prose responses.
- Checks for required fields: task_name, steps.
- Does NOT replace lerobot strict validation.
"""

import json
import re
from typing import Any, Dict, Iterable

ALLOWED_STEP_KINDS = frozenset(("robot_skill", "human_step", "vlm_gate"))
FORBIDDEN_FIELDS = frozenset(
    (
        "raw_xml",
        "xml",
        "action",
        "condition",
        "explanation",
        "free_text",
        "fallback",
        "parallel",
        "human_fallback",
    )
)

FENCED_BLOCK_RE = re.compile(r"\A```\s*([^\n`]*)?\n(.*?)\n?```\s*\Z", re.DOTALL)
XML_LIKE_RE = re.compile(
    r"<!--.*?-->|<!DOCTYPE\b[^>]*>|<\?xml\b[^>]*\?>|"
    r"</?\s*[A-Za-z_][\w:.-]*(?:\s+[^<>]*)?/?>",
    re.IGNORECASE | re.DOTALL,
)


class PlanParseError(Exception):
    pass


def _json_payload_from_response(response: str) -> str:
    if not isinstance(response, str):
        raise PlanParseError("Response must be a string.")

    text = response.strip()
    if not text:
        raise PlanParseError("Response is empty.")

    if XML_LIKE_RE.search(text):
        raise PlanParseError("Response contains XML-like content, not Linear IR JSON.")

    if text.startswith("```") or text.endswith("```"):
        match = FENCED_BLOCK_RE.fullmatch(text)
        if match is None:
            raise PlanParseError("Fenced responses must contain exactly one JSON block and no prose.")
        language = (match.group(1) or "").strip().lower()
        if language and language != "json":
            raise PlanParseError("Fenced response language must be json.")
        payload = match.group(2).strip()
        if not payload:
            raise PlanParseError("Fenced JSON block is empty.")
        return payload

    return text


def _iter_forbidden_paths(value: Any, path: str = "$") -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_FIELDS:
                yield child_path
            yield from _iter_forbidden_paths(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_forbidden_paths(child, f"{path}[{index}]")


def parse_linear_ir_plan(response: str) -> Dict[str, Any]:
    """
    Parse and lightly validate a Linear IR JSON plan.
    Raises PlanParseError on failure.
    """
    payload = _json_payload_from_response(response)
    try:
        plan = json.loads(payload)
    except json.JSONDecodeError as e:
        raise PlanParseError(f"Invalid JSON: {e}")

    if not isinstance(plan, dict):
        raise PlanParseError("Plan must be a JSON object.")

    forbidden_paths = list(_iter_forbidden_paths(plan))
    if forbidden_paths:
        raise PlanParseError(f"Plan contains forbidden field: {forbidden_paths[0]}.")

    if "task_name" not in plan or not isinstance(plan["task_name"], str) or not plan["task_name"].strip():
        raise PlanParseError("Missing or empty 'task_name' in plan.")

    if "steps" not in plan or not isinstance(plan["steps"], list) or not plan["steps"]:
        raise PlanParseError("Missing or invalid 'steps' in plan.")

    for i, step in enumerate(plan["steps"]):
        if not isinstance(step, dict):
            raise PlanParseError(f"Step {i} is not an object.")
        if "type" in step:
            raise PlanParseError(f"Step {i} uses 'type' instead of 'kind'.")
        if "kind" not in step:
            raise PlanParseError(f"Step {i} missing 'kind'.")
        if "name" not in step:
            raise PlanParseError(f"Step {i} missing 'name'.")
        if not isinstance(step["kind"], str) or step["kind"] not in ALLOWED_STEP_KINDS:
            raise PlanParseError(f"Step {i} has invalid 'kind': {step['kind']}")
        if not isinstance(step["name"], str) or not step["name"].strip():
            raise PlanParseError(f"Step {i} has invalid or empty 'name'.")

    return plan
