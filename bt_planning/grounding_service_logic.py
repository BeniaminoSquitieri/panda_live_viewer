"""Pure request handling for the GroundInstruction ROS service.

This is only the transport boundary: it builds the grounding prompt, calls the
VLM backend, and returns the RAW, UNPARSED response. It never parses,
canonicalizes, or validates — lerobot (bt_generation/) remains the sole safety
boundary, exactly as for the GenerateTaskPlan service.
"""

import json
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from bt_planning.grounding_prompt_builder import build_grounding_prompt


GroundingBackend = Callable[[str], str]


@dataclass(frozen=True)
class GroundInstructionResult:
    success: bool
    raw_response: str
    error_message: str
    prompt: str = ""


def _loads_json_object(raw: str, field_name: str, required: bool) -> Dict:
    text = (raw or "").strip()
    if not text:
        if required:
            raise ValueError(f"{field_name} is required.")
        return {}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {field_name}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return payload


def build_ground_instruction_response(
    nl_instruction: str,
    grounding_payload_json: str,
    last_error_json: str = "",
    *,
    vlm_backend: Optional[GroundingBackend] = None,
) -> GroundInstructionResult:
    """Return GroundInstruction fields; lerobot remains the definitive validator."""

    prompt = ""
    try:
        instruction = str(nl_instruction or "").strip()
        if not instruction:
            raise ValueError("nl_instruction is required.")

        grounding_payload = _loads_json_object(
            grounding_payload_json, "grounding_payload_json", required=True
        )
        last_error = _loads_json_object(last_error_json, "last_error_json", required=False)

        prompt = build_grounding_prompt(instruction, grounding_payload, last_error or None)

        if vlm_backend is None:
            raise RuntimeError("Grounding VLM backend is not configured.")
        raw_response = vlm_backend(prompt)

        # Transport boundary only: no parsing/validation here. Return the raw VLM
        # text unchanged for robot-side parse + strict validation.
        return GroundInstructionResult(
            success=True,
            raw_response=raw_response,
            error_message="",
            prompt=prompt,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        return GroundInstructionResult(
            success=False,
            raw_response="",
            error_message=f"{exc}\n---\n{tb}",
            prompt=prompt,
        )
