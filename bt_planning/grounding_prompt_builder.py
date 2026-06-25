"""Prompt builder for the NL instruction-grounding VLM.

Unlike the planner prompt, this prompt intentionally does NOT contain the
canonical task sequence (the payload from build_grounding_payload withholds it):
the model must reconstruct task_name + steps by reasoning about the instruction
and the capability menu. Output is Linear IR JSON only, or an error_message when
no known task matches.
"""

import json
from typing import Optional


def _json_block(value) -> str:
    return json.dumps(value, indent=2)


def build_grounding_prompt(
    nl_instruction: str,
    grounding_payload: dict,
    last_error: Optional[dict] = None,
) -> str:
    """Build the grounding prompt; repair feedback (if any) follows the rules.

    The output-format rules stay the last thing the model reads before answering
    (the repair block ends by restating the format constraint), which maximises
    format compliance on retries.
    """

    available_tasks = grounding_payload.get("available_tasks", [])
    prompt = [
        "You ground a natural-language instruction into a Behavior Tree plan.",
        "Instruction:",
        nl_instruction,
        "---",
        "Available tasks (choose the one that matches the instruction):",
        _json_block(available_tasks),
        "Available robot_skills:",
        _json_block(grounding_payload.get("robot_skills", [])),
        "Available human_steps:",
        _json_block(grounding_payload.get("human_steps", [])),
        "Available vlm_gates:",
        _json_block(grounding_payload.get("vlm_gates", [])),
    ]

    if grounding_payload.get("allowed_objects"):
        prompt += ["Allowed objects:", _json_block(grounding_payload.get("allowed_objects"))]

    prompt += [
        "---",
        "Output format (JSON only, no prose, no XML, no markdown fences):",
        _json_block(
            {
                "task_name": "<one of the available tasks>",
                "steps": [{"kind": "<robot_skill|human_step|vlm_gate>", "name": "<registered name>"}],
            }
        ),
        "---",
        "Rules:",
        "- Reconstruct the full plan (task_name and the ordered steps) yourself.",
        "- Return Linear IR JSON only.",
        "- Do NOT return XML.",
        "- Do NOT include prose or markdown.",
        "- Use only names from the registry above.",
        "- Use robot_skill only for listed robot_skills.",
        "- Use human_step only for listed human_steps.",
        "- Use vlm_gate only for listed vlm_gates.",
        "- Do not add fields beyond kind and name to a step.",
        '- If no available task matches the instruction, return {"error_message": "..."} '
        "listing the available task names instead of steps.",
    ]

    # Repair feedback is placed AFTER the rules, and its closing line restates the
    # format constraint so the rules remain the last thing the model reads.
    if last_error:
        stage = last_error.get("stage", "unknown")
        reason = last_error.get("reason", "")
        prompt += [
            "---",
            "Your previous attempt was rejected. Correct exactly this problem:",
            f"- stage: {stage}",
            f"- reason: {reason}",
            "Return the full corrected plan.",
            "Return ONLY the JSON object (task_name + steps). No prose. No XML.",
        ]

    return "\n".join(prompt)
