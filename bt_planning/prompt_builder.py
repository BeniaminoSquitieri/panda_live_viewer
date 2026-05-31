"""
Helper to build prompts for the VLM planner.

- Returns prompts for Linear IR JSON only.
- Does NOT request XML, prose, or skills not in registry.
"""
import json
from typing import Optional


def _json_block(value) -> str:
    return json.dumps(value, indent=2)


def build_planner_prompt(task_name: str, planner_registry_json: dict, scene_facts_json: Optional[dict] = None) -> str:
    """
    Build a prompt for the VLM planner.
    - Only use names from planner_registry_json
    - Use robot_skill only for listed robot_skills
    - Use human_step only for listed human_steps
    - Use vlm_gate only for listed vlm_gates
    - Return JSON only. Do not return XML or prose.
    - Do not invent skill names.
    """
    registry_str = _json_block(planner_registry_json)
    scene_facts_str = _json_block(scene_facts_json) if scene_facts_json else None
    canonical_task_sequence = planner_registry_json.get("canonical_task_sequence")
    ordering_constraints = planner_registry_json.get("ordering_constraints")

    prompt = [
        f"Task: {task_name}",
        "---",
        "Required registry fields:",
        f"task_name: {task_name}",
        "robot_skills:",
        _json_block(planner_registry_json.get("robot_skills", [])),
        "human_steps:",
        _json_block(planner_registry_json.get("human_steps", [])),
        "vlm_gates:",
        _json_block(planner_registry_json.get("vlm_gates", [])),
    ]

    if "objects" in planner_registry_json:
        prompt += ["objects:", _json_block(planner_registry_json.get("objects"))]
    if "aliases" in planner_registry_json:
        prompt += ["aliases:", _json_block(planner_registry_json.get("aliases"))]
    if "object_aliases" in planner_registry_json:
        prompt += ["object_aliases:", _json_block(planner_registry_json.get("object_aliases"))]

    if canonical_task_sequence is not None:
        prompt += [
            "---",
            "canonical_task_sequence (authoritative BT leaf order):",
            _json_block(canonical_task_sequence),
        ]

    if ordering_constraints is not None:
        prompt += [
            "---",
            "ordering_constraints (must be respected):",
            _json_block(ordering_constraints),
        ]

    prompt += [
        "---",
        "Registry (JSON):",
        registry_str,
    ]
    if scene_facts_str:
        prompt += ["---", "Scene facts (JSON):", scene_facts_str]

    prompt += [
        "---",
        "Instructions:",
        "- Return Linear IR JSON only.",
        "- Return a valid JSON plan only.",
        "- Do NOT return XML.",
        "- Do NOT include prose.",
        "- Do NOT invent skill names.",
        "- Use only names from the registry above.",
        "- Use robot_skill only for listed robot_skills.",
        "- Use human_step only for listed human_steps.",
        "- Use vlm_gate only for listed vlm_gates."
    ]

    if canonical_task_sequence is not None:
        prompt += [
            "- canonical_task_sequence is the authoritative BT leaf order.",
            "- Follow canonical_task_sequence exactly.",
            "- Do not reorder steps.",
            "- Do not add steps.",
            "- Do not remove steps.",
            "- Use only kind/name pairs from canonical_task_sequence.",
            "- Do not infer order from object names.",
        ]

    if ordering_constraints is not None:
        prompt += [
            "- Respect ordering_constraints exactly; they are additional ordering requirements from lerobot.",
        ]

    return "\n".join(prompt)
