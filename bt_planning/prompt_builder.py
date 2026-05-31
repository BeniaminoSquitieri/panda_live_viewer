"""
Helper to build prompts for the VLM planner.

- Returns prompts for Linear IR JSON only.
- Does NOT request XML, prose, or skills not in registry.
"""
import json
from typing import Optional

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
    registry_str = json.dumps(planner_registry_json, indent=2)
    scene_facts_str = json.dumps(scene_facts_json, indent=2) if scene_facts_json else None
    prompt = [
        f"Task: {task_name}",
        "---",
        "Registry (JSON):",
        registry_str,
    ]
    if scene_facts_str:
        prompt += ["---", "Scene facts (JSON):", scene_facts_str]
    prompt += [
        "---",
        "Instructions:",
        "- Return a valid JSON plan only.",
        "- Do NOT return XML.",
        "- Do NOT include prose.",
        "- Do NOT invent skill names.",
        "- Use only names from the registry above.",
        "- Use robot_skill only for listed robot_skills.",
        "- Use human_step only for listed human_steps.",
        "- Use vlm_gate only for listed vlm_gates."
    ]
    return "\n".join(prompt)
