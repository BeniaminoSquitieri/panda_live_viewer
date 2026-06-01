"""Baseline prompt builders for offline planner ablation studies.

These builders exist to produce *measurable* baselines for a paper. Only the
constrained Linear IR prompt is the safe runtime path used by the ROS service.

Safety notes:
- `build_constrained_linear_ir_prompt` is the safe prompt (reuses the runtime
  prompt builder). It is the only one used by the ROS service.
- `build_unconstrained_linear_ir_prompt` drops the canonical-sequence ordering
  constraints. It is an OFFLINE baseline only.
- `build_direct_xml_prompt` asks the model for executable XML directly. It is an
  UNSAFE OFFLINE-ONLY baseline and must never be used by the ROS service or any
  robot execution path.
"""

from __future__ import annotations

import json
from typing import Optional

from bt_planning.prompt_builder import build_planner_prompt

# Marker string that flags the direct-XML prompt as an unsafe offline baseline.
BASELINE_UNSAFE_MARKER = "BASELINE_UNSAFE_OFFLINE_ONLY"


def _json_block(value) -> str:
    return json.dumps(value, indent=2)


def build_constrained_linear_ir_prompt(
    task_name: str,
    registry: dict,
    scene_facts: Optional[dict] = None,
) -> str:
    """Safe constrained prompt. Reuses the runtime Linear IR prompt builder."""

    return build_planner_prompt(task_name, registry, scene_facts)


def build_unconstrained_linear_ir_prompt(
    task_name: str,
    registry: dict,
    scene_facts: Optional[dict] = None,
) -> str:
    """OFFLINE baseline: Linear IR JSON without canonical-sequence ordering.

    Includes the registry vocabulary but intentionally omits
    canonical_task_sequence and ordering_constraints, so the model must choose
    an order itself. Still forbids XML and prose.
    """

    prompt = [
        f"Task: {task_name}",
        "(OFFLINE BASELINE: unconstrained Linear IR. Not a robot execution path.)",
        "---",
        "Required registry fields:",
        f"task_name: {task_name}",
        "robot_skills:",
        _json_block(registry.get("robot_skills", [])),
        "human_steps:",
        _json_block(registry.get("human_steps", [])),
        "vlm_gates:",
        _json_block(registry.get("vlm_gates", [])),
    ]

    if "objects" in registry:
        prompt += ["objects:", _json_block(registry.get("objects"))]

    if scene_facts:
        prompt += ["---", "Scene facts (JSON):", _json_block(scene_facts)]

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
        "- Use vlm_gate only for listed vlm_gates.",
        "- Choose a sensible order yourself; no canonical sequence is provided.",
    ]
    return "\n".join(prompt)


def build_direct_xml_prompt(
    task_name: str,
    registry: dict,
    scene_facts: Optional[dict] = None,
) -> str:
    """UNSAFE OFFLINE baseline: ask the model for BehaviorTree.CPP XML directly.

    This prompt is for offline measurement only. The generated XML must never be
    executed and is never used by the safe ROS service.
    """

    prompt = [
        f"{BASELINE_UNSAFE_MARKER}",
        "This prompt is an UNSAFE offline baseline. The output must NOT be executed",
        "on a robot and is NOT used by the safe ROS planning service.",
        "---",
        f"Task: {task_name}",
        "Produce a BehaviorTree.CPP v3 XML behavior tree for this task.",
        "---",
        "Available vocabulary (for reference only):",
        "robot_skills:",
        _json_block(registry.get("robot_skills", [])),
        "human_steps:",
        _json_block(registry.get("human_steps", [])),
        "vlm_gates:",
        _json_block(registry.get("vlm_gates", [])),
    ]

    if "objects" in registry:
        prompt += ["objects:", _json_block(registry.get("objects"))]

    if scene_facts:
        prompt += ["---", "Scene facts (JSON):", _json_block(scene_facts)]

    prompt += [
        "---",
        "Instructions:",
        "- Return a single <root> ... </root> BehaviorTree.CPP XML document.",
        "- Do NOT return JSON.",
        "- This is an offline baseline for measurement only.",
        f"- {BASELINE_UNSAFE_MARKER}: never run this output on a robot.",
    ]
    return "\n".join(prompt)
