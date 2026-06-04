"""Pure request handling for the GenerateTaskPlan ROS service."""

import json
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from bt_planning.dry_run_plan import make_dummy_plan
from bt_planning.plan_parser import parse_linear_ir_plan
from bt_planning.prompt_builder import build_planner_prompt


PlannerBackend = Callable[[str], str]


@dataclass(frozen=True)
class GeneratePlanResult:
    success: bool
    plan_json: str
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


def _step_signature(step: Dict, index: int, source: str) -> Tuple[str, str]:
    if not isinstance(step, dict):
        raise ValueError(f"{source} step {index} must be an object.")

    kind = step.get("kind")
    name = step.get("name")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{source} step {index} missing valid kind.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source} step {index} missing valid name.")
    return kind, name


def _sequence_signature(steps: Iterable[Dict], source: str) -> List[Tuple[str, str]]:
    return [_step_signature(step, index, source) for index, step in enumerate(steps)]


def _check_preliminary_plan_constraints(plan: Dict, task_name: str, registry: Dict) -> None:
    """Catch obvious service-boundary mistakes before lerobot validates fully."""
    if plan["task_name"] != task_name:
        raise ValueError(
            f"Plan task_name {plan['task_name']!r} does not match request task_name {task_name!r}."
        )

    canonical_task_sequence = registry.get("canonical_task_sequence")
    if canonical_task_sequence is None:
        return
    if not isinstance(canonical_task_sequence, list):
        raise ValueError("planner_registry_json canonical_task_sequence must be a list.")

    plan_sequence = _sequence_signature(plan["steps"], "plan")
    canonical_sequence = _sequence_signature(canonical_task_sequence, "canonical_task_sequence")
    if plan_sequence != canonical_sequence:
        raise ValueError(
            "Plan steps do not match planner_registry_json canonical_task_sequence exactly: "
            f"plan={plan_sequence!r}, canonical={canonical_sequence!r}."
        )


def build_generate_plan_response(
    task_name: str,
    planner_registry_json: str,
    scene_facts_json: str = "",
    *,
    dry_run: bool,
    vlm_backend: Optional[PlannerBackend] = None,
) -> GeneratePlanResult:
    """Return GenerateTaskPlan fields; lerobot remains the definitive validator."""
    prompt = ""
    try:
        normalized_task_name = str(task_name or "").strip()
        if not normalized_task_name:
            raise ValueError("task_name is required.")

        registry = _loads_json_object(planner_registry_json, "planner_registry_json", required=True)
        scene_facts = _loads_json_object(scene_facts_json, "scene_facts_json", required=False)
        prompt = build_planner_prompt(normalized_task_name, registry, scene_facts or None)

        if dry_run:
            plan = make_dummy_plan(normalized_task_name, registry)
        else:
            if vlm_backend is None:
                raise RuntimeError("VLM planner backend is not configured.")
            raw_plan = vlm_backend(prompt)
            plan = parse_linear_ir_plan(raw_plan)

        plan_json = json.dumps(plan, indent=2)
        parse_linear_ir_plan(plan_json)
        _check_preliminary_plan_constraints(plan, normalized_task_name, registry)
        return GeneratePlanResult(
            success=True,
            plan_json=plan_json,
            error_message="",
            prompt=prompt,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        return GeneratePlanResult(
            success=False,
            plan_json="",
            error_message=f"{exc}\n---\n{tb}",
            prompt=prompt,
        )
