"""
Lightweight parser for Linear IR JSON plans.
- Rejects XML-like or prose responses.
- Checks for required fields: task_name, steps.
- Does NOT replace lerobot strict validation.
"""
import json
from typing import Any, Dict

class PlanParseError(Exception):
    pass

def parse_linear_ir_plan(response: str) -> Dict[str, Any]:
    """
    Parse and lightly validate a Linear IR JSON plan.
    Raises PlanParseError on failure.
    """
    if response.strip().startswith("<"):
        raise PlanParseError("Response looks like XML, not JSON.")
    try:
        plan = json.loads(response)
    except Exception as e:
        raise PlanParseError(f"Invalid JSON: {e}")
    if not isinstance(plan, dict):
        raise PlanParseError("Plan must be a JSON object.")
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
        if not isinstance(step["kind"], str) or step["kind"] not in ("robot_skill", "human_step", "vlm_gate"):
            raise PlanParseError(f"Step {i} has invalid 'kind': {step['kind']}")
        if not isinstance(step["name"], str) or not step["name"].strip():
            raise PlanParseError(f"Step {i} has invalid or empty 'name'.")
        if "raw_xml" in step or "xml" in step:
            raise PlanParseError(f"Step {i} contains forbidden XML field.")
    return plan
