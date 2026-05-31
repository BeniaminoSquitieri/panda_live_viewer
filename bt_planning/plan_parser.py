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
    if "task_name" not in plan:
        raise PlanParseError("Missing 'task_name' in plan.")
    if "steps" not in plan or not isinstance(plan["steps"], list):
        raise PlanParseError("Missing or invalid 'steps' in plan.")
    return plan
