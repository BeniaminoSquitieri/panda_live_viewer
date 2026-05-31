"""
Dry-run/mock planner for Linear IR JSON plan generation.
- Reads planner_registry_json
- Returns canonical_task_sequence exactly when it is valid
- Does NOT call VLM
- Does NOT generate XML
- For debug/testing only
"""
import argparse
import json
from pathlib import Path

__all__ = ["make_dummy_plan", "make_canonical_plan", "extract_names"]


def extract_names(entries):
    names = []
    for entry in entries or []:
        if isinstance(entry, str):
            name = entry.strip()
        elif isinstance(entry, dict):
            name = str(entry.get("name", "")).strip()
        else:
            continue
        if name:
            names.append(name)
    return names


def load_registry(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _allowed_vocabulary(registry):
    return {
        "robot_skill": set(extract_names(registry.get("robot_skills", []))),
        "human_step": set(extract_names(registry.get("human_steps", []))),
        "vlm_gate": set(extract_names(registry.get("vlm_gates", []))),
    }


def _validate_canonical_step(step, index, allowed):
    if not isinstance(step, dict):
        raise RuntimeError(f"canonical_task_sequence step {index} must be an object.")

    if "kind" not in step:
        raise RuntimeError(f"canonical_task_sequence step {index} missing kind.")
    if "name" not in step:
        raise RuntimeError(f"canonical_task_sequence step {index} missing name.")
    if "type" in step:
        raise RuntimeError(f"canonical_task_sequence step {index} uses type instead of kind.")

    kind = step["kind"]
    name = step["name"]
    if not isinstance(kind, str) or kind not in allowed:
        raise RuntimeError(f"canonical_task_sequence step {index} has invalid kind: {kind!r}.")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError(f"canonical_task_sequence step {index} has invalid name: {name!r}.")
    if name not in allowed[kind]:
        raise RuntimeError(
            f"canonical_task_sequence step {index} ({kind}:{name}) is not listed "
            f"in the allowed {kind} vocabulary."
        )


def make_canonical_plan(task_name, registry):
    canonical_task_sequence = registry.get("canonical_task_sequence")
    if canonical_task_sequence is None:
        raise RuntimeError(
            "planner_registry_json missing canonical_task_sequence; "
            "dry-run mode will not infer task order."
        )
    if not isinstance(canonical_task_sequence, list) or not canonical_task_sequence:
        raise RuntimeError("canonical_task_sequence must be a non-empty list.")

    allowed = _allowed_vocabulary(registry)
    for index, step in enumerate(canonical_task_sequence):
        _validate_canonical_step(step, index, allowed)

    return {
        "task_name": task_name,
        "steps": canonical_task_sequence,
    }


def make_dummy_plan(task_name, registry):
    return make_canonical_plan(task_name, registry)


def main():
    parser = argparse.ArgumentParser(description="Dry-run Linear IR planner (mock, does not call VLM)")
    parser.add_argument('--task', required=True, help='Task name')
    parser.add_argument('--planner-registry', required=True, help='Path to planner_registry JSON')
    parser.add_argument('--out-plan', required=True, help='Output path for Linear IR JSON plan')
    args = parser.parse_args()

    registry = load_registry(args.planner_registry)
    plan = make_dummy_plan(args.task, registry)

    out_path = Path(args.out_plan)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Dry-run plan written to {out_path}")


if __name__ == '__main__':
    main()
