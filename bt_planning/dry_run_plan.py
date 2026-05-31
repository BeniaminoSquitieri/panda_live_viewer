__all__ = ["make_dummy_plan", "extract_names"]
"""
Dry-run/mock planner for Linear IR JSON plan generation.
- Reads planner_registry_json
- Produces a valid-looking Linear IR JSON using only names from registry
- Does NOT call VLM
- Does NOT generate XML
- For debug/testing only
"""
import argparse
import json
from pathlib import Path

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

def make_dummy_plan(task_name, registry):
    robot_skills = extract_names(registry.get('robot_skills', []))
    human_steps = extract_names(registry.get('human_steps', []))
    vlm_gates = extract_names(registry.get('vlm_gates', []))

    if task_name == "make_sandwich":
        required = {
            'vlm_gate': ["initial_scene_ready", "ingredient_poured", "second_toast_ready", "make_sandwich.task_complete"],
            'robot_skill': ["place_first_toast", "place_second_toast"],
            'human_step': ["pour_ingredient"]
        }
        missing = []
        for k, names in required.items():
            if k == 'vlm_gate':
                for n in names:
                    if n not in vlm_gates:
                        missing.append(f"vlm_gate:{n}")
            elif k == 'robot_skill':
                for n in names:
                    if n not in robot_skills:
                        missing.append(f"robot_skill:{n}")
            elif k == 'human_step':
                for n in names:
                    if n not in human_steps:
                        missing.append(f"human_step:{n}")
        if missing:
            raise RuntimeError(f"Missing required registry entries for make_sandwich: {', '.join(missing)}")
        steps = [
            {"kind": "vlm_gate", "name": "initial_scene_ready"},
            {"kind": "robot_skill", "name": "place_first_toast"},
            {"kind": "human_step", "name": "pour_ingredient"},
            {"kind": "vlm_gate", "name": "ingredient_poured"},
            {"kind": "vlm_gate", "name": "second_toast_ready"},
            {"kind": "robot_skill", "name": "place_second_toast"},
            {"kind": "vlm_gate", "name": "make_sandwich.task_complete"}
        ]
        return {
            'task_name': task_name,
            'steps': steps
        }
    else:
        raise RuntimeError(f"dry-run deterministic plan not implemented for task: {task_name}")

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
