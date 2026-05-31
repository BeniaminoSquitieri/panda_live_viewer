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
import os
import random
from pathlib import Path

def load_registry(path):
    with open(path, 'r') as f:
        return json.load(f)

def make_dummy_plan(task_name, registry):
    robot_skills = registry.get('robot_skills', [])
    human_steps = registry.get('human_steps', [])
    vlm_gates = registry.get('vlm_gates', [])
    steps = []
    # Add at least one of each type if available
    if robot_skills:
        steps.append({'type': 'robot_skill', 'name': random.choice(robot_skills)})
    if human_steps:
        steps.append({'type': 'human_step', 'name': random.choice(human_steps)})
    if vlm_gates:
        steps.append({'type': 'vlm_gate', 'name': random.choice(vlm_gates)})
    # Fill up to 3-5 steps
    all_names = robot_skills + human_steps + vlm_gates
    for _ in range(3, random.randint(4, 6)):
        if not all_names:
            break
        name = random.choice(all_names)
        if name in robot_skills:
            steps.append({'type': 'robot_skill', 'name': name})
        elif name in human_steps:
            steps.append({'type': 'human_step', 'name': name})
        else:
            steps.append({'type': 'vlm_gate', 'name': name})
    return {
        'task_name': task_name,
        'steps': steps
    }

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
    with open(out_path, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"Dry-run plan written to {out_path}")

if __name__ == '__main__':
    main()
