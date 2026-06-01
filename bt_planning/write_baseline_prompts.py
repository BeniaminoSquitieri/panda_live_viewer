"""CLI to write baseline planner prompts for offline ablation studies.

Generates three prompt variants for a task from a planner registry JSON file:
- constrained Linear IR (safe; same family as the ROS service prompt);
- unconstrained Linear IR (offline baseline);
- direct XML (UNSAFE offline baseline; not a robot execution path).

This CLI only writes prompt text files. It never calls a model or the robot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bt_planning.baseline_prompts import (
    build_constrained_linear_ir_prompt,
    build_direct_xml_prompt,
    build_unconstrained_linear_ir_prompt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write baseline planner prompts for offline ablation.",
    )
    parser.add_argument("--task", required=True, help="Task name.")
    parser.add_argument(
        "--planner-registry",
        required=True,
        type=Path,
        help="Path to planner_registry JSON (as produced by lerobot).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory to write the prompt .txt files into.",
    )
    args = parser.parse_args(argv)

    registry = json.loads(args.planner_registry.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        parser.error("planner-registry JSON must be an object.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"{args.task}_constrained_linear_ir_prompt.txt": build_constrained_linear_ir_prompt(
            args.task, registry
        ),
        f"{args.task}_unconstrained_linear_ir_prompt.txt": build_unconstrained_linear_ir_prompt(
            args.task, registry
        ),
        f"{args.task}_direct_xml_prompt.txt": build_direct_xml_prompt(args.task, registry),
    }

    for filename, content in outputs.items():
        path = args.out_dir / filename
        path.write_text(content + "\n", encoding="utf-8")
        print(f"wrote prompt: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
