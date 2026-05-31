# bt_planning/

Helpers for prompt building and plan parsing for the visual planner (panda_live_viewer).

- `prompt_builder.py`: Builds prompts for the VLM planner. Ensures only Linear IR JSON is requested, using only names from the provided registry, and treats `canonical_task_sequence` as the authoritative order when present.
- `plan_parser.py`: Lightweight parser for Linear IR JSON. Rejects XML/prose, checks for required fields, does not replace lerobot validation.
- `dry_run_plan.py`: Dry-run planner that returns the exact `canonical_task_sequence` after checking every `kind`/`name` pair against the allowed registry vocabulary. Does not call VLM or generate XML.
- `service_logic.py`: Pure GenerateTaskPlan request handling shared by the ROS node and manual checks.

## Responsibilities

- panda_live_viewer proposes Linear IR JSON plans only.
- lerobot is responsible for validation, XML/YAML generation, and execution.
- lerobot supplies `canonical_task_sequence`; panda_live_viewer must not reorder, add, or remove those steps.
- Do not add XML/BT generation here.
- Do not duplicate the full registry manually.
