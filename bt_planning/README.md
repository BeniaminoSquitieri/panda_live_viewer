# bt_planning/

Helpers for prompt building and plan parsing for the visual planner (panda_live_viewer).

- `prompt_builder.py`: Builds prompts for the VLM planner. Ensures only Linear IR JSON is requested, using only names from the provided registry.
- `plan_parser.py`: Lightweight parser for Linear IR JSON. Rejects XML/prose, checks for required fields, does not replace lerobot validation.
- `dry_run_plan.py`: Dry-run/mock planner for generating valid-looking Linear IR JSON using only names from the registry. Does not call VLM or generate XML.

## Responsibilities

- panda_live_viewer proposes Linear IR JSON plans only.
- lerobot is responsible for validation, XML/YAML generation, and execution.
- Do not add XML/BT generation here.
- Do not duplicate the full registry manually.
