
# Panda VLM Live Verifier & Planner (panda_live_viewer)

## Architecture & Roles

**panda_live_viewer** (this repo):
- Visual-side only: observes the real camera scene via ROS.
- Can propose a Linear IR JSON plan at the start of an episode.
- Does NOT generate BehaviorTree.CPP XML.
- Does NOT validate or compile BTs.
- Does NOT execute BTs.
- Does NOT own the registry; it consumes `planner_registry_json` from lerobot.
- May save debug planner outputs under `generated_plans/` (not for live integration).

**lerobot** (external):
- Owns validation, XML/YAML generation, and execution.
- Sends `task_name` + `planner_registry_json` (and optionally `scene_facts_json`) to panda_live_viewer via ROS service `/lerobot_bt/generate_plan`.
- Receives only Linear IR JSON as `plan_json`.
- Validates and compiles plans to XML/YAML.
- Provides `canonical_task_sequence`, the authoritative BT leaf order.

## ROS Planning Service

`/lerobot_bt/generate_plan` is implemented on the VLM node.

- Service type: `lerobot_bt_interfaces/srv/GenerateTaskPlan`
- Request fields: `task_name`, `planner_registry_json`, `scene_facts_json`
- Response fields: `success`, `plan_json`, `error_message`
- The service returns Linear IR JSON only.
- panda_live_viewer does not generate XML/YAML, validate definitively, or execute plans.
- lerobot validates the returned Linear IR and compiles it to XML/YAML.
- `canonical_task_sequence` comes from lerobot and is treated as the authoritative order.
- The VLM prompt requires the model to follow `canonical_task_sequence` exactly.

Dry-run service mode is available with the ROS parameter `planner_dry_run:=true`.
In dry-run mode the service does not call the VLM; it returns `canonical_task_sequence`
from `planner_registry_json` as `plan_json` after checking each `kind`/`name` pair
against the allowed `robot_skills`, `human_steps`, and `vlm_gates` vocabularies.
With the default `lazy_load_model:=true`, this dry-run service can answer
`/lerobot_bt/generate_plan` without loading Qwen or touching the GPU.

The ROS parameter `require_generate_plan_service:=true` fails startup if
`lerobot_bt_interfaces/srv/GenerateTaskPlan` is unavailable. Fix the ROS
environment with:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash && source install/setup.bash
```

Set `require_generate_plan_service:=false` only when intentionally running the
verifier without the planning service.

## Planner Output Path

- Debug outputs (raw model responses, Linear IR JSON) may be saved under `generated_plans/`.
- Do NOT use shared files for live integration; use the ROS service when available.
- Do NOT commit generated plan artifacts unless intentionally adding examples.


## Linear IR Step Format

- Each step in a Linear IR plan may contain only `kind`, `name`, `object`, and `objects`.
- `kind` is one of: `robot_skill`, `human_step`, `vlm_gate`.
- `type` is not valid for Linear IR steps.

See `bt_planning/` for helpers to build prompts and parse Linear IR JSON plans.

## Verifier Role (During Execution)

- During BT execution, lerobot asks panda_live_viewer for condition verification only.
- panda_live_viewer returns only statuses allowed by each request.
- Current lerobot requests allow `SUCCESS`/`FAILURE`/`RUNNING`; they do not support `WAIT_HUMAN` end-to-end.
- If the VLM proposes `WAIT_HUMAN` without request support, panda_live_viewer publishes `RUNNING` and prefixes the message with `WAIT_HUMAN: `.
- Plan JSON is NOT a status result.
- STATUS/REASON is NOT a plan.
- Do NOT reuse `/lerobot_bt/vlm_result` for plans.
- Planner and verifier are separate flows: planning happens once before BT execution, verification happens during execution.

## Tests & Checks

- If tests exist, run: `python -m pytest -svv`
- If no tests, run: `python -m compileall .`


## Dry-run Planner

- The dry-run planner (`bt_planning/dry_run_plan.py`) is deterministic and for debug/integration testing.
- It emits the exact `canonical_task_sequence` from the registry and never invents an order.
- It fails clearly when `canonical_task_sequence` is missing or references names outside the allowed vocabulary.

## Folder Structure

- `generated_plans/`: Debug planner outputs only
  - `raw_model_responses/`
  - `linear_ir/`
- `bt_planning/`: Prompt and parser helpers

## Definition of Done

- panda_live_viewer returns Linear IR JSON only.
- lerobot validates and compiles to XML/YAML.
- No misleading XML/BT ownership claims.
- Debug artifacts go under `generated_plans/`.
- No BehaviorTree.CPP XML generation here.
- Verifier behavior is not broken.
- Tests or compile checks pass.


# (Verifier usage and install/run instructions follow below)
