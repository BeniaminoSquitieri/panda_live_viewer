
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
- Sends `task_name` + `planner_registry_json` (and optionally `scene_facts_json`) to panda_live_viewer via ROS service `/lerobot_bt/generate_plan` (future integration).
- Receives only Linear IR JSON as `plan_json`.
- Validates and compiles plans to XML/YAML.

## Planner Output Path

- Debug outputs (raw model responses, Linear IR JSON) may be saved under `generated_plans/`.
- Do NOT use shared files for live integration; use the ROS service when available.
- Do NOT commit generated plan artifacts unless intentionally adding examples.


## Linear IR Step Format

- Each step in a Linear IR plan must have exactly two fields: `kind` and `name`.
- `kind` is one of: `robot_skill`, `human_step`, `vlm_gate`.
- `type` is not valid for Linear IR steps.

See `bt_planning/` for helpers to build prompts and parse Linear IR JSON plans.

## Verifier Role (During Execution)

- During BT execution, lerobot asks panda_live_viewer for condition verification only.
- panda_live_viewer returns `SUCCESS`/`FAILURE`/`RUNNING`/`WAIT_HUMAN` for individual checks.
- Plan JSON is NOT a status result.
- STATUS/REASON is NOT a plan.
- Do NOT reuse `/lerobot_bt/vlm_result` for plans.

## Live Planning Integration (Future)

- Live planning should use `/lerobot_bt/generate_plan` once the service is available.
- Do NOT implement or guess the service definition unless `GenerateTaskPlan.srv` is present.

## Tests & Checks

- If tests exist, run: `python -m pytest -svv`
- If no tests, run: `python -m compileall .`


## Dry-run Planner

- The dry-run planner (`bt_planning/dry_run_plan.py`) is deterministic and for debug only.
- It emits valid Linear IR JSON for known tasks (e.g., `make_sandwich`) if all required names are present in the registry.

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
