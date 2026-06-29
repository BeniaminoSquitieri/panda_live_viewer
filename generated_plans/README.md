# generated_plans/

This folder is for debug outputs from the visual planner (panda_live_viewer).

- **lerobot** is responsible for validation, XML/YAML generation, and execution.
- **panda_live_viewer** only proposes Linear IR JSON plans.
- Live integration should use the ROS service, not shared files.
- Do **not** commit generated plan artifacts unless intentionally adding examples.

Subfolders:
- `raw_planner_responses/`: Raw outputs from the VLM or planner (debug only)
- `linear_ir/`: Linear IR JSON plans (debug only)
