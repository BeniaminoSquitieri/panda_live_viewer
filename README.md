
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

## Runtime BT planner/verifier service

This repo owns the visual side: camera scene, VLM planner service, and VLM
verification during BT execution. It does not generate XML/YAML. `lerobot`
validates Linear IR and compiles BT artifacts.

## ROS environment setup (system or conda)

If you use system ROS (apt in `/opt/ros`):

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
```

If you use a conda/robostack ROS env (e.g., `ros2_jazzy`), `/opt/ros` may not
exist. Use the conda prefix instead:

```bash
conda activate ros2_jazzy
source $CONDA_PREFIX/setup.bash
```

`panda_live_viewer` itself is not a ROS package (no `package.xml`), so running
`colcon build` here will report `0` packages and will not create
`install/setup.bash`. The ROS workspace to build is `~/lerobot`.

## lerobot ROS workspace (interfaces only)

The planner service requires `lerobot_bt_interfaces` to be built in a ROS
workspace (commonly `~/lerobot`). A minimal workspace containing only this
package is enough. Example:

```bash
mkdir -p ~/lerobot/src

git clone --depth 1 --branch runtime-bt-generation-mvp-c \
  https://github.com/BeniaminoSquitieri/lerobot.git /tmp/lerobot_src
cp -a /tmp/lerobot_src/src/lerobot_bt_interfaces ~/lerobot/src/
rm -rf /tmp/lerobot_src

cd ~/lerobot
colcon build --symlink-install
```

Dry-run planner service:

```bash
cd ~/panda_live_viewer
source /opt/ros/$ROS_DISTRO/setup.bash  # or: source $CONDA_PREFIX/setup.bash
source ~/lerobot/install/setup.bash

python3 -m vlm_live.cli \
  --ros-args \
  -p planner_dry_run:=true \
  -p lazy_load_model:=true \
  -p require_generate_plan_service:=true
```

Live planner service, only after dry-run service and `lerobot --no-run`
generation pass:

```bash
python3 -m vlm_live.cli \
  --ros-args \
  -p planner_dry_run:=false \
  -p lazy_load_model:=true \
  -p require_generate_plan_service:=true
```

Check the ROS service:

```bash
ros2 service list | grep /lerobot_bt/generate_plan
ros2 service type /lerobot_bt/generate_plan
```

Expected type:

```text
lerobot_bt_interfaces/srv/GenerateTaskPlan
```

`lazy_load_model:=true` defers Qwen/GPU loading until a live VLM call actually
needs the model. With `planner_dry_run:=true`, the service should answer
without loading Qwen.
This allows dry-run planner service startup even when `qwen_vl_utils`,
`transformers`, and `torch` are not installed.

`require_generate_plan_service:=true` makes startup fail if
`lerobot_bt_interfaces/srv/GenerateTaskPlan` is unavailable; source and build
the `lerobot` ROS workspace before starting this node.

Robot-day runbook: see the sibling checkout
`~/lerobot/docs/runtime_bt_generation_robot_runbook.md` (or
`../lerobot/docs/runtime_bt_generation_robot_runbook.md` when both repos share
the same parent directory).

> WARNING: This repo does not generate XML/YAML. `lerobot` validates and
> compiles.

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

Planner baselines (paper measurement):

- The safe service uses the constrained Linear IR prompt only.
- Baseline prompts can be generated offline with `python3 -m bt_planning.write_baseline_prompts`.
- The direct XML prompt is an unsafe offline baseline, not an execution path.

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
source /opt/ros/$ROS_DISTRO/setup.bash  # or: source $CONDA_PREFIX/setup.bash
source ~/lerobot/install/setup.bash
```

Set `require_generate_plan_service:=false` only when intentionally running the
verifier without the planning service.

## Perception Scene Facts

The passive RGB-D perception bridge can be run alongside the VLM node:

```bash
python3 -m perception.cli \
  --ros-args \
  -p planner_registry_json:='{"objects":[{"canonical_name":"cup","aliases":["mug"]}]}'
```

It subscribes to:

- `/panda/camera/front/image_compressed`
- `/panda/camera/front/depth`
- `/panda/camera/front/camera_info`
- `/panda/camera/wrist/image_compressed`
- `/panda/camera/wrist/depth`
- `/panda/camera/wrist/camera_info`

It publishes latched JSON scene facts on `/perception/scene_facts` and, after
`lerobot_bt_interfaces` is rebuilt with `QueryObjectPose.srv`, serves object
pose lookups on `/perception/query_pose`.

The VLM planner node also subscribes to `/perception/scene_facts`. If a
`/lerobot_bt/generate_plan` request omits `scene_facts_json`, the latest
published scene facts are injected into the planner prompt.

The default segmenter backend is OWL-ViT through `transformers`, loaded lazily
from `segmenter_model_path` (`google/owlvit-base-patch32` by default). It uses
the registry object names as zero-shot labels and converts detections into
box/GrabCut masks. Use `segmenter_backend:=noop` only for verifier-only dry
runs. If the model or dependencies are unavailable, the node publishes
availability plus a segmenter warning instead of invented object poses.

When detections are available, depth is back-projected with `CameraInfo`, object
orientation is estimated from point-cloud PCA, pose confidence/covariance are
emitted, and missing or stale RGB-D/TF data degrades the fact instead of
fabricating geometry. `require_query_pose_service` defaults to true so the node
fails fast when `QueryObjectPose.srv` has not been rebuilt and sourced.

## Robot-Day Step-by-Step Testing

End-to-end smoke test for the RGB-D perception path on the real robot. Run the
steps in order; do not skip the verification command at each step.

> Camera ownership: the **lerobot BT server** opens the RealSense devices and
> republishes their frames. Do **NOT** start `panda_live_camera` (or any other
> RealSense publisher) at the same time — opening the same device twice causes a
> hardware conflict and frame drops.

### 1. Build & source both workspaces

```bash
# lerobot ROS workspace (rebuild after QueryObjectPose.srv changes)
cd ~/lerobot
colcon build --packages-select lerobot_bt_interfaces lerobot_bt_python
source install/setup.bash

# panda_live_viewer environment
cd ~/panda_live_viewer
source <your-conda-or-venv-activate>
```

### 2. Start the lerobot publisher (camera owner)

Launch the BT executor whose YAML enables depth + camera republishing, e.g.
`make_coffee_executor.yaml`. Confirm the config has `use_depth: true` and the
`camera_*_map` / `camera_static_tf_map` entries populated.

### 3. Verify the RGB-D topics are live

```bash
# color (compressed), depth, and intrinsics for each camera
ros2 topic hz /panda/camera/front/image_compressed
ros2 topic hz /panda/camera/front/depth
ros2 topic echo --once /panda/camera/front/camera_info
ros2 topic hz /panda/camera/wrist/image_compressed
ros2 topic hz /panda/camera/wrist/depth
ros2 topic echo --once /panda/camera/wrist/camera_info
```

Expected: non-zero, stable Hz on the image/depth topics; `camera_info` with a
non-empty `k` (fx/fy/ppx/ppy) and `distortion_model: plumb_bob`. Depth and color
must report the **same width/height** (depth is aligned to color).

### 4. Verify the static TF frames

```bash
ros2 run tf2_ros tf2_echo base_link panda_camera_front
ros2 run tf2_ros tf2_echo base_link panda_camera_wrist
```

Expected: a steady transform (no "frame does not exist" errors). The perception
node lifts poses into `base_link` using these transforms.

### 5. Start the perception node

```bash
cd ~/panda_live_viewer
python3 -m perception.cli \
  --ros-args \
  -p planner_registry_json:='{"objects":[{"canonical_name":"cup","aliases":["mug"]}]}' \
  -p segmenter_backend:=owlvit \
  -p require_query_pose_service:=true
```

Expected: the node logs camera/TF availability and the OWL-ViT model load. It
fails fast if `QueryObjectPose.srv` was not rebuilt/sourced (step 1).

### 6. Verify scene facts

```bash
ros2 topic echo --once /perception/scene_facts
```

Expected JSON with per-object entries containing `pose` (translation +
quaternion), `covariance`, `pose_confidence`, `pose_residual_m`, `inlier_ratio`,
and `warnings`. With no detection, the fact degrades (availability + warning)
instead of fabricating a pose.

### 7. Query a pose via the service

```bash
ros2 service call /perception/query_pose \
  lerobot_bt_interfaces/srv/QueryObjectPose \
  "{object_name: 'cup', require_fresh: true, max_age_s: 1.0}"
```

Expected: `success: true` with a `pose_json` payload, or `success: false` with a
clear `error_message` (e.g. stale data) — never a silent empty pose.

### 8. Metric sanity checks

- Place the object at a known distance; confirm `pose.translation` in
  `base_link` matches the tape-measured position within a few centimeters.
- Confirm `pose_residual_m` is small and `inlier_ratio` is high for a clean
  detection; both degrade for partial/occluded views.
- Confirm `warnings` flags expected conditions (`insufficient_depth`,
  `orientation_estimated_pca`, stale RGB-D/TF) rather than staying empty when
  data is poor.

### 9. Planner + gate (optional end-to-end)

Issue a `/lerobot_bt/generate_plan` request **without** `scene_facts_json`; the
planner injects the latest `/perception/scene_facts` automatically. Confirm the
Linear IR plan references the perceived objects.

### Troubleshooting

- No depth / poses always empty: re-check `use_depth: true` and that depth and
  color report the same resolution (alignment). If depth resolution differs,
  the core `rs.align` step is not running.
- `query_pose` service missing: rebuild `lerobot_bt_interfaces` and re-source.
- Frame drops / device busy: ensure only the lerobot server owns the RealSense
  devices (no `panda_live_camera`).
- Wrong object location: verify the `base_link`→camera TF and camera index
  mapping (front vs wrist) in the executor YAML.

### Collect logs for review

```bash
ros2 topic echo /perception/scene_facts > /tmp/scene_facts.log &
# set verifier_experiment_log_path to capture verifier events (optional)
```

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

### Verifier experiment logging (optional)

- Set the ROS parameter `verifier_experiment_log_path` to a JSONL file path to
  record one verifier event per completed inference attempt (skill name, attempt
  id, raw VLM status, published status, the `WAIT_HUMAN`→`RUNNING` coercion flag,
  reason, frame availability, and duration).
- This logging is provenance only: it does **not** change the
  `/lerobot_bt/vlm_result` topic payload, does not alter status decisions, and
  never saves camera images (only boolean frame-availability flags).
- Leave the parameter empty (the default) to disable logging entirely.

## Tests & Checks

- Run unit tests: `python3 -m unittest discover -v`
- Run compile check: `python3 -m compileall .`
- Do not report `pytest` as passing unless it is installed and actually run.


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
