# VLM Server Protocol

This document describes the status-only protocol implemented by the Panda VLM
live verifier. The server receives a Behavior Tree verification request, checks
the latest Panda camera frames with a VLM, and publishes a normalized status.

## Topics

| Direction | Topic | Type | Purpose |
|-----------|-------|------|---------|
| IN | `/panda/camera/front/image_compressed` | `sensor_msgs/msg/CompressedImage` | Front camera stream |
| IN | `/panda/camera/wrist/image_compressed` | `sensor_msgs/msg/CompressedImage` | Wrist camera stream |
| IN | `/lerobot_bt/vlm_request` | `std_msgs/msg/String` | JSON request from the Behavior Tree |
| OUT | `/lerobot_bt/vlm_result` | `std_msgs/msg/String` | JSON status result from the VLM server |

The request and result topic names can be overridden with the ROS parameters
`vlm_request_topic` and `vlm_result_topic`.

## Request Format

The request message is a JSON object stored in the `data` field of
`std_msgs/msg/String`.

```json
{
  "skill_name": "place_first_toast",
  "attempt_id": 3,
  "task": "Verify that the first toast has been placed correctly.",
  "message": "Awaiting VLM result for skill 'place_first_toast'.",
  "allowed_statuses": [
    "RUNNING",
    "SUCCESS",
    "FAILURE",
    "WAIT_HUMAN"
  ]
}
```

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `skill_name` | `string` | Yes | Name of the current Behavior Tree checkpoint |
| `attempt_id` | `int` | No | Attempt counter echoed back in the response, defaults to `0` |
| `task` | `string` | No | Task description added to the VLM prompt |
| `message` | `string` | No | Behavior Tree context message added to the VLM prompt |
| `allowed_statuses` | `list[string]` | No | Status values accepted by the caller; invalid values are ignored |

Only `skill_name` is required. Missing or invalid optional fields are normalized
by `vlm_live/protocol.py`.

## Response Format

The VLM server publishes a JSON object on `/lerobot_bt/vlm_result`.

```json
{
  "skill_name": "place_first_toast",
  "attempt_id": 3,
  "status": "SUCCESS",
  "message": "The toast is correctly placed on the sandwich."
}
```

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `skill_name` | `string` | Yes | Copied from the active request |
| `attempt_id` | `int` | Yes | Copied from the active request |
| `status` | `string` | Yes | Normalized VLM verdict |
| `message` | `string` | No | Clean explanation from the VLM, without a repeated `STATUS=` line |

## Status Values

| Status | Behavior Tree effect | When to use it |
|--------|----------------------|----------------|
| `RUNNING` | BT waits | The VLM is still processing or the scene is inconclusive |
| `SUCCESS` | BT advances | The requested condition is fully satisfied |
| `FAILURE` | BT retries/fails the skill | The requested condition is clearly not satisfied |
| `WAIT_HUMAN` | BT routes to human handling | Explicit human intervention, action, or decision is required |

## Runtime Behavior

1. The node stores the latest front and wrist camera frames.
2. When a request arrives, the node immediately publishes `RUNNING` with
   `message="VLM processing"`.
3. The latest frames are composed side by side and sent to the VLM with the
   normalized request context.
4. The raw VLM output is parsed into one supported `status`.
5. If the VLM output contains `REASON=...`, only the reason text is published in
   `message`.
6. Non-final statuses are reevaluated after `check_period_seconds`.
7. `SUCCESS`, `FAILURE`, and `WAIT_HUMAN` clear the active request.

## Example Commands

Publish a request:

```bash
ros2 topic pub /lerobot_bt/vlm_request std_msgs/msg/String "{data: '{\"skill_name\":\"place_first_toast\",\"attempt_id\":1,\"task\":\"Verify that the first toast has been placed correctly.\",\"message\":\"Awaiting VLM result for skill place_first_toast.\",\"allowed_statuses\":[\"RUNNING\",\"SUCCESS\",\"FAILURE\",\"WAIT_HUMAN\"]}'}"
```

Read results:

```bash
ros2 topic echo /lerobot_bt/vlm_result
```
