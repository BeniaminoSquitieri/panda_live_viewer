# Panda VLM Live Verifier

ROS 2 node that reads the Panda robot camera topics in live mode, listens for VLM verification requests, and publishes task status results for the Behavior Tree stack.

## Files

- `panda_vlm_live.py`: thin entry point that starts the verifier
- `vlm_live/cli.py`: command-line bootstrap for the ROS2 node
- `vlm_live/node.py`: main ROS2 node and control loop
- `vlm_live/model.py`: model loading and VLM inference
- `vlm_live/camera.py`: camera-frame decoding, composition, and temp file helpers
- `vlm_live/protocol.py`: request parsing, token normalization, and result payload building
- `vlm_live/prompt.py`: prompt building and status normalization
- `vlm_live/view.py`: optional browser MJPEG viewer
- `vlm_live/const.py`: shared defaults and protocol constants
- `VLM_SERVER_REQUIREMENTS.md`: protocol notes for the LeRobot BT VLM integration

## What It Does

The node:

1. Subscribes to `/panda/camera/front/image_compressed` and `/panda/camera/wrist/image_compressed`.
2. Waits for a JSON request on `/lerobot_bt/vlm_request`.
3. Combines the live camera frames and sends them to Qwen3-VL.
4. Publishes a JSON response on `/lerobot_bt/vlm_result` with a `status` value such as:
   - `SUCCESS`
   - `FAILURE`
   - `RUNNING`

An optional MJPEG viewer can be enabled from the same node with the ROS parameter `enable_viewer:=true`.

## Input And Output

The node consumes:

1. Two live compressed camera streams:
   - `/panda/camera/front/image_compressed`
   - `/panda/camera/wrist/image_compressed`
2. A JSON request on `/lerobot_bt/vlm_request` wrapped in `std_msgs/msg/String`.

The request JSON is normalized by `vlm_live/protocol.py` and should include at least:

- `skill_name` as a non-empty string
- `attempt_id` as an integer-like value
- `message` as an optional free-text BT message
- `task` as an optional task description for the VLM prompt
- `allowed_statuses` as an optional list of accepted status tokens

The node publishes:

- A JSON response on `/lerobot_bt/vlm_result`, also wrapped in `std_msgs/msg/String`
- A payload with `skill_name`, `attempt_id`, `status`, and `message` when the VLM returns a reason

The returned `status` is one of:

- `SUCCESS`
- `FAILURE`
- `RUNNING`

The `message` field in the result contains only the model reason when the model output includes a non-empty explanation.

## Requirements

- Python 3.8+
- ROS 2 installed and sourced
- `torch`
- `transformers`
- `accelerate`
- `qwen-vl-utils`
- `opencv-python`
- `flask` (only needed when `enable_viewer` is true)
- ROS 2 Python packages: `rclpy`, `sensor_msgs`, `std_msgs`

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then source your ROS 2 environment:

```bash
source /opt/ros/<distro>/setup.bash
```

## Run

Start the node:

```bash
python3 panda_vlm_live.py
```

Start the node with the browser viewer enabled:

```bash
python3 panda_vlm_live.py --ros-args -p enable_viewer:=true
```

Publish a JSON request on `/lerobot_bt/vlm_request`, for example:

```bash
ros2 topic pub /lerobot_bt/vlm_request std_msgs/msg/String "{data: '{\"skill_name\":\"place_first_toast\",\"attempt_id\":1,\"task\":\"Verify that the first toast has been placed correctly.\",\"message\":\"Awaiting VLM result for skill place_first_toast.\",\"allowed_statuses\":[\"RUNNING\",\"SUCCESS\",\"FAILURE\"]}'}"
```

Read the result from `/lerobot_bt/vlm_result`:

```bash
ros2 topic echo /lerobot_bt/vlm_result
```

## Notes

- The node keeps running forever and reevaluates the active request using the latest live camera frames.
- You can override topic names via ROS parameters `vlm_request_topic` and `vlm_result_topic`.
- If `accelerate` is missing, `device_map="auto"` will fail during model loading.
