# Panda VLM Live Verifier

This project contains a ROS 2 node that reads the Panda robot camera topics in live mode, listens for a text command on another topic, and uses a Vision-Language Model to decide whether the task is still running, succeeded, or failed.

## Files

- `panda_vlm_live.py`: live VLM verifier node
- `panda_live_viewer.py`: MJPEG browser viewer for the same camera topics
- `prova_queen.py`: earlier VLM prototype kept for reference

## What It Does

The node:

1. Subscribes to `/panda/camera/front/image_compressed` and `/panda/camera/wrist/image_compressed`.
2. Waits for a JSON request on `/lerobot_bt/vlm_request`.
3. Combines the live camera frames and sends them to Qwen3-VL.
4. Publishes a JSON response on `/lerobot_bt/vlm_result` with a `status` value such as:
	- `SUCCESS`
	- `FAILURE`
	- `RUNNING`
	- `WAIT_HUMAN`
	- `MANUAL_INTERVENTION_REQUIRED`
	- `PENDING`

## Requirements

- Python 3.8+
- ROS 2 installed and sourced
- `torch`
- `transformers`
- `accelerate`
- `qwen-vl-utils`
- `opencv-python`
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

Publish a JSON request on `/lerobot_bt/vlm_request`, for example:

```bash
ros2 topic pub /lerobot_bt/vlm_request std_msgs/msg/String "{data: '{\"event\":\"vlm_check_requested\",\"skill_name\":\"place_first_toast\",\"attempt_id\":1,\"status\":\"PENDING\",\"message\":\"Awaiting VLM result for skill place_first_toast.\",\"allowed_statuses\":[\"PENDING\",\"RUNNING\",\"WAIT_HUMAN\",\"MANUAL_INTERVENTION_REQUIRED\",\"SUCCESS\",\"FAILURE\"],\"allowed_next_actions\":[\"CONTINUE\",\"RETRY_SKILL\",\"WAIT_HUMAN\",\"REQUEST_MANUAL_INTERVENTION\"]}'}"
```

Read the result from `/lerobot_bt/vlm_result`:

```bash
ros2 topic echo /lerobot_bt/vlm_result
```

## Notes

- The node keeps running forever and reevaluates the active request using the latest live camera frames.
- You can override topic names via ROS parameters `vlm_request_topic` and `vlm_result_topic` (legacy `command_topic` and `status_topic` still work).
- If `accelerate` is missing, `device_map="auto"` will fail during model loading.
