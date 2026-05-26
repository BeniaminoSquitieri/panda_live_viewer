# Panda VLM Live Verifier

This project contains a ROS 2 node that reads the Panda robot camera topics in live mode, listens for a text command on another topic, and uses a Vision-Language Model to decide whether the task is still running, succeeded, or failed.

## Files

- `panda_vlm_live.py`: live VLM verifier node
- `panda_live_viewer.py`: MJPEG browser viewer for the same camera topics
- `prova_queen.py`: earlier VLM prototype kept for reference

## What It Does

The node:

1. Subscribes to `/panda/camera/front/image_compressed` and `/panda/camera/wrist/image_compressed`.
2. Waits for a text command on `/panda/vlm/request`.
3. Combines the live camera frames and sends them to Qwen3-VL.
4. Publishes one of these status values on `/panda/vlm/status`:
	- `SUCCESS`
	- `FAILED`
	- `STILL_RUNNING`

`STILL_RUNNING` means the task is not finished yet, or the current scene does not provide enough evidence to conclude.

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

Publish a command string on `/panda/vlm/request`, for example:

```bash
ros2 topic pub /panda/vlm/request std_msgs/msg/String "{data: 'Did the robot place the coffee pod inside the coffee machine?'}"
```

Read the result from `/panda/vlm/status`:

```bash
ros2 topic echo /panda/vlm/status
```

## Notes

- The node keeps running forever and reevaluates the active command using the latest live camera frames.
- If you want to change the command or status topic names, edit the constants at the top of `panda_vlm_live.py`.
- If `accelerate` is missing, `device_map="auto"` will fail during model loading.
- `prova_queen.py` has been restored as the original standalone prototype and is not the main live node.
