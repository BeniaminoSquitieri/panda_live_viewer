"""Shared defaults and protocol constants for the Panda VLM live verifier."""

import os

# Local 32B checkpoint used by default for the highest scene-understanding
# quality on safety-critical semantic gates. Smaller Hub models are still
# available via the ROS parameter `model_path` when faster but less reliable
# checks are acceptable.
MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "Qwen/Qwen3-VL-32B-Instruct")

# ROS CONTRACT with the separate VLM-BT-BC repo. These topic/service names and
# the STATUS_* vocabulary below are a wire contract: VLM-BT-BC consumes them by
# string. A rename here breaks the runtime silently. Mirror any change in
# VLM-BT-BC (config.py / vlm/verification.py) and its docs/VLM_BT_BC_ARCHITETTURA.md.
FRONT_TOPIC = "/panda/camera/front/image_compressed"
WRIST_TOPIC = "/panda/camera/wrist/image_compressed"
REQUEST_TOPIC = "/lerobot_bt/vlm_request"
RESULT_TOPIC = "/lerobot_bt/vlm_result"
GENERATE_PLAN_SERVICE = "/lerobot_bt/generate_plan"
GROUND_INSTRUCTION_SERVICE = "/lerobot_bt/ground_instruction"
SCENE_FACTS_TOPIC = "/perception/scene_facts"

CHECK_PERIOD_SECONDS = 1.0

STATUS_RUNNING = "RUNNING"
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILURE = "FAILURE"
STATUS_WAIT_HUMAN = "WAIT_HUMAN"

SUPPORTED_STATUSES = [
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_FAILURE,
    STATUS_WAIT_HUMAN,
]

DEFAULT_ALLOWED_STATUSES = [
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_FAILURE,
    STATUS_WAIT_HUMAN,
]

VIEW_HOST = "0.0.0.0"
VIEW_PORT = 8081
VIEW_FPS = 10.0
MAX_NEW_TOKENS = 64
PLANNER_MAX_NEW_TOKENS = 1024

# When non-empty, inference is forwarded to a persistent model_server process
# via Unix socket instead of loading the model in-process. No network port is
# opened. Start the server once with:
#   python3 -m vlm_live.model_server --model-path <path>
MODEL_SERVER_URL = ""
