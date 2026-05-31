"""Shared defaults and protocol constants for the Panda VLM live verifier."""

MODEL_PATH = "/home/bsquitieri-iit.local/models/Qwen3-VL-32B-Instruct"

FRONT_TOPIC = "/panda/camera/front/image_compressed"
WRIST_TOPIC = "/panda/camera/wrist/image_compressed"
REQUEST_TOPIC = "/lerobot_bt/vlm_request"
RESULT_TOPIC = "/lerobot_bt/vlm_result"
GENERATE_PLAN_SERVICE = "/lerobot_bt/generate_plan"

CHECK_PERIOD_SECONDS = 1.0

STATUS_RUNNING = "RUNNING"
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILURE = "FAILURE"

DEFAULT_ALLOWED_STATUSES = [
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_FAILURE,
]

VIEW_HOST = "0.0.0.0"
VIEW_PORT = 8081
VIEW_FPS = 10.0
MAX_NEW_TOKENS = 64
PLANNER_MAX_NEW_TOKENS = 2048
