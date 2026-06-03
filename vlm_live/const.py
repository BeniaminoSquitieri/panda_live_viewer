"""Shared defaults and protocol constants for the Panda VLM live verifier."""

# Hugging Face repo id (or local path). On first run the weights are downloaded
# and cached under ~/.cache/huggingface/hub, so later runs reuse the local copy
# without re-downloading. Smaller than Qwen3-VL-32B for faster scene checks;
# drop to "Qwen/Qwen3-VL-4B-Instruct" or "Qwen/Qwen3-VL-2B-Instruct" if you need
# even faster inference, or restore the local 32B path for max quality.
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"

FRONT_TOPIC = "/panda/camera/front/image_compressed"
WRIST_TOPIC = "/panda/camera/wrist/image_compressed"
REQUEST_TOPIC = "/lerobot_bt/vlm_request"
RESULT_TOPIC = "/lerobot_bt/vlm_result"
GENERATE_PLAN_SERVICE = "/lerobot_bt/generate_plan"
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
]

VIEW_HOST = "0.0.0.0"
VIEW_PORT = 8081
VIEW_FPS = 10.0
MAX_NEW_TOKENS = 64
PLANNER_MAX_NEW_TOKENS = 2048
