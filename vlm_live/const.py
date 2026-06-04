"""Shared defaults and protocol constants for the Panda VLM live verifier."""

# Local 32B checkpoint used by default for the highest scene-understanding
# quality on safety-critical semantic gates. Smaller Hub models are still
# available via the ROS parameter `model_path` when faster but less reliable
# checks are acceptable.
MODEL_PATH = "/home/bsquitieri-iit.local/models/Qwen3-VL-32B-Instruct"

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
PLANNER_MAX_NEW_TOKENS = 512

# When non-empty, inference is forwarded to a persistent model_server process
# via Unix socket instead of loading the model in-process. No network port is
# opened. Start the server once with:
#   python3 -m vlm_live.model_server --model-path <path>
MODEL_SERVER_URL = ""
