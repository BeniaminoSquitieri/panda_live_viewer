"""ROS2 node that subscribes to the Panda cameras and runs VLM verification."""

import json
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from bt_planning.service_logic import build_generate_plan_response

from .const import (
    CHECK_PERIOD_SECONDS,
    DEFAULT_ALLOWED_STATUSES,
    FRONT_TOPIC,
    GENERATE_PLAN_SERVICE,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    PLANNER_MAX_NEW_TOKENS,
    STATUS_FAILURE,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_WAIT_HUMAN,
    VIEW_FPS,
    VIEW_HOST,
    VIEW_PORT,
    REQUEST_TOPIC,
    RESULT_TOPIC,
    WRIST_TOPIC,
)
from .camera import compose, decode_image
from .model import load_model, run_inference, run_text_inference
from .prompt import build_prompt, fit_status
from .protocol import build_result_payload, coerce_result_for_request, parse_request
from .view import start_view

try:
    from lerobot_bt_interfaces.srv import GenerateTaskPlan
except ImportError:
    GenerateTaskPlan = None


@dataclass(frozen=True)
class NodeConfig:
    """Resolved runtime configuration for the VLM node."""

    model_path: str
    front_topic: str
    wrist_topic: str
    req_topic: str
    res_topic: str
    plan_service: str
    check_s: float
    reasoning: bool
    log_out: bool
    show_view: bool
    view_host: str
    view_port: int
    view_fps: float
    max_tokens: int
    planner_dry_run: bool
    planner_max_tokens: int
    lazy_load_model: bool
    require_generate_plan_service: bool


class VlmNode(Node):
    """ROS2 node that keeps the latest camera frames and evaluates VLM checks."""

    def __init__(self):
        super().__init__("lerobot_bt_vlm_server")

        self._declare_parameters()
        cfg = self._load_config()

        self.model_path = cfg.model_path
        self.front_topic = cfg.front_topic
        self.wrist_topic = cfg.wrist_topic
        self.req_topic = cfg.req_topic
        self.res_topic = cfg.res_topic
        self.plan_service = cfg.plan_service
        self.check_s = cfg.check_s
        self.reasoning = cfg.reasoning
        self.log_out = cfg.log_out
        self.show_view = cfg.show_view
        self.view_host = cfg.view_host
        self.view_port = cfg.view_port
        self.view_fps = cfg.view_fps
        self.max_tokens = cfg.max_tokens
        self.planner_dry_run = cfg.planner_dry_run
        self.planner_max_tokens = cfg.planner_max_tokens
        self.lazy_load_model = cfg.lazy_load_model
        self.require_generate_plan_service = cfg.require_generate_plan_service

        self._init_state()
        self._init_ros_interfaces()
        if not self.lazy_load_model:
            self._ensure_vlm_loaded()
        self._start_worker()

        if self.show_view:
            self._start_viewer()

        self.get_logger().info(
            f"Ready. Listening on {self.front_topic}, {self.wrist_topic}, "
            f"{self.req_topic}, and {self._plan_service_status()}"
        )

    def _declare_parameters(self) -> None:
        """Declare all configurable ROS parameters for this node."""
        self.declare_parameter("model_path", MODEL_PATH)
        self.declare_parameter("front_topic", FRONT_TOPIC)
        self.declare_parameter("wrist_topic", WRIST_TOPIC)
        self.declare_parameter("vlm_request_topic", REQUEST_TOPIC)
        self.declare_parameter("vlm_result_topic", RESULT_TOPIC)
        self.declare_parameter("generate_plan_service", GENERATE_PLAN_SERVICE)
        self.declare_parameter("check_period_seconds", CHECK_PERIOD_SECONDS)
        self.declare_parameter("include_reasoning", True)
        self.declare_parameter("log_model_output", True)
        self.declare_parameter("enable_viewer", False)
        self.declare_parameter("viewer_host", VIEW_HOST)
        self.declare_parameter("viewer_port", VIEW_PORT)
        self.declare_parameter("viewer_fps", VIEW_FPS)
        self.declare_parameter("max_new_tokens", MAX_NEW_TOKENS)
        self.declare_parameter("planner_dry_run", False)
        self.declare_parameter("planner_max_new_tokens", PLANNER_MAX_NEW_TOKENS)
        self.declare_parameter("lazy_load_model", True)
        self.declare_parameter("require_generate_plan_service", True)

    def _load_config(self) -> NodeConfig:
        """Read and normalize parameter values into an immutable config object."""
        req_topic = self.get_parameter("vlm_request_topic").value
        res_topic = self.get_parameter("vlm_result_topic").value

        return NodeConfig(
            model_path=self.get_parameter("model_path").value,
            front_topic=self.get_parameter("front_topic").value,
            wrist_topic=self.get_parameter("wrist_topic").value,
            req_topic=req_topic,
            res_topic=res_topic,
            plan_service=self.get_parameter("generate_plan_service").value,
            check_s=float(self.get_parameter("check_period_seconds").value),
            reasoning=bool(self.get_parameter("include_reasoning").value),
            log_out=bool(self.get_parameter("log_model_output").value),
            show_view=bool(self.get_parameter("enable_viewer").value),
            view_host=str(self.get_parameter("viewer_host").value),
            view_port=int(self.get_parameter("viewer_port").value),
            view_fps=float(self.get_parameter("viewer_fps").value),
            max_tokens=max(1, int(self.get_parameter("max_new_tokens").value)),
            planner_dry_run=bool(self.get_parameter("planner_dry_run").value),
            planner_max_tokens=max(1, int(self.get_parameter("planner_max_new_tokens").value)),
            lazy_load_model=bool(self.get_parameter("lazy_load_model").value),
            require_generate_plan_service=bool(
                self.get_parameter("require_generate_plan_service").value
            ),
        )

    def _init_state(self) -> None:
        """Initialize thread synchronization and runtime state variables."""
        self.lock = threading.Lock()
        self.inference_lock = threading.RLock()
        self.callback_group = ReentrantCallbackGroup()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.frames = {"front": None, "wrist": None}
        self.req = None
        self.req_id = 0
        self.processor = None
        self.model = None
        self.device = None

    def _init_ros_interfaces(self) -> None:
        """Create ROS subscriptions and publishers."""
        self.generate_plan_srv = None
        if GenerateTaskPlan is None:
            message = (
                "GenerateTaskPlan service type is unavailable; "
                f"{self.plan_service} was not created. Fix your ROS environment with: "
                "source /opt/ros/$ROS_DISTRO/setup.bash && source install/setup.bash"
            )
            if self.require_generate_plan_service:
                self.get_logger().error(message)
                raise RuntimeError(message)
            self.get_logger().warning(message)

        self.front_sub = self.create_subscription(
            CompressedImage,
            self.front_topic,
            lambda msg: self._on_image(msg, "front"),
            10,
            callback_group=self.callback_group,
        )
        self.wrist_sub = self.create_subscription(
            CompressedImage,
            self.wrist_topic,
            lambda msg: self._on_image(msg, "wrist"),
            10,
            callback_group=self.callback_group,
        )
        self.request_sub = self.create_subscription(
            String,
            self.req_topic,
            self._on_req,
            10,
            callback_group=self.callback_group,
        )
        self.status_pub = self.create_publisher(String, self.res_topic, 10)
        if GenerateTaskPlan is not None:
            self.generate_plan_srv = self.create_service(
                GenerateTaskPlan,
                self.plan_service,
                self._on_generate_plan,
                callback_group=self.callback_group,
            )

    def _plan_service_status(self) -> str:
        if self.generate_plan_srv is None:
            return "GenerateTaskPlan service disabled"
        return self.plan_service

    def _load_vlm(self) -> None:
        """Load processor and model weights."""
        self.get_logger().info(f"Loading VLM from {self.model_path}")
        self.processor, self.model, self.device = load_model(self.model_path)

    def _vlm_loaded(self) -> bool:
        return self.processor is not None and self.model is not None and self.device is not None

    def _ensure_vlm_loaded(self) -> None:
        """Load the VLM on first use, guarded by the inference lock."""
        if self._vlm_loaded():
            return

        with self.inference_lock:
            if not self._vlm_loaded():
                self._load_vlm()

    def _start_worker(self) -> None:
        """Start background evaluation loop."""
        self.worker = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.worker.start()

    def _on_image(self, msg: CompressedImage, camera_name: str) -> None:
        frame = decode_image(msg)
        if frame is None:
            self.get_logger().warning(f"Could not decode {camera_name} frame")
            return

        with self.lock:
            self.frames[camera_name] = frame

    def _publish_result(self, request: dict, status: str, reason: str) -> None:
        payload = build_result_payload(request, status, reason)
        if payload is None:
            return

        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)
        self.get_logger().info(
            f"{status}: {reason} (skill={payload['skill_name']}, attempt={payload['attempt_id']})"
        )

    def _on_req(self, msg: String) -> None:
        request = parse_request(msg, self.get_logger())
        if request is None:
            return

        with self.lock:
            self.req = request
            self.req_id += 1

        self.get_logger().info(
            f"Received VLM request for {request['skill_name']} (attempt {request['attempt_id']})"
        )
        if request.get("allowed_statuses"):
            self.get_logger().info(f"Allowed statuses: {', '.join(request['allowed_statuses'])}")
        task_text = request.get("task") or ""
        if task_text:
            self.get_logger().info(f"Task description: {task_text}")
        if request.get("message"):
            self.get_logger().info(f"BT message: {request['message']}")
        request_prompt = build_prompt(request, self.reasoning)
        self.get_logger().info(f"VLM prompt:\n{request_prompt}")
        self._publish_result(request, STATUS_RUNNING, "VLM processing")
        self.wake.set()

    def _get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            front = self.frames["front"]
            wrist = self.frames["wrist"]
            front_copy = None if front is None else front.copy()
            wrist_copy = None if wrist is None else wrist.copy()
        return front_copy, wrist_copy

    def _get_active_request(self) -> Tuple[Optional[dict], int]:
        with self.lock:
            if self.req is None:
                return None, 0
            return dict(self.req), self.req_id

    def _set_result(self, request: dict, status: str, reason: str) -> None:
        self._publish_result(request, status, reason)

    def _start_viewer(self) -> None:
        start_view(
            get_frames=self._get_frames,
            scene_fn=self._compose_scene,
            stop_event=self.stop,
            host=self.view_host,
            port=self.view_port,
            fps=self.view_fps,
            logger=self.get_logger(),
        )

    def _compose_scene(self) -> Optional[np.ndarray]:
        front, wrist = self._get_frames()
        return compose(front, wrist)

    def _run_vlm(self, request: dict):
        scene = self._compose_scene()
        if scene is None:
            return STATUS_RUNNING, "Waiting for both camera streams."

        prompt = build_prompt(request, self.reasoning)
        with self.inference_lock:
            self._ensure_vlm_loaded()
            return run_inference(
                scene=scene,
                prompt=prompt,
                processor=self.processor,
                model=self.model,
                device=self.device,
                tokens=self.max_tokens,
                reasoning=self.reasoning,
                log_out=self.log_out,
                logger=self.get_logger(),
            )

    def _run_planner_vlm(self, prompt: str) -> str:
        scene = self._compose_scene()
        if scene is None:
            raise RuntimeError("Waiting for camera streams before planning.")

        with self.inference_lock:
            self._ensure_vlm_loaded()
            return run_text_inference(
                scene=scene,
                prompt=prompt,
                processor=self.processor,
                model=self.model,
                device=self.device,
                tokens=self.planner_max_tokens,
                log_out=self.log_out,
                logger=self.get_logger(),
            )

    def _on_generate_plan(self, request, response):
        result = build_generate_plan_response(
            task_name=request.task_name,
            planner_registry_json=request.planner_registry_json,
            scene_facts_json=request.scene_facts_json,
            dry_run=self.planner_dry_run,
            vlm_backend=self._run_planner_vlm,
        )

        if result.prompt:
            self.get_logger().info(f"Planner prompt:\n{result.prompt}")

        response.success = result.success
        response.plan_json = result.plan_json
        response.error_message = result.error_message
        if result.success:
            self.get_logger().info(
                f"Generated Linear IR plan for task {request.task_name!r} "
                f"(dry_run={self.planner_dry_run})"
            )
        else:
            self.get_logger().warning(
                f"GenerateTaskPlan failed for task {request.task_name!r}: "
                f"{result.error_message}"
            )
        return response

    def _evaluation_loop(self) -> None:
        while rclpy.ok() and not self.stop.is_set():
            request, request_id = self._get_active_request()
            if request is None:
                self.wake.wait(timeout=0.5)
                self.wake.clear()
                continue

            try:
                status, reason = self._run_vlm(request)
            except Exception as exc:
                self.get_logger().exception("VLM inference failed")
                status, reason = STATUS_RUNNING, f"Inference error: {exc}"

            _, current_request_id = self._get_active_request()
            if current_request_id != request_id:
                continue

            raw_status = status
            status, reason = coerce_result_for_request(request, status, reason)
            if not (raw_status == STATUS_WAIT_HUMAN and status == STATUS_RUNNING):
                status = fit_status(status, request.get("allowed_statuses", DEFAULT_ALLOWED_STATUSES))
            self._set_result(request, status, reason)

            if status in (STATUS_SUCCESS, STATUS_FAILURE, STATUS_WAIT_HUMAN):
                with self.lock:
                    if self.req_id == request_id:
                        self.req = None
                self.wake.clear()
            else:
                self.wake.wait(timeout=self.check_s)
                self.wake.clear()

    def destroy_node(self):
        self.stop.set()
        self.wake.set()
        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=2.0)
        return super().destroy_node()
