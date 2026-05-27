"""ROS 2 node that uses a VLM to verify whether a Panda robot task is complete."""

import json
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
from qwen_vl_utils import process_vision_info
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# Default local path to the multimodal model.
MODEL_PATH = "/home/bsquitieri-iit.local/models/Qwen3-VL-32B-Instruct"
# ROS topic for the front camera compressed image stream.
FRONT_TOPIC = "/panda/camera/front/image_compressed"
# ROS topic for the wrist camera compressed image stream.
WRIST_TOPIC = "/panda/camera/wrist/image_compressed"
# ROS topic where the BT stack publishes VLM requests.
VLM_REQUEST_TOPIC = "/lerobot_bt/vlm_request"
# ROS topic where the VLM result is published.
VLM_RESULT_TOPIC = "/lerobot_bt/vlm_result"
# Time in seconds between repeated checks while the task is still running.
CHECK_PERIOD_SECONDS = 1.0
# Status values accepted by the LeRobot BT stack.
STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_WAIT_HUMAN = "WAIT_HUMAN"
STATUS_MANUAL_INTERVENTION_REQUIRED = "MANUAL_INTERVENTION_REQUIRED"
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILURE = "FAILURE"

DEFAULT_ALLOWED_STATUSES = [
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_WAIT_HUMAN,
    STATUS_MANUAL_INTERVENTION_REQUIRED,
    STATUS_SUCCESS,
    STATUS_FAILURE,
]

DEFAULT_ALLOWED_NEXT_ACTIONS = [
    "CONTINUE",
    "RETRY_SKILL",
    "WAIT_HUMAN",
    "REQUEST_MANUAL_INTERVENTION",
]

VIEWER_HTML = """
<!doctype html>
<html>
<head>
    <title>VLM Live Viewer</title>
    <style>
        body {
            font-family: sans-serif;
            background: #111;
            color: #eee;
            margin: 24px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        .card {
            background: #1b1b1b;
            padding: 16px;
            border-radius: 12px;
        }
        img {
            width: 100%;
            border-radius: 8px;
            background: #000;
        }
    </style>
</head>
<body>
    <h1>VLM Live Viewer</h1>
    <p>Front, wrist, and composed scene streams</p>

    <div class="grid">
        <div class="card">
            <h2>Front camera</h2>
            <img src="/stream/front">
        </div>
        <div class="card">
            <h2>Wrist camera</h2>
            <img src="/stream/wrist">
        </div>
        <div class="card">
            <h2>Composed scene</h2>
            <img src="/stream/composed">
        </div>
    </div>
</body>
</html>
"""


def decode_compressed_image(msg: CompressedImage) -> Optional[np.ndarray]:
    # Convert the ROS compressed payload into a NumPy byte array.
    np_arr = np.frombuffer(msg.data, np.uint8)
    # Decode the JPEG or PNG bytes into a BGR OpenCV image.
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # Return the decoded frame, or None if decoding failed.
    return frame


def make_placeholder(label: str, size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    # Unpack the requested image size as width and height.
    width, height = size
    # Create a black image with three color channels.
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw the message in the center-left of the placeholder image.
    cv2.putText(
        image,
        label,
        (40, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    # Return the generated placeholder image.
    return image


def label_frame(frame: np.ndarray, title: str) -> np.ndarray:
    # Work on a copy so the original frame is not modified in place.
    output = frame.copy()
    # Add a visible title in the top-left corner.
    cv2.putText(
        output,
        title,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    # Return the labeled image.
    return output


class PandaVLMVerifier(Node):
    def __init__(self):
        # Initialize the ROS node with a fixed name.
        super().__init__("lerobot_bt_vlm_server")

        # Declare configurable ROS parameters with default values.
        self.declare_parameter("model_path", MODEL_PATH)
        self.declare_parameter("front_topic", FRONT_TOPIC)
        self.declare_parameter("wrist_topic", WRIST_TOPIC)
        self.declare_parameter("vlm_request_topic", VLM_REQUEST_TOPIC)
        self.declare_parameter("vlm_result_topic", VLM_RESULT_TOPIC)
        self.declare_parameter("command_topic", "")
        self.declare_parameter("status_topic", "")
        self.declare_parameter("check_period_seconds", CHECK_PERIOD_SECONDS)
        self.declare_parameter("include_reasoning", True)
        self.declare_parameter("log_model_output", True)
        self.declare_parameter("enable_viewer", False)
        self.declare_parameter("viewer_host", "0.0.0.0")
        self.declare_parameter("viewer_port", 8081)
        self.declare_parameter("viewer_fps", 10.0)
        self.declare_parameter("max_new_tokens", 64)

        # Read the effective parameter values after ROS overrides.
        self.model_path = self.get_parameter("model_path").value
        self.front_topic = self.get_parameter("front_topic").value
        self.wrist_topic = self.get_parameter("wrist_topic").value
        self.vlm_request_topic = self.get_parameter("vlm_request_topic").value
        self.vlm_result_topic = self.get_parameter("vlm_result_topic").value
        legacy_command_topic = self.get_parameter("command_topic").value
        legacy_status_topic = self.get_parameter("status_topic").value
        self.check_period_seconds = float(self.get_parameter("check_period_seconds").value)
        self.include_reasoning = bool(self.get_parameter("include_reasoning").value)
        self.log_model_output = bool(self.get_parameter("log_model_output").value)
        self.enable_viewer = bool(self.get_parameter("enable_viewer").value)
        self.viewer_host = str(self.get_parameter("viewer_host").value)
        self.viewer_port = int(self.get_parameter("viewer_port").value)
        self.viewer_fps = float(self.get_parameter("viewer_fps").value)
        self.viewer_period = 1.0 / max(self.viewer_fps, 1.0)
        self.max_new_tokens = max(1, int(self.get_parameter("max_new_tokens").value))

        if legacy_command_topic:
            self.get_logger().warning("command_topic is deprecated; use vlm_request_topic")
            self.vlm_request_topic = legacy_command_topic
        if legacy_status_topic:
            self.get_logger().warning("status_topic is deprecated; use vlm_result_topic")
            self.vlm_result_topic = legacy_status_topic

        # Protect shared state accessed by the ROS callbacks and worker thread.
        self._lock = threading.Lock()
        # Signal used to stop the background evaluation loop.
        self._stop_event = threading.Event()
        # Signal used to wake the worker when a new request arrives.
        self._new_request_event = threading.Event()
        # Store the most recent frame from each camera.
        self._latest_frames = {"front": None, "wrist": None}
        # Store the request currently being evaluated.
        self._active_request = None
        self._active_request_id = 0
        # Keep the latest published status in memory.
        self._latest_status = STATUS_RUNNING
        # Keep the latest human-readable reason in memory.
        self._latest_reason = "Waiting for a VLM request."

        # Subscribe to the front camera compressed image topic.
        self.front_sub = self.create_subscription(
            CompressedImage,
            self.front_topic,
            lambda msg: self._image_callback(msg, "front"),
            10,
        )
        # Subscribe to the wrist camera compressed image topic.
        self.wrist_sub = self.create_subscription(
            CompressedImage,
            self.wrist_topic,
            lambda msg: self._image_callback(msg, "wrist"),
            10,
        )
        # Subscribe to the request topic that triggers verification.
        self.request_sub = self.create_subscription(
            String,
            self.vlm_request_topic,
            self._request_callback,
            10,
        )
        # Create a publisher for the verification status.
        self.status_pub = self.create_publisher(String, self.vlm_result_topic, 10)

        # Load the processor and the VLM model from disk.
        self.get_logger().info(f"Loading VLM from {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        # Switch the model to inference mode.
        self.model.eval()
        # Remember the device where the model weights live.
        self.model_input_device = next(self.model.parameters()).device

        # Start the background worker that periodically runs inference.
        self.worker = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.worker.start()

        if self.enable_viewer:
            self._start_viewer()

        # Log the topics the node is listening to.
        self.get_logger().info(
            f"Ready. Listening on {self.front_topic}, {self.wrist_topic}, and {self.vlm_request_topic}"
        )

    def _image_callback(self, msg: CompressedImage, camera_name: str) -> None:
        # Decode the compressed ROS image into an OpenCV frame.
        frame = decode_compressed_image(msg)
        if frame is None:
            # Log a warning if decoding fails and skip the frame.
            self.get_logger().warning(f"Could not decode {camera_name} frame")
            return

        # Store the newest frame for this camera under lock.
        with self._lock:
            self._latest_frames[camera_name] = frame

    def _sanitize_status_list(self, value) -> list:
        if not isinstance(value, list):
            return DEFAULT_ALLOWED_STATUSES.copy()
        cleaned = []
        for item in value:
            token = str(item).strip().upper()
            if token and token in DEFAULT_ALLOWED_STATUSES and token not in cleaned:
                cleaned.append(token)
        return cleaned or DEFAULT_ALLOWED_STATUSES.copy()

    def _sanitize_action_list(self, value) -> list:
        if not isinstance(value, list):
            return DEFAULT_ALLOWED_NEXT_ACTIONS.copy()
        cleaned = []
        for item in value:
            token = str(item).strip().upper()
            if token and token not in cleaned:
                cleaned.append(token)
        return cleaned or DEFAULT_ALLOWED_NEXT_ACTIONS.copy()

    def _parse_request(self, msg: String) -> Optional[dict]:
        raw = msg.data.strip()
        if not raw:
            return None

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.get_logger().warning("Ignoring malformed JSON on VLM request topic")
            return None

        if not isinstance(payload, dict):
            self.get_logger().warning("Ignoring VLM request that is not a JSON object")
            return None

        skill_name = str(payload.get("skill_name", "")).strip()
        if not skill_name:
            self.get_logger().warning("Ignoring VLM request with empty skill_name")
            return None

        attempt_raw = payload.get("attempt_id", 0)
        try:
            attempt_id = int(attempt_raw)
        except (TypeError, ValueError):
            attempt_id = 0
        if attempt_id < 0:
            attempt_id = 0

        message = str(payload.get("message", "")).strip()
        task_description = payload.get("task") or payload.get("task_description") or payload.get("skill_description")
        task_description = str(task_description).strip() if task_description is not None else ""
        allowed_statuses = self._sanitize_status_list(payload.get("allowed_statuses"))
        allowed_next_actions = self._sanitize_action_list(payload.get("allowed_next_actions"))

        return {
            "skill_name": skill_name,
            "attempt_id": attempt_id,
            "message": message,
            "task": task_description,
            "allowed_statuses": allowed_statuses,
            "allowed_next_actions": allowed_next_actions,
        }

    def _publish_result(self, request: dict, status: str, reason: str) -> None:
        skill_name = request.get("skill_name", "").strip()
        if not skill_name:
            return

        payload = {
            "skill_name": skill_name,
            "attempt_id": request.get("attempt_id", 0),
            "status": status,
        }
        if reason:
            payload["message"] = reason

        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)
        self.get_logger().info(
            f"{status}: {reason} (skill={skill_name}, attempt={payload['attempt_id']})"
        )

    def _request_callback(self, msg: String) -> None:
        request = self._parse_request(msg)
        if request is None:
            return

        with self._lock:
            self._active_request = request
            self._active_request_id += 1
            self._latest_status = STATUS_RUNNING
            self._latest_reason = "New VLM request received."

        self.get_logger().info(
            f"Received VLM request for {request['skill_name']} (attempt {request['attempt_id']})"
        )
        if request.get("allowed_statuses"):
            self.get_logger().info(
                f"Allowed statuses: {', '.join(request['allowed_statuses'])}"
            )
        if request.get("allowed_next_actions"):
            self.get_logger().info(
                f"Allowed next actions: {', '.join(request['allowed_next_actions'])}"
            )
        task_description = request.get("task") or ""
        if task_description:
            self.get_logger().info(f"Task description: {task_description}")
        if request.get("message"):
            self.get_logger().info(f"BT message: {request['message']}")
        self.get_logger().info(f"VLM prompt:\n{self._build_prompt(request)}")
        self._publish_result(request, STATUS_RUNNING, "VLM processing")
        self._new_request_event.set()

    def _get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # Read both latest frames under lock to avoid concurrent writes.
        with self._lock:
            front = self._latest_frames["front"]
            wrist = self._latest_frames["wrist"]
            # Copy the arrays so inference works on stable data.
            front_copy = None if front is None else front.copy()
            wrist_copy = None if wrist is None else wrist.copy()
        # Return the two camera frames to the caller.
        return front_copy, wrist_copy

    def _get_active_request(self) -> Tuple[Optional[dict], int]:
        # Read the current request under lock.
        with self._lock:
            if self._active_request is None:
                return None, 0
            return dict(self._active_request), self._active_request_id

    def _set_result(self, request: dict, status: str, reason: str) -> None:
        # Update the cached result under lock.
        with self._lock:
            self._latest_status = status
            self._latest_reason = reason

        # Publish the JSON status over ROS.
        self._publish_result(request, status, reason)

    def _start_viewer(self) -> None:
        try:
            from flask import Flask, Response, render_template_string
        except Exception as exc:
            self.get_logger().warning(f"Viewer disabled (Flask import failed): {exc}")
            return

        app = Flask(__name__)

        def generate_stream(camera_name: str):
            while rclpy.ok() and not self._stop_event.is_set():
                if camera_name == "composed":
                    frame = self._compose_scene()
                    if frame is None:
                        frame = make_placeholder("Waiting for cameras...")
                else:
                    front, wrist = self._get_frames()
                    frame = front if camera_name == "front" else wrist
                    if frame is None:
                        frame = make_placeholder(f"Waiting for {camera_name} camera...")

                ok, jpg = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                )
                if ok:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                    )

                time.sleep(self.viewer_period)

        @app.route("/")
        def index():
            return render_template_string(VIEWER_HTML)

        @app.route("/stream/<camera_name>")
        def stream(camera_name):
            if camera_name not in ("front", "wrist", "composed"):
                return "Unknown camera", 404
            return Response(
                generate_stream(camera_name),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        def run_app():
            app.run(
                host=self.viewer_host,
                port=self.viewer_port,
                threaded=True,
                debug=False,
                use_reloader=False,
            )

        thread = threading.Thread(target=run_app, daemon=True)
        thread.start()
        self.get_logger().info(
            f"Viewer running on http://{self.viewer_host}:{self.viewer_port}"
        )

    def _compose_scene(self) -> Optional[np.ndarray]:
        # Fetch the most recent front and wrist frames.
        front, wrist = self._get_frames()

        if front is None and wrist is None:
            # If both streams are missing, there is no scene to analyze.
            return None

        if front is None:
            # Use a placeholder if the front camera has not produced frames yet.
            front = make_placeholder("Waiting for front camera...")
        if wrist is None:
            # Use a placeholder if the wrist camera has not produced frames yet.
            wrist = make_placeholder("Waiting for wrist camera...")

        # Normalize both frames to the same resolution.
        front = cv2.resize(front, (640, 480))
        wrist = cv2.resize(wrist, (640, 480))

        # Label each frame and place them side by side.
        composed = np.hstack([label_frame(front, "Front camera"), label_frame(wrist, "Wrist camera")])
        # Return the final scene image.
        return composed

    def _save_scene_image(self, scene: np.ndarray) -> str:
        # Create a temporary PNG file that the VLM processor can read.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # Write the composed scene to disk.
            cv2.imwrite(tmp.name, scene)
            # Return the temporary file path.
            return tmp.name

    def _build_prompt(self, request: dict) -> str:
        # Build the textual instruction that guides the VLM output.
        message = request.get("message") or "No additional context."
        task_description = request.get("task") or ""
        task_line = f"Task description: {task_description}\n" if task_description else ""
        if self.include_reasoning:
            return (
                "You are a robotic task verifier. Look at the live camera scene and decide whether the requested condition is satisfied. "
                f"Check name: {request['skill_name']}\n"
                f"Attempt id: {request['attempt_id']}\n"
                f"{task_line}"
                f"BT message: {message}\n\n"
                "Return a STATUS line followed by a REASON section:\n"
                "STATUS=<PENDING|RUNNING|WAIT_HUMAN|MANUAL_INTERVENTION_REQUIRED|SUCCESS|FAILURE>\n"
                "REASON=<short explanation; may span multiple lines>\n"
                "Use PENDING or RUNNING if the condition is not yet satisfied or the scene does not provide enough evidence.\n"
                "Use FAILURE if the condition is clearly not satisfied in the current attempt; the robot will retry from the beginning.\n"
                "Use SUCCESS only if the condition is fully satisfied as requested."
            )

        return (
            "You are a robotic task verifier. Look at the live camera scene and decide whether the requested condition is satisfied. "
            f"Check name: {request['skill_name']}\n"
            f"Attempt id: {request['attempt_id']}\n"
            f"{task_line}"
            f"BT message: {message}\n\n"
            "Return exactly one token: PENDING, RUNNING, WAIT_HUMAN, MANUAL_INTERVENTION_REQUIRED, SUCCESS, or FAILURE.\n"
            "Use PENDING or RUNNING if the condition is not yet satisfied or the scene does not provide enough evidence.\n"
            "Use FAILURE if the condition is clearly not satisfied in the current attempt; the robot will retry from the beginning.\n"
            "Use SUCCESS only if the condition is fully satisfied as requested."
        )

    def _coerce_status(self, status: str, allowed_statuses: list) -> str:
        if status in allowed_statuses:
            return status
        if STATUS_RUNNING in allowed_statuses:
            return STATUS_RUNNING
        if STATUS_FAILURE in allowed_statuses:
            return STATUS_FAILURE
        if STATUS_SUCCESS in allowed_statuses:
            return STATUS_SUCCESS
        if STATUS_WAIT_HUMAN in allowed_statuses:
            return STATUS_WAIT_HUMAN
        if STATUS_MANUAL_INTERVENTION_REQUIRED in allowed_statuses:
            return STATUS_MANUAL_INTERVENTION_REQUIRED
        if STATUS_PENDING in allowed_statuses:
            return STATUS_PENDING
        return STATUS_RUNNING

    def _parse_status(self, text: str) -> str:
        # Normalize the output so matching is case-insensitive.
        upper = text.upper()
        match = re.search(
            r"STATUS\s*[:=]\s*(PENDING|RUNNING|WAIT_HUMAN|MANUAL_INTERVENTION_REQUIRED|SUCCESS|FAILURE)",
            upper,
        )
        if match:
            return match.group(1)
        if STATUS_SUCCESS in upper:
            return STATUS_SUCCESS
        if STATUS_FAILURE in upper or "FAILED" in upper:
            return STATUS_FAILURE
        if STATUS_MANUAL_INTERVENTION_REQUIRED in upper or "MANUAL_INTERVENTION" in upper:
            return STATUS_MANUAL_INTERVENTION_REQUIRED
        if STATUS_WAIT_HUMAN in upper or "WAIT HUMAN" in upper or "HUMAN_HELP" in upper:
            return STATUS_WAIT_HUMAN
        if STATUS_PENDING in upper:
            return STATUS_PENDING
        if STATUS_RUNNING in upper or "STILL_RUNNING" in upper or "STILL RUNNING" in upper:
            return STATUS_RUNNING
        # Default to running when the output is ambiguous.
        return STATUS_RUNNING

    def _run_vlm(self, request: dict) -> Tuple[str, str]:
        # Build the current scene from the latest camera frames.
        scene = self._compose_scene()
        if scene is None:
            # If no scene exists yet, keep waiting for camera data.
            return STATUS_RUNNING, "Waiting for both camera streams."

        # Save the scene to a temporary file for the multimodal processor.
        scene_path = self._save_scene_image(scene)
        try:
            # Build the chat-style message expected by the model.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": scene_path},
                        {"type": "text", "text": self._build_prompt(request)},
                    ],
                }
            ]

            # Convert the message into the model's chat template format.
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Extract the image and video inputs required by the VLM pipeline.
            image_inputs, video_inputs = process_vision_info(messages)
            # Tokenize and package the multimodal inputs as PyTorch tensors.
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            # Move all tensors to the same device as the model.
            inputs = inputs.to(self.model_input_device)

            # Run generation without gradients to save memory and time.
            with torch.inference_mode():
                max_tokens = self.max_new_tokens
                if not self.include_reasoning:
                    max_tokens = min(max_tokens, 4)
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_tokens,
                )

            # Remove the prompt tokens so only newly generated tokens remain.
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            # Decode the generated token ids back into text.
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            if self.log_model_output:
                self.get_logger().info(f"VLM output: {output_text}")

            # Convert the free-form model output into one of the accepted statuses.
            status = self._parse_status(output_text)
            # Return both the normalized status and the raw model output.
            return status, output_text or "Empty model output."
        finally:
            try:
                # Remove the temporary image even if inference fails.
                Path(scene_path).unlink(missing_ok=True)
            except Exception:
                # Ignore cleanup errors because they are non-fatal.
                pass

    def _evaluation_loop(self) -> None:
        # Keep evaluating while ROS is alive and the node has not been asked to stop.
        while rclpy.ok() and not self._stop_event.is_set():
            # Read the current request to decide whether work is needed.
            request, request_id = self._get_active_request()
            if request is None:
                # Wait briefly until a request arrives, then check again.
                self._new_request_event.wait(timeout=0.5)
                self._new_request_event.clear()
                continue

            try:
                # Run the VLM on the latest scene.
                status, reason = self._run_vlm(request)
            except Exception as exc:
                # Keep the loop alive even if inference crashes.
                self.get_logger().exception("VLM inference failed")
                status, reason = STATUS_RUNNING, f"Inference error: {exc}"

            # Skip publishing if a newer request arrived during inference.
            _, current_request_id = self._get_active_request()
            if current_request_id != request_id:
                continue

            status = self._coerce_status(status, request.get("allowed_statuses", []))
            # Publish the result of this evaluation pass.
            self._set_result(request, status, reason)

            if status in (STATUS_SUCCESS, STATUS_FAILURE):
                # Clear the active request once a final decision is reached.
                with self._lock:
                    if self._active_request_id == request_id:
                        self._active_request = None
                # Clear the wake-up event so the loop waits for the next request.
                self._new_request_event.clear()
            else:
                # Sleep before checking the same request again.
                time.sleep(self.check_period_seconds)

    def destroy_node(self):
        # Ask the background thread to stop.
        self._stop_event.set()
        # Wake the thread in case it is waiting for a request.
        self._new_request_event.set()
        if hasattr(self, "worker") and self.worker.is_alive():
            # Join the worker thread briefly so shutdown is clean.
            self.worker.join(timeout=2.0)
        # Delegate the rest of the teardown to the parent ROS node.
        return super().destroy_node()


def main() -> None:
    # Initialize the ROS client library.
    rclpy.init()
    # Create the verification node.
    node = PandaVLMVerifier()
    try:
        # Enter the ROS event loop.
        rclpy.spin(node)
    finally:
        # Ensure the node is destroyed even if spin exits with an error.
        node.destroy_node()
        # Shut down ROS cleanly.
        rclpy.shutdown()


if __name__ == "__main__":
    # Run the entry point only when the file is executed directly.
    main()