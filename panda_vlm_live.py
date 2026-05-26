"""ROS 2 node that uses a VLM to verify whether a Panda robot task is complete."""

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
# ROS topic where the task command is received.
COMMAND_TOPIC = "/panda/vlm/request"
# ROS topic where the verification result is published.
STATUS_TOPIC = "/panda/vlm/status"
# Time in seconds between repeated checks while the task is still running.
CHECK_PERIOD_SECONDS = 1.0
# Status value used when the task is complete.
STATUS_SUCCESS = "SUCCESS"
# Status value used when the task can no longer succeed.
STATUS_FAILED = "FAILED"
# Status value used when the task is not yet decided.
STATUS_STILL_RUNNING = "STILL_RUNNING"


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
        super().__init__("panda_vlm_verifier")

        # Declare configurable ROS parameters with default values.
        self.declare_parameter("model_path", MODEL_PATH)
        self.declare_parameter("front_topic", FRONT_TOPIC)
        self.declare_parameter("wrist_topic", WRIST_TOPIC)
        self.declare_parameter("command_topic", COMMAND_TOPIC)
        self.declare_parameter("status_topic", STATUS_TOPIC)
        self.declare_parameter("check_period_seconds", CHECK_PERIOD_SECONDS)

        # Read the effective parameter values after ROS overrides.
        self.model_path = self.get_parameter("model_path").value
        self.front_topic = self.get_parameter("front_topic").value
        self.wrist_topic = self.get_parameter("wrist_topic").value
        self.command_topic = self.get_parameter("command_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.check_period_seconds = float(self.get_parameter("check_period_seconds").value)

        # Protect shared state accessed by the ROS callbacks and worker thread.
        self._lock = threading.Lock()
        # Signal used to stop the background evaluation loop.
        self._stop_event = threading.Event()
        # Signal used to wake the worker when a new command arrives.
        self._new_command_event = threading.Event()
        # Store the most recent frame from each camera.
        self._latest_frames = {"front": None, "wrist": None}
        # Store the command currently being evaluated.
        self._active_command = ""
        # Keep the latest published status in memory.
        self._latest_status = STATUS_STILL_RUNNING
        # Keep the latest human-readable reason in memory.
        self._latest_reason = "Waiting for a command."

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
        # Subscribe to the command topic that triggers verification.
        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self._command_callback,
            10,
        )
        # Create a publisher for the verification status.
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

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

        # Log the topics the node is listening to.
        self.get_logger().info(
            f"Ready. Listening on {self.front_topic}, {self.wrist_topic}, and {self.command_topic}"
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

    def _command_callback(self, msg: String) -> None:
        # Remove leading and trailing whitespace from the incoming command.
        command = msg.data.strip()
        if not command:
            # Ignore empty commands.
            return

        # Update shared command state safely.
        with self._lock:
            self._active_command = command
            self._latest_status = STATUS_STILL_RUNNING
            self._latest_reason = "New command received."

        # Log the received command for traceability.
        self.get_logger().info(f"Received command: {command}")
        # Wake the background loop so it can evaluate the command immediately.
        self._new_command_event.set()

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

    def _get_active_command(self) -> str:
        # Read the current command under lock.
        with self._lock:
            return self._active_command

    def _set_result(self, status: str, reason: str) -> None:
        # Update the cached result under lock.
        with self._lock:
            self._latest_status = status
            self._latest_reason = reason

        # Publish the coarse status over ROS.
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        # Log the detailed reason locally.
        self.get_logger().info(f"{status}: {reason}")

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

    def _build_prompt(self, command: str) -> str:
        # Build the textual instruction that guides the VLM output.
        return (
            "You are a robotic task verifier. Look at the live camera scene and decide whether the task is complete. "
            f"Task: {command}\n\n"
            "Return exactly one token: SUCCESS, FAILED, or STILL_RUNNING.\n"
            "Use STILL_RUNNING if the task is not finished yet or the scene does not provide enough evidence.\n"
            "Use FAILED only if there is clear evidence that the task cannot succeed anymore.\n"
            "Use SUCCESS only if the task is fully completed as requested."
        )

    def _parse_status(self, text: str) -> str:
        # Normalize the output so matching is case-insensitive.
        upper = text.upper()
        if STATUS_SUCCESS in upper:
            return STATUS_SUCCESS
        if STATUS_FAILED in upper:
            return STATUS_FAILED
        if STATUS_STILL_RUNNING in upper:
            return STATUS_STILL_RUNNING
        # Default to still running when the output is ambiguous.
        return STATUS_STILL_RUNNING

    def _run_vlm(self, command: str) -> Tuple[str, str]:
        # Build the current scene from the latest camera frames.
        scene = self._compose_scene()
        if scene is None:
            # If no scene exists yet, keep waiting for camera data.
            return STATUS_STILL_RUNNING, "Waiting for both camera streams."

        # Save the scene to a temporary file for the multimodal processor.
        scene_path = self._save_scene_image(scene)
        try:
            # Build the chat-style message expected by the model.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": scene_path},
                        {"type": "text", "text": self._build_prompt(command)},
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
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=16,
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
            # Read the current command to decide whether work is needed.
            command = self._get_active_command()
            if not command:
                # Wait briefly until a command arrives, then check again.
                self._new_command_event.wait(timeout=0.5)
                self._new_command_event.clear()
                continue

            try:
                # Run the VLM on the latest scene.
                status, reason = self._run_vlm(command)
            except Exception as exc:
                # Keep the loop alive even if inference crashes.
                self.get_logger().exception("VLM inference failed")
                status, reason = STATUS_STILL_RUNNING, f"Inference error: {exc}"

            # Publish the result of this evaluation pass.
            self._set_result(status, reason)

            if status in (STATUS_SUCCESS, STATUS_FAILED):
                # Clear the active command once a final decision is reached.
                with self._lock:
                    self._active_command = ""
                # Clear the wake-up event so the loop waits for the next command.
                self._new_command_event.clear()
            else:
                # Sleep before checking the same command again.
                time.sleep(self.check_period_seconds)

    def destroy_node(self):
        # Ask the background thread to stop.
        self._stop_event.set()
        # Wake the thread in case it is waiting for a command.
        self._new_command_event.set()
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