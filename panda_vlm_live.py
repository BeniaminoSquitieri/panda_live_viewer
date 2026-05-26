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


MODEL_PATH = "/home/bsquitieri-iit.local/models/Qwen3-VL-32B-Instruct"
FRONT_TOPIC = "/panda/camera/front/image_compressed"
WRIST_TOPIC = "/panda/camera/wrist/image_compressed"
COMMAND_TOPIC = "/panda/vlm/request"
STATUS_TOPIC = "/panda/vlm/status"
CHECK_PERIOD_SECONDS = 1.0
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_STILL_RUNNING = "STILL_RUNNING"


def decode_compressed_image(msg: CompressedImage) -> Optional[np.ndarray]:
    np_arr = np.frombuffer(msg.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def make_placeholder(label: str, size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        image,
        label,
        (40, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    return image


def label_frame(frame: np.ndarray, title: str) -> np.ndarray:
    output = frame.copy()
    cv2.putText(
        output,
        title,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    return output


class PandaVLMVerifier(Node):
    def __init__(self):
        super().__init__("panda_vlm_verifier")

        self.declare_parameter("model_path", MODEL_PATH)
        self.declare_parameter("front_topic", FRONT_TOPIC)
        self.declare_parameter("wrist_topic", WRIST_TOPIC)
        self.declare_parameter("command_topic", COMMAND_TOPIC)
        self.declare_parameter("status_topic", STATUS_TOPIC)
        self.declare_parameter("check_period_seconds", CHECK_PERIOD_SECONDS)

        self.model_path = self.get_parameter("model_path").value
        self.front_topic = self.get_parameter("front_topic").value
        self.wrist_topic = self.get_parameter("wrist_topic").value
        self.command_topic = self.get_parameter("command_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.check_period_seconds = float(self.get_parameter("check_period_seconds").value)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._new_command_event = threading.Event()
        self._latest_frames = {"front": None, "wrist": None}
        self._active_command = ""
        self._latest_status = STATUS_STILL_RUNNING
        self._latest_reason = "Waiting for a command."

        self.front_sub = self.create_subscription(
            CompressedImage,
            self.front_topic,
            lambda msg: self._image_callback(msg, "front"),
            10,
        )
        self.wrist_sub = self.create_subscription(
            CompressedImage,
            self.wrist_topic,
            lambda msg: self._image_callback(msg, "wrist"),
            10,
        )
        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self._command_callback,
            10,
        )
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.get_logger().info(f"Loading VLM from {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        self.model_input_device = next(self.model.parameters()).device

        self.worker = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(
            f"Ready. Listening on {self.front_topic}, {self.wrist_topic}, and {self.command_topic}"
        )

    def _image_callback(self, msg: CompressedImage, camera_name: str) -> None:
        frame = decode_compressed_image(msg)
        if frame is None:
            self.get_logger().warning(f"Could not decode {camera_name} frame")
            return

        with self._lock:
            self._latest_frames[camera_name] = frame

    def _command_callback(self, msg: String) -> None:
        command = msg.data.strip()
        if not command:
            return

        with self._lock:
            self._active_command = command
            self._latest_status = STATUS_STILL_RUNNING
            self._latest_reason = "New command received."

        self.get_logger().info(f"Received command: {command}")
        self._new_command_event.set()

    def _get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            front = self._latest_frames["front"]
            wrist = self._latest_frames["wrist"]
            front_copy = None if front is None else front.copy()
            wrist_copy = None if wrist is None else wrist.copy()
        return front_copy, wrist_copy

    def _get_active_command(self) -> str:
        with self._lock:
            return self._active_command

    def _set_result(self, status: str, reason: str) -> None:
        with self._lock:
            self._latest_status = status
            self._latest_reason = reason

        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f"{status}: {reason}")

    def _compose_scene(self) -> Optional[np.ndarray]:
        front, wrist = self._get_frames()

        if front is None and wrist is None:
            return None

        if front is None:
            front = make_placeholder("Waiting for front camera...")
        if wrist is None:
            wrist = make_placeholder("Waiting for wrist camera...")

        front = cv2.resize(front, (640, 480))
        wrist = cv2.resize(wrist, (640, 480))

        composed = np.hstack([label_frame(front, "Front camera"), label_frame(wrist, "Wrist camera")])
        return composed

    def _save_scene_image(self, scene: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, scene)
            return tmp.name

    def _build_prompt(self, command: str) -> str:
        return (
            "You are a robotic task verifier. Look at the live camera scene and decide whether the task is complete. "
            f"Task: {command}\n\n"
            "Return exactly one token: SUCCESS, FAILED, or STILL_RUNNING.\n"
            "Use STILL_RUNNING if the task is not finished yet or the scene does not provide enough evidence.\n"
            "Use FAILED only if there is clear evidence that the task cannot succeed anymore.\n"
            "Use SUCCESS only if the task is fully completed as requested."
        )

    def _parse_status(self, text: str) -> str:
        upper = text.upper()
        if STATUS_SUCCESS in upper:
            return STATUS_SUCCESS
        if STATUS_FAILED in upper:
            return STATUS_FAILED
        if STATUS_STILL_RUNNING in upper:
            return STATUS_STILL_RUNNING
        return STATUS_STILL_RUNNING

    def _run_vlm(self, command: str) -> Tuple[str, str]:
        scene = self._compose_scene()
        if scene is None:
            return STATUS_STILL_RUNNING, "Waiting for both camera streams."

        scene_path = self._save_scene_image(scene)
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": scene_path},
                        {"type": "text", "text": self._build_prompt(command)},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model_input_device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=16,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            status = self._parse_status(output_text)
            return status, output_text or "Empty model output."
        finally:
            try:
                Path(scene_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _evaluation_loop(self) -> None:
        while rclpy.ok() and not self._stop_event.is_set():
            command = self._get_active_command()
            if not command:
                self._new_command_event.wait(timeout=0.5)
                self._new_command_event.clear()
                continue

            try:
                status, reason = self._run_vlm(command)
            except Exception as exc:
                self.get_logger().exception("VLM inference failed")
                status, reason = STATUS_STILL_RUNNING, f"Inference error: {exc}"

            self._set_result(status, reason)

            if status in (STATUS_SUCCESS, STATUS_FAILED):
                with self._lock:
                    self._active_command = ""
                self._new_command_event.clear()
            else:
                time.sleep(self.check_period_seconds)

    def destroy_node(self):
        self._stop_event.set()
        self._new_command_event.set()
        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=2.0)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = PandaVLMVerifier()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()