"""ROS2 perception node that publishes scene_facts JSON."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String

from bt_planning.scene_facts import build_scene_facts_stub
from vlm_live.camera import decode_image
from vlm_live.const import FRONT_TOPIC, WRIST_TOPIC

from .experiment_log import append_perception_event, build_perception_event
from .geometry import (
    CameraIntrinsics,
    camera_info_to_intrinsics,
    decode_depth_image,
    quaternion_multiply_xyzw,
    rotate_covariance_6x6_xyzw,
    rotate_vector_xyzw,
    stamp_to_float,
)
from .pipeline import PerceptionPipeline
from .segmenters import SegmenterConfig, build_segmenter

try:
    from lerobot_bt_interfaces.srv import QueryObjectPose
except ImportError:
    QueryObjectPose = None


DEFAULT_FRONT_DEPTH_TOPIC = "/panda/camera/front/depth"
DEFAULT_WRIST_DEPTH_TOPIC = "/panda/camera/wrist/depth"
DEFAULT_FRONT_CAMERA_INFO_TOPIC = "/panda/camera/front/camera_info"
DEFAULT_WRIST_CAMERA_INFO_TOPIC = "/panda/camera/wrist/camera_info"
DEFAULT_SCENE_FACTS_TOPIC = "/perception/scene_facts"
DEFAULT_QUERY_POSE_SERVICE = "/perception/query_pose"


@dataclass
class _CameraCache:
    rgb: np.ndarray | None = None
    rgb_stamp: float | None = None
    depth: np.ndarray | None = None
    depth_stamp: float | None = None
    intrinsics: CameraIntrinsics | None = None
    camera_info_stamp: float | None = None
    frame_id: str = ""


class PerceptionNode(Node):
    """Passive RGB-D perception bridge for planner scene facts and pose gates."""

    def __init__(self) -> None:
        super().__init__("panda_perception_scene_facts")
        self.callback_group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.latest_scene_facts: dict[str, Any] | None = None
        self.tf_buffer = None
        self.tf_listener = None

        self._declare_parameters()
        self._load_parameters()
        self.cameras = {
            name: _CameraCache(frame_id=name) for name in self.enabled_cameras
        }
        self.pipeline = PerceptionPipeline(
            segmenter=build_segmenter(self.segmenter_config, self.registry),
            min_depth_points=self.min_depth_points,
            pose_max_depth_m=self.pose_max_depth_m,
            pose_depth_band_m=self.pose_depth_band_m,
        )
        self._init_tf()
        self._init_ros_interfaces()
        self.get_logger().info(
            f"Perception scene facts ready on {self.scene_facts_topic}; "
            f"query service={self._query_service_status()}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("front_image_topic", FRONT_TOPIC)
        self.declare_parameter("wrist_image_topic", WRIST_TOPIC)
        self.declare_parameter("front_depth_topic", DEFAULT_FRONT_DEPTH_TOPIC)
        self.declare_parameter("wrist_depth_topic", DEFAULT_WRIST_DEPTH_TOPIC)
        self.declare_parameter("front_camera_info_topic", DEFAULT_FRONT_CAMERA_INFO_TOPIC)
        self.declare_parameter("wrist_camera_info_topic", DEFAULT_WRIST_CAMERA_INFO_TOPIC)
        self.declare_parameter("scene_facts_topic", DEFAULT_SCENE_FACTS_TOPIC)
        self.declare_parameter("query_pose_service", DEFAULT_QUERY_POSE_SERVICE)
        self.declare_parameter("publish_period_s", 0.5)
        self.declare_parameter("max_sync_delta_s", 0.15)
        self.declare_parameter("target_frame_id", "base_link")
        self.declare_parameter("planner_registry_json", "")
        self.declare_parameter("segmenter_backend", "owlvit")
        self.declare_parameter("segmenter_model_path", "google/owlvit-base-patch32")
        self.declare_parameter("segmenter_score_threshold", 0.2)
        self.declare_parameter("segmenter_mask_mode", "grabcut")
        self.declare_parameter("segmenter_image_color_order", "bgr")
        self.declare_parameter("min_depth_points", 25)
        self.declare_parameter("perception_log_path", "")
        self.declare_parameter("require_query_pose_service", True)
        self.declare_parameter("enabled_cameras", ["front", "wrist"])
        self.declare_parameter("pose_max_depth_m", 0.0)
        self.declare_parameter("pose_depth_band_m", 0.0)

    def _load_parameters(self) -> None:
        self.image_topics = {
            "front": self.get_parameter("front_image_topic").value,
            "wrist": self.get_parameter("wrist_image_topic").value,
        }
        self.depth_topics = {
            "front": self.get_parameter("front_depth_topic").value,
            "wrist": self.get_parameter("wrist_depth_topic").value,
        }
        self.camera_info_topics = {
            "front": self.get_parameter("front_camera_info_topic").value,
            "wrist": self.get_parameter("wrist_camera_info_topic").value,
        }
        self.scene_facts_topic = str(self.get_parameter("scene_facts_topic").value)
        self.query_pose_service = str(self.get_parameter("query_pose_service").value)
        self.publish_period_s = float(self.get_parameter("publish_period_s").value)
        self.max_sync_delta_s = float(self.get_parameter("max_sync_delta_s").value)
        self.target_frame_id = str(self.get_parameter("target_frame_id").value)
        self.registry = self._load_registry(str(self.get_parameter("planner_registry_json").value))
        self.segmenter_config = SegmenterConfig(
            backend=str(self.get_parameter("segmenter_backend").value),
            model_path=str(self.get_parameter("segmenter_model_path").value),
            score_threshold=float(self.get_parameter("segmenter_score_threshold").value),
            mask_mode=str(self.get_parameter("segmenter_mask_mode").value),
            image_color_order=str(self.get_parameter("segmenter_image_color_order").value),
        )
        self.min_depth_points = int(self.get_parameter("min_depth_points").value)
        self.perception_log_path = str(self.get_parameter("perception_log_path").value)
        self.require_query_pose_service = bool(self.get_parameter("require_query_pose_service").value)
        requested = [str(name).strip() for name in (self.get_parameter("enabled_cameras").value or [])]
        valid = [name for name in ("front", "wrist") if name in requested]
        self.enabled_cameras = valid or ["front", "wrist"]
        if self.enabled_cameras != ["front", "wrist"]:
            self.get_logger().info(f"Perception restricted to cameras: {self.enabled_cameras}")
        self.pose_max_depth_m = float(self.get_parameter("pose_max_depth_m").value)
        self.pose_depth_band_m = float(self.get_parameter("pose_depth_band_m").value)

    def _init_tf(self) -> None:
        try:
            from tf2_ros import Buffer, TransformListener
        except Exception as exc:
            self.get_logger().warning(f"tf2_ros unavailable; poses will remain camera-frame only: {exc}")
            return
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    @staticmethod
    def _load_registry(raw_value: str) -> dict[str, Any]:
        raw_value = (raw_value or "").strip()
        if not raw_value:
            return {}
        # Accept either a path to a JSON file or an inline JSON object string.
        if not raw_value.startswith(("{", "[")):
            registry_path = os.path.expanduser(raw_value)
            if os.path.isfile(registry_path):
                with open(registry_path, "r", encoding="utf-8") as handle:
                    raw_value = handle.read().strip()
            else:
                raise FileNotFoundError(
                    f"planner_registry_json '{registry_path}' is not a file and is not inline JSON."
                )
        payload = json.loads(raw_value)
        if not isinstance(payload, dict):
            raise ValueError("planner_registry_json must be a JSON object.")
        return payload

    def _init_ros_interfaces(self) -> None:
        self.camera_subscriptions = []
        for camera_name in self.enabled_cameras:
            self.camera_subscriptions.append(
                self.create_subscription(
                    CompressedImage,
                    self.image_topics[camera_name],
                    lambda msg, name=camera_name: self._on_rgb(msg, name),
                    10,
                    callback_group=self.callback_group,
                )
            )
            self.camera_subscriptions.append(
                self.create_subscription(
                    Image,
                    self.depth_topics[camera_name],
                    lambda msg, name=camera_name: self._on_depth(msg, name),
                    10,
                    callback_group=self.callback_group,
                )
            )
            self.camera_subscriptions.append(
                self.create_subscription(
                    CameraInfo,
                    self.camera_info_topics[camera_name],
                    lambda msg, name=camera_name: self._on_camera_info(msg, name),
                    10,
                    callback_group=self.callback_group,
                )
            )

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.scene_facts_pub = self.create_publisher(String, self.scene_facts_topic, latched_qos)
        self.query_pose_srv = None
        if QueryObjectPose is None:
            message = (
                "QueryObjectPose service type is unavailable; "
                "source the lerobot ROS workspace after rebuilding lerobot_bt_interfaces."
            )
            if self.require_query_pose_service:
                raise RuntimeError(message)
            self.get_logger().warning(message)
        else:
            self.query_pose_srv = self.create_service(
                QueryObjectPose,
                self.query_pose_service,
                self._on_query_pose,
                callback_group=self.callback_group,
            )
        self.publish_timer = self.create_timer(self.publish_period_s, self._publish_scene_facts)

    def _query_service_status(self) -> str:
        if self.query_pose_srv is None:
            return "disabled"
        return self.query_pose_service

    def _on_rgb(self, msg: CompressedImage, camera_name: str) -> None:
        frame = decode_image(msg)
        if frame is None:
            self.get_logger().warning(f"Could not decode {camera_name} RGB frame")
            return
        stamp = stamp_to_float(msg.header.stamp)
        with self.lock:
            cache = self.cameras[camera_name]
            cache.rgb = frame
            cache.rgb_stamp = stamp
            if msg.header.frame_id:
                cache.frame_id = msg.header.frame_id

    def _on_depth(self, msg: Image, camera_name: str) -> None:
        try:
            depth = decode_depth_image(msg)
        except ValueError as exc:
            self.get_logger().warning(f"Could not decode {camera_name} depth frame: {exc}")
            return
        stamp = stamp_to_float(msg.header.stamp)
        with self.lock:
            cache = self.cameras[camera_name]
            cache.depth = depth
            cache.depth_stamp = stamp
            if msg.header.frame_id:
                cache.frame_id = msg.header.frame_id

    def _on_camera_info(self, msg: CameraInfo, camera_name: str) -> None:
        try:
            intrinsics = camera_info_to_intrinsics(msg)
        except ValueError as exc:
            self.get_logger().warning(f"Could not parse {camera_name} CameraInfo: {exc}")
            return
        stamp = stamp_to_float(msg.header.stamp)
        with self.lock:
            cache = self.cameras[camera_name]
            cache.intrinsics = intrinsics
            cache.camera_info_stamp = stamp
            if msg.header.frame_id:
                cache.frame_id = msg.header.frame_id

    def _snapshot(self) -> dict[str, _CameraCache]:
        with self.lock:
            return {
                name: _CameraCache(
                    rgb=None if cache.rgb is None else cache.rgb.copy(),
                    rgb_stamp=cache.rgb_stamp,
                    depth=None if cache.depth is None else cache.depth.copy(),
                    depth_stamp=cache.depth_stamp,
                    intrinsics=cache.intrinsics,
                    camera_info_stamp=cache.camera_info_stamp,
                    frame_id=cache.frame_id,
                )
                for name, cache in self.cameras.items()
            }

    def _synchronized_depth(self, camera_name: str, cache: _CameraCache, warnings: list[str]) -> np.ndarray | None:
        if cache.rgb_stamp is None or cache.depth_stamp is None:
            return None
        if abs(cache.rgb_stamp - cache.depth_stamp) > self.max_sync_delta_s:
            warnings.append(f"{camera_name}_rgb_depth_desynchronized")
            return None
        return cache.depth

    def _lookup_transform(self, source_frame_id: str) -> Any | None:
        if not self.target_frame_id or not source_frame_id or self.target_frame_id == source_frame_id:
            return None
        if self.tf_buffer is None:
            return None
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame_id,
                source_frame_id,
                Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return None

    def _tf_available(self, source_frame_id: str) -> bool:
        if not self.target_frame_id or not source_frame_id or self.target_frame_id == source_frame_id:
            return True
        return self._lookup_transform(source_frame_id) is not None

    @staticmethod
    def _transform_fact(fact: dict[str, Any], transform: Any, target_frame_id: str) -> dict[str, Any]:
        pose = fact.get("pose")
        if not pose:
            fact["frame_id"] = target_frame_id
            return fact

        tf_translation = transform.transform.translation
        tf_rotation = transform.transform.rotation
        q_tf = {
            "x": float(tf_rotation.x),
            "y": float(tf_rotation.y),
            "z": float(tf_rotation.z),
            "w": float(tf_rotation.w),
        }
        translation = {key: float(value) for key, value in pose["translation"].items()}
        rotated = rotate_vector_xyzw(translation, q_tf)
        pose["translation"] = {
            "x": rotated["x"] + float(tf_translation.x),
            "y": rotated["y"] + float(tf_translation.y),
            "z": rotated["z"] + float(tf_translation.z),
        }
        pose["quaternion_xyzw"] = quaternion_multiply_xyzw(
            q_tf,
            {key: float(value) for key, value in pose["quaternion_xyzw"].items()},
        )
        if "covariance" in fact:
            try:
                fact["covariance"] = rotate_covariance_6x6_xyzw(
                    [float(value) for value in fact["covariance"]],
                    q_tf,
                )
            except ValueError:
                fact.setdefault("warnings", []).append("covariance_transform_failed")
        fact["frame_id"] = target_frame_id
        return fact

    def _lift_facts_to_target(
        self,
        *,
        source_frame_id: str,
        facts: dict[str, Any],
        warnings: list[str],
    ) -> bool:
        if not self.target_frame_id or self.target_frame_id == source_frame_id:
            return True

        transform = self._lookup_transform(source_frame_id)
        if transform is None:
            warnings.append("tf_unavailable")
            for fact in facts.values():
                fact.pop("pose", None)
                fact.pop("covariance", None)
                fact["frame_id"] = self.target_frame_id
                fact["pose_confidence"] = 0.0
                fact.setdefault("warnings", []).append("tf_unavailable")
            return False

        for fact in facts.values():
            self._transform_fact(fact, transform, self.target_frame_id)
        return True

    def _run_camera(self, camera_name: str, cache: _CameraCache) -> tuple[dict[str, Any], list[str], int, bool]:
        warnings: list[str] = []
        depth = self._synchronized_depth(camera_name, cache, warnings)
        stamp = cache.rgb_stamp
        source_frame_id = cache.frame_id or camera_name
        start = time.perf_counter()
        result = self.pipeline.run(
            rgb=cache.rgb,
            depth=depth,
            intrinsics=cache.intrinsics,
            registry=self.registry,
            frame_id=source_frame_id,
            stamp=stamp,
        )
        latency_s = time.perf_counter() - start
        warnings.extend(result.warnings)
        tf_available = self._lift_facts_to_target(
            source_frame_id=source_frame_id,
            facts=result.facts,
            warnings=warnings,
        )
        if self.perception_log_path:
            append_perception_event(
                self.perception_log_path,
                build_perception_event(
                    camera_name=camera_name,
                    frame_id=self.target_frame_id if tf_available else source_frame_id,
                    stamp=stamp,
                    facts=result.facts,
                    warnings=warnings,
                    detections_seen=result.detections_seen,
                    latency_s=latency_s,
                ),
            )
        return result.facts, warnings, result.detections_seen, tf_available

    @staticmethod
    def _merge_facts(existing: dict[str, Any], incoming: dict[str, Any], warnings: list[str]) -> None:
        for name, fact in incoming.items():
            current = existing.get(name)
            if current is None:
                existing[name] = fact
                continue
            warnings.append(f"ambiguous_multiple_camera_observations:{name}")
            if float(fact.get("pose_confidence", 0.0)) > float(current.get("pose_confidence", 0.0)):
                existing[name] = fact

    def _publish_scene_facts(self) -> None:
        snapshot = self._snapshot()
        facts: dict[str, Any] = {}
        warnings: list[str] = []
        tf_available_by_camera: dict[str, bool] = {}

        for camera_name, cache in snapshot.items():
            camera_facts, camera_warnings, _, tf_available = self._run_camera(camera_name, cache)
            tf_available_by_camera[camera_name] = tf_available
            warnings.extend(f"{camera_name}:{warning}" for warning in camera_warnings)
            self._merge_facts(facts, camera_facts, warnings)

        camera_details = {
            name: {
                "frame_id": cache.frame_id or name,
                "depth_available": cache.depth is not None,
                "camera_info_available": cache.intrinsics is not None,
                "tf_available": tf_available_by_camera.get(name, self._tf_available(cache.frame_id or name)),
            }
            for name, cache in snapshot.items()
        }
        scene_facts = build_scene_facts_stub(
            front_available="front" in snapshot and snapshot["front"].rgb is not None,
            wrist_available="wrist" in snapshot and snapshot["wrist"].rgb is not None,
            timestamps={
                "front": snapshot["front"].rgb_stamp if "front" in snapshot else None,
                "wrist": snapshot["wrist"].rgb_stamp if "wrist" in snapshot else None,
            },
            camera_details=camera_details,
            facts=facts,
            warnings=warnings,
        )
        with self.lock:
            self.latest_scene_facts = scene_facts

        msg = String()
        msg.data = json.dumps(scene_facts, sort_keys=True)
        self.scene_facts_pub.publish(msg)

    def _on_query_pose(self, request: Any, response: Any) -> Any:
        object_name = str(getattr(request, "object_name", "")).strip()
        if not object_name:
            response.success = False
            response.pose_json = ""
            response.error_message = "object_name is required."
            return response

        with self.lock:
            scene_facts = None if self.latest_scene_facts is None else dict(self.latest_scene_facts)

        if scene_facts is None:
            response.success = False
            response.pose_json = ""
            response.error_message = "No scene facts have been published yet."
            return response

        fact = dict(scene_facts.get("facts", {}).get(object_name) or {})
        if not fact or not fact.get("present") or "pose" not in fact:
            response.success = False
            response.pose_json = json.dumps(fact) if fact else ""
            response.error_message = f"No usable pose for object {object_name!r}."
            return response

        if bool(getattr(request, "require_fresh", False)):
            stamp = stamp_to_float(fact.get("stamp"))
            max_age_s = float(getattr(request, "max_age_s", 0.0) or 0.0)
            now = self.get_clock().now()
            now_s = stamp_to_float(now.to_msg())
            if stamp is None or now_s is None or max_age_s <= 0.0:
                response.success = False
                response.pose_json = json.dumps(fact, sort_keys=True)
                response.error_message = "Freshness requested but pose stamp or max_age_s is invalid."
                return response
            if now_s - stamp > max_age_s:
                response.success = False
                response.pose_json = json.dumps(fact, sort_keys=True)
                response.error_message = f"Pose for {object_name!r} is older than {max_age_s:.3f}s."
                return response

        response.success = True
        response.pose_json = json.dumps(fact, sort_keys=True)
        response.error_message = ""
        return response
