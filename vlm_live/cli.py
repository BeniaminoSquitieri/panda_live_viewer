"""Command-line entry point for the Panda VLM live verifier."""

import rclpy

from .node import VlmNode


def main() -> None:
    """Initialize ROS2, run the verifier node, and shut down cleanly."""
    rclpy.init()
    node = VlmNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
