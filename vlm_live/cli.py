"""Command-line entry point for the Panda VLM live verifier."""

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .node import VlmNode


def main() -> None:
    """Initialize ROS2, run the verifier node, and shut down cleanly."""
    rclpy.init()
    node = None
    executor = MultiThreadedExecutor()
    try:
        node = VlmNode()
        executor.add_node(node)
        executor.spin()
    finally:
        if node is not None:
            node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
