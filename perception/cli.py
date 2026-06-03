"""Command-line entry point for the Panda perception scene facts node."""

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .node import PerceptionNode


def main() -> None:
    rclpy.init()
    node = None
    executor = MultiThreadedExecutor()
    try:
        node = PerceptionNode()
        executor.add_node(node)
        executor.spin()
    finally:
        if node is not None:
            node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
