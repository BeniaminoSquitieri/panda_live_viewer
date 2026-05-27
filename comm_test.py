#!/usr/bin/env python3
"""Test ROS2 cross-machine communication."""

import argparse, json, sys, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

TOPIC = "/bt_comm_test"

class PubNode(Node):
    def __init__(self):
        super().__init__("comm_test_pub")
        self.pub = self.create_publisher(String, TOPIC, 10)
        self.count = 0
        self.timer = self.create_timer(1.0, self._tick)
        self.get_logger().info(f"PUBLISHER ready on {TOPIC}")
    def _tick(self):
        self.count += 1
        msg = String()
        msg.data = json.dumps({"host": "server", "count": self.count, "time": time.time()})
        self.pub.publish(msg)
        self.get_logger().info(f"PUB #{self.count}")

class SubNode(Node):
    def __init__(self):
        super().__init__("comm_test_sub")
        self.sub = self.create_subscription(String, TOPIC, self._on_msg, 10)
        self.get_logger().info(f"SUBSCRIBER ready on {TOPIC}")
    def _on_msg(self, msg: String):
        try: data = json.loads(msg.data)
        except: data = {"raw": msg.data}
        self.get_logger().info(f"RECV: {json.dumps(data)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pub", action="store_true")
    parser.add_argument("--sub", action="store_true")
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()
    rclpy.init()
    node = PubNode() if args.pub else SubNode()
    try:
        for _ in range(args.timeout * 2):
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
