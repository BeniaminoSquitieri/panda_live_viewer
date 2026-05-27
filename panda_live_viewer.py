import threading
import time

import cv2
import numpy as np
import rclpy
from flask import Flask, Response, render_template_string
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


# Initialize Flask application
app = Flask(__name__)

# Dictionary to store the most recent frames from each camera
latest_frames = {
    "front": None,
    "wrist": None,
}

# Dictionary to track the timestamp of the last received frame
latest_times = {
    "front": 0.0,
    "wrist": 0.0,
}

# Lock to ensure thread-safe read/write of frames
lock = threading.Lock()


HTML = """
<!doctype html>
<html>
<head>
  <title>Panda Live Cameras</title>
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
    code {
      color: #9ae6b4;
    }
  </style>
</head>
<body>
  <h1>Panda Live Cameras</h1>
  <p>ROS 2 subscriber server + MJPEG web viewer</p>

  <div class="grid">
    <div class="card">
      <h2>Front camera</h2>
      <img src="/stream/front">
      <p>Topic: <code>/panda/camera/front/image_compressed</code></p>
    </div>
    <div class="card">
      <h2>Wrist camera</h2>
      <img src="/stream/wrist">
      <p>Topic: <code>/panda/camera/wrist/image_compressed</code></p>
    </div>
  </div>
</body>
</html>
"""


# ROS 2 Node for subscribing to camera streams
class PandaImageSubscriber(Node):
    def __init__(self):
        super().__init__("panda_live_viewer_subscriber")

        self.front_sub = self.create_subscription(
            CompressedImage,
            "/panda/camera/front/image_compressed",
            lambda msg: self.image_callback(msg, "front"),
            10,
        )

        self.wrist_sub = self.create_subscription(
            CompressedImage,
            "/panda/camera/wrist/image_compressed",
            lambda msg: self.image_callback(msg, "wrist"),
            10,
        )

        self.get_logger().info("Subscribed to Panda compressed camera topics")

    # Callback function executed when a new image message is received
    def image_callback(self, msg, camera_name):
        # Convert compressed image data back to an OpenCV frame
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warning(f"Could not decode {camera_name} frame")
            return

        with lock:
            latest_frames[camera_name] = frame
            latest_times[camera_name] = time.time()


# Dedicated thread function to spin the ROS 2 node
def ros_thread():
    rclpy.init()
    node = PandaImageSubscriber()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# Route for the main dashboard
@app.route("/")
def index():
    return render_template_string(HTML)


# Generator function to continuously yield JPEG frames for the MJPEG stream
def generate_stream(camera_name):
    while True:
        with lock:
            frame = latest_frames.get(camera_name)

        if frame is None:
            # If no frame is received yet, display a blank placeholder image
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                f"Waiting for {camera_name} camera...",
                (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            frame_to_send = blank
        else:
            frame_to_send = frame

        ok, jpg = cv2.imencode(".jpg", frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            )

        time.sleep(0.05)


# Route to serve the MJPEG video stream for a specific camera
@app.route("/stream/<camera_name>")
def stream(camera_name):
    if camera_name not in latest_frames:
        return "Unknown camera", 404
    return Response(
        generate_stream(camera_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# Main execution block
if __name__ == "__main__":
    import sys

    # Use port from command line arguments or default to 8080
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

    # Start the ROS 2 node in a background daemon thread
    t = threading.Thread(target=ros_thread, daemon=True)
    t.start()

    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, threaded=True)
