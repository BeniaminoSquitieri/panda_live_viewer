"""Optional Flask-based MJPEG viewer for the live Panda camera streams."""

import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np
import rclpy

from .camera import placeholder

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


def start_view(get_frames: Callable[[], Tuple[Optional[np.ndarray], Optional[np.ndarray]]], scene_fn: Callable[[], Optional[np.ndarray]], stop_event: threading.Event, host: str, port: int, fps: float, logger) -> None:
    """Start the optional browser viewer in a daemon thread."""
    try:
        from flask import Flask, Response, render_template_string
    except Exception as exc:
        logger.warning(f"Viewer disabled (Flask import failed): {exc}")
        return
    try:
        import cv2
    except ModuleNotFoundError as exc:
        logger.warning(f"Viewer disabled (OpenCV import failed): {exc}")
        return

    app = Flask(__name__)
    frame_s = 1.0 / max(fps, 1.0)

    def generate_stream(camera_name: str):
        while rclpy.ok() and not stop_event.is_set():
            if camera_name == "composed":
                frame = scene_fn()
                if frame is None:
                    frame = placeholder("Waiting for cameras...")
            else:
                front, wrist = get_frames()
                frame = front if camera_name == "front" else wrist
                if frame is None:
                    frame = placeholder(f"Waiting for {camera_name} camera...")

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                )

            time.sleep(frame_s)

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
            host=host,
            port=port,
            threaded=True,
            debug=False,
            use_reloader=False,
        )

    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    logger.info(f"Viewer running on http://{host}:{port}")
