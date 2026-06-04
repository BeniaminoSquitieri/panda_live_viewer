"""Persistent VLM model server — communicates via Unix domain socket.

Run once on the machine to keep the model loaded in GPU memory across
multiple ROS node restarts:

    python3 -m vlm_live.model_server --model-path /path/to/model

The server listens on a Unix socket (default: /tmp/vlm_server.sock).
No network port is opened.

Protocol: newline-delimited JSON over the socket.
  Request:  {"cmd": "infer" | "text_infer" | "health", ...fields}
  Response: {"ok": true, ...result} | {"ok": false, "error": "..."}
"""

import argparse
import base64
import io
import json
import logging
import os
import signal
import socket
import socketserver
import sys
import threading

import numpy as np

logging.basicConfig(level=logging.INFO, format="[model_server] %(levelname)s %(message)s")
log = logging.getLogger("model_server")

DEFAULT_SOCKET_PATH = "/tmp/vlm_server.sock"

# Populated at startup
_processor = None
_model = None
_device = None
_inference_lock = threading.Lock()


def _decode_image_b64(b64: str) -> np.ndarray:
    from PIL import Image

    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


def _handle_request(payload: dict) -> dict:
    cmd = payload.get("cmd")

    if cmd == "health":
        return {"ok": True, "model_loaded": _model is not None}

    if cmd == "infer":
        from .model import run_inference

        try:
            scene = _decode_image_b64(payload["image_b64"])
        except Exception as exc:
            return {"ok": False, "error": f"image decode failed: {exc}"}

        try:
            with _inference_lock:
                status, reason = run_inference(
                    scene=scene,
                    prompt=payload.get("prompt", ""),
                    processor=_processor,
                    model=_model,
                    device=_device,
                    tokens=int(payload.get("tokens", 64)),
                    reasoning=bool(payload.get("reasoning", True)),
                    log_out=bool(payload.get("log_out", False)),
                    logger=log,
                )
            return {"ok": True, "status": status, "reason": reason}
        except Exception as exc:
            log.error(f"infer failed: {exc}")
            return {"ok": False, "error": str(exc)}

    if cmd == "text_infer":
        from .model import run_text_inference

        try:
            scene = _decode_image_b64(payload["image_b64"])
        except Exception as exc:
            return {"ok": False, "error": f"image decode failed: {exc}"}

        try:
            with _inference_lock:
                text = run_text_inference(
                    scene=scene,
                    prompt=payload.get("prompt", ""),
                    processor=_processor,
                    model=_model,
                    device=_device,
                    tokens=int(payload.get("tokens", 512)),
                    log_out=bool(payload.get("log_out", False)),
                    logger=log,
                )
            return {"ok": True, "text": text}
        except Exception as exc:
            log.error(f"text_infer failed: {exc}")
            return {"ok": False, "error": str(exc)}

    return {"ok": False, "error": f"unknown command: {cmd!r}"}


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            raw = self.rfile.readline()
            if not raw:
                return
            payload = json.loads(raw.decode())
            response = _handle_request(payload)
        except Exception as exc:
            response = {"ok": False, "error": f"server error: {exc}"}
        try:
            self.wfile.write((json.dumps(response) + "\n").encode())
        except Exception:
            pass


def _load(model_path: str):
    global _processor, _model, _device
    from .model import load_model

    log.info(f"Loading model from {model_path} ...")
    _processor, _model, _device = load_model(model_path)
    log.info("Model loaded and ready.")


def main():
    parser = argparse.ArgumentParser(description="Persistent VLM Unix socket server")
    parser.add_argument("--model-path", required=True, help="Path to model directory")
    parser.add_argument(
        "--socket-path",
        default=DEFAULT_SOCKET_PATH,
        help=f"Unix socket path (default: {DEFAULT_SOCKET_PATH})",
    )
    args = parser.parse_args()

    _load(args.model_path)

    # Remove stale socket file if present
    if os.path.exists(args.socket_path):
        os.unlink(args.socket_path)

    server = socketserver.UnixStreamServer(args.socket_path, _Handler)
    os.chmod(args.socket_path, 0o600)  # owner-only access

    def _shutdown(signum, frame):
        log.info("Shutting down.")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info(f"Listening on {args.socket_path}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if os.path.exists(args.socket_path):
            os.unlink(args.socket_path)
        sys.exit(0)


if __name__ == "__main__":
    main()

