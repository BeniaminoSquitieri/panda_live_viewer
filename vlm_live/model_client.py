"""Unix socket client that forwards inference requests to a running model_server.

Drop-in replacement for the local ``run_inference`` / ``run_text_inference``
functions in model.py — same signatures, but sends the scene over a Unix
socket to a persistent server process. No network port is opened.
"""

import base64
import io
import json
import socket
from typing import Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOCKET_PATH = os.path.join(PROJECT_ROOT, "runtime", "vlm_server.sock")


def _encode_image_b64(scene: np.ndarray) -> str:
    from PIL import Image

    img = Image.fromarray(scene.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _call(socket_path: str, payload: dict, timeout: float = 120.0) -> dict:
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(socket_path)
        sock.sendall((json.dumps(payload) + "\n").encode())
        raw = b""
        while not raw.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            raw += chunk
        sock.close()
    except FileNotFoundError:
        raise RuntimeError(
            f"Unix socket not found: {socket_path}. "
            "Start the model server with: python3 -m vlm_live.model_server --model-path <path>"
        )
    except ConnectionRefusedError:
        raise RuntimeError(
            f"Model server not running at {socket_path}. "
            "Start it with: python3 -m vlm_live.model_server --model-path <path>"
        )
    result = json.loads(raw.decode())
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "unknown server error"))
    return result


def run_inference_via_server(
    server_url: str,
    scene: np.ndarray,
    prompt: str,
    tokens: int,
    reasoning: bool,
    log_out: bool,
    logger,
) -> Tuple[str, str]:
    """Forward an inference request to the persistent model server via Unix socket."""
    socket_path = server_url  # server_url holds the socket path in this mode
    result = _call(
        socket_path,
        {
            "cmd": "infer",
            "image_b64": _encode_image_b64(scene),
            "prompt": prompt,
            "tokens": tokens,
            "reasoning": reasoning,
            "log_out": log_out,
        },
    )
    return result["status"], result["reason"]


def run_text_inference_via_server(
    server_url: str,
    scene: np.ndarray,
    prompt: str,
    tokens: int,
    log_out: bool,
    logger,
) -> str:
    """Forward a text inference request to the persistent model server via Unix socket."""
    socket_path = server_url  # server_url holds the socket path in this mode
    result = _call(
        socket_path,
        {
            "cmd": "text_infer",
            "image_b64": _encode_image_b64(scene),
            "prompt": prompt,
            "tokens": tokens,
            "log_out": log_out,
        },
    )
    return result["text"]

