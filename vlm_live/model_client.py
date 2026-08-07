"""Client helpers that forward inference requests to a running model server.

Drop-in replacement for the local ``run_inference`` / ``run_text_inference``
functions in model.py. Supports this repo's Unix socket model_server and
OpenAI-compatible HTTP servers such as vLLM.
"""

import base64
import io
import json
import os
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

import numpy as np

from .prompt import log_response, parse_verdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOCKET_PATH = os.path.join(PROJECT_ROOT, "runtime", "vlm_server.sock")


def _encode_image_b64(scene: np.ndarray) -> str:
    from PIL import Image

    img = Image.fromarray(scene.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _call(socket_path: str, payload: dict, timeout: float | None = None) -> dict:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(socket_path)
            sock.sendall((json.dumps(payload) + "\n").encode())
            raw = b""
            while not raw.endswith(b"\n") and (chunk := sock.recv(65536)):
                raw += chunk
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        missing = isinstance(exc, FileNotFoundError)
        detail = f"Unix socket not found: {socket_path}" if missing else f"Model server not running at {socket_path}"
        action = "Start the model server with" if missing else "Start it with"
        raise RuntimeError(f"{detail}. {action}: python3 -m vlm_live.model_server --model-path <path>") from exc
    result = json.loads(raw.decode())
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "unknown server error"))
    return result


def _is_http_url(server_url: str) -> bool:
    return urlparse(server_url).scheme in {"http", "https"}


def _http_json(url: str, payload: dict | None = None, timeout: float = 120.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP model server error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach HTTP model server at {url}: {exc}") from exc


def _openai_model_id(server_url: str) -> str:
    models_url = server_url.rstrip("/") + "/v1/models"
    models = _http_json(models_url, timeout=10.0).get("data") or []
    if not models:
        raise RuntimeError(f"No models reported by {models_url}")
    return str(models[0]["id"])


def _run_text_inference_via_openai_http(
    server_url: str,
    scene: np.ndarray,
    prompt: str,
    tokens: int,
    log_out: bool,
    logger,
) -> str:
    model_id = _openai_model_id(server_url)
    image_data = _encode_image_b64(scene)
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": tokens,
        "temperature": 0,
    }
    data = _http_json(server_url.rstrip("/") + "/v1/chat/completions", payload=payload)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"HTTP model server returned no choices: {data}")
    message = choices[0].get("message") or {}
    text = str(message.get("content") or "").strip()

    if log_out:
        log_response(text, logger)

    return text


def run_inference_via_server(
    server_url: str,
    scene: np.ndarray,
    prompt: str,
    tokens: int,
    reasoning: bool,
    log_out: bool,
    logger,
) -> tuple[str, str]:
    """Forward an inference request to a persistent model server."""
    if _is_http_url(server_url):
        generation_limit = tokens if reasoning else min(tokens, 4)
        output_text = _run_text_inference_via_openai_http(
            server_url=server_url,
            scene=scene,
            prompt=prompt,
            tokens=generation_limit,
            log_out=log_out,
            logger=logger,
        )
        return parse_verdict(output_text)

    result = _socket_inference(server_url, scene, "infer", prompt, tokens, log_out, reasoning=reasoning)
    return result["status"], result["reason"]


def run_text_inference_via_server(
    server_url: str,
    scene: np.ndarray,
    prompt: str,
    tokens: int,
    log_out: bool,
    logger,
) -> str:
    """Forward a text inference request to a persistent model server."""
    if _is_http_url(server_url):
        return _run_text_inference_via_openai_http(
            server_url=server_url,
            scene=scene,
            prompt=prompt,
            tokens=tokens,
            log_out=log_out,
            logger=logger,
        )

    result = _socket_inference(server_url, scene, "text_infer", prompt, tokens, log_out)
    return result["text"]


def _socket_inference(server_url: str, scene: np.ndarray, cmd: str, prompt: str, tokens: int, log_out: bool, **extra) -> dict:
    payload = {"cmd": cmd, "image_b64": _encode_image_b64(scene), "prompt": prompt, "tokens": tokens, "log_out": log_out, **extra}
    return _call(server_url, payload)
