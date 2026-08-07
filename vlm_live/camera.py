"""Image helpers for decoding, labeling, composing, and saving camera frames."""

import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image as _PILImage
from sensor_msgs.msg import CompressedImage

# Target resolution for each camera panel in the composed scene.
_PANEL_W, _PANEL_H = 640, 480


def _resize(frame: np.ndarray, w: int = _PANEL_W, h: int = _PANEL_H) -> np.ndarray:
    return np.array(_PILImage.fromarray(frame).resize((w, h), _PILImage.BILINEAR))


def _annotate(img: _PILImage.Image, text: str, position: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
    from PIL import ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text(position, text, fill=color, font=font)
    return np.array(img)


def decode_image(msg: CompressedImage) -> np.ndarray | None:
    """Decode a ROS compressed image message into an RGB numpy array."""
    try:
        return np.array(_PILImage.open(io.BytesIO(bytes(msg.data))).convert("RGB"))
    except Exception:
        return None


def placeholder(text: str, size: tuple[int, int] = (_PANEL_W, _PANEL_H)) -> np.ndarray:
    """Create a black placeholder frame with a centered status label."""
    w, h = size
    return _annotate(_PILImage.new("RGB", (w, h), (0, 0, 0)), text, (40, h // 2), (255, 255, 255))


def label(frame: np.ndarray, title: str) -> np.ndarray:
    """Return a copy of the frame with a visible title overlay."""
    return _annotate(_PILImage.fromarray(frame).copy(), title, (24, 12), (0, 255, 255))


def compose(front: np.ndarray | None, wrist: np.ndarray | None) -> np.ndarray | None:
    """Build a side-by-side scene from the front and wrist cameras."""
    if front is None and wrist is None:
        return None

    if front is None:
        front = placeholder("Waiting for front camera...")
    if wrist is None:
        wrist = placeholder("Waiting for wrist camera...")

    front = _resize(front)
    wrist = _resize(wrist)
    return np.hstack([label(front, "Front camera"), label(wrist, "Wrist camera")])


def compose_front(front: np.ndarray | None) -> np.ndarray | None:
    """Build a scene from the front camera only."""
    if front is None:
        return None
    return label(_resize(front), "Front camera")


def save_image(scene: np.ndarray) -> str:
    """Persist a scene to a temporary PNG file and return the file path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        _PILImage.fromarray(scene).save(tmp.name, format="PNG")
        return tmp.name


def cleanup(path: str) -> None:
    """Remove a temporary file if it still exists."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass
