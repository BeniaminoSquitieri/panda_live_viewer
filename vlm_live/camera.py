"""Image helpers for decoding, labeling, composing, and saving camera frames."""

import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image as _PILImage
from sensor_msgs.msg import CompressedImage

# Target resolution for each camera panel in the composed scene.
_PANEL_W, _PANEL_H = 640, 480


def _resize(frame: np.ndarray, w: int = _PANEL_W, h: int = _PANEL_H) -> np.ndarray:
    img = _PILImage.fromarray(frame)
    img = img.resize((w, h), _PILImage.BILINEAR)
    return np.array(img)


def decode_image(msg: CompressedImage) -> Optional[np.ndarray]:
    """Decode a ROS compressed image message into an RGB numpy array."""
    try:
        img = _PILImage.open(io.BytesIO(bytes(msg.data))).convert("RGB")
        return np.array(img)
    except Exception:
        return None


def placeholder(text: str, size: Tuple[int, int] = (_PANEL_W, _PANEL_H)) -> np.ndarray:
    """Create a black placeholder frame with a centered status label."""
    from PIL import ImageDraw, ImageFont

    w, h = size
    img = _PILImage.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((40, h // 2), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def label(frame: np.ndarray, title: str) -> np.ndarray:
    """Return a copy of the frame with a visible title overlay."""
    from PIL import ImageDraw, ImageFont

    img = _PILImage.fromarray(frame).copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((24, 12), title, fill=(0, 255, 255), font=font)
    return np.array(img)


def compose(front: Optional[np.ndarray], wrist: Optional[np.ndarray]) -> Optional[np.ndarray]:
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


def save_image(scene: np.ndarray) -> str:
    """Persist a scene to a temporary PNG file and return the file path."""
    img = _PILImage.fromarray(scene)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, format="PNG")
        return tmp.name


def cleanup(path: str) -> None:
    """Remove a temporary file if it still exists."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass
