"""Image helpers for decoding, labeling, composing, and saving camera frames."""

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sensor_msgs.msg import CompressedImage


def _cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV (cv2) is required for camera image processing.") from exc
    return cv2


def decode_image(msg: CompressedImage) -> Optional[np.ndarray]:
    """Decode a ROS compressed image message into a BGR OpenCV frame."""
    cv2 = _cv2()
    np_arr = np.frombuffer(msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def placeholder(label: str, size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """Create a black placeholder frame with a centered status label."""
    cv2 = _cv2()
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        image,
        label,
        (40, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    return image


def label(frame: np.ndarray, title: str) -> np.ndarray:
    """Return a copy of the frame with a visible title overlay."""
    cv2 = _cv2()
    output = frame.copy()
    cv2.putText(
        output,
        title,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    return output


def compose(front: Optional[np.ndarray], wrist: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Build a side-by-side scene from the front and wrist cameras."""
    cv2 = _cv2()
    if front is None and wrist is None:
        return None

    if front is None:
        front = placeholder("Waiting for front camera...")
    if wrist is None:
        wrist = placeholder("Waiting for wrist camera...")

    front = cv2.resize(front, (640, 480))
    wrist = cv2.resize(wrist, (640, 480))
    return np.hstack([label(front, "Front camera"), label(wrist, "Wrist camera")])


def save_image(scene: np.ndarray) -> str:
    """Persist a scene to a temporary PNG file and return the file path."""
    cv2 = _cv2()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, scene)
        return tmp.name


def cleanup(path: str) -> None:
    """Remove a temporary file if it still exists."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass
