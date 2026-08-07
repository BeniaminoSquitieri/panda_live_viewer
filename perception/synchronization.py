"""Approximate RGB-depth timestamp synchronizer without ROS dependencies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SynchronizedRgbd:
    rgb: Any
    depth: Any
    rgb_stamp: float
    depth_stamp: float
    frame_id: str

    @property
    def stamp(self) -> float:
        return max(self.rgb_stamp, self.depth_stamp)


class ApproximateRgbdSynchronizer:
    """Match the closest RGB/depth pair inside a bounded timestamp window."""

    def __init__(self, *, max_delta_s: float, queue_size: int = 8) -> None:
        if max_delta_s < 0:
            raise ValueError("max_delta_s must be >= 0.")
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1.")
        self.max_delta_s = float(max_delta_s)
        self._rgb: deque[tuple[float, Any, str]] = deque(maxlen=queue_size)
        self._depth: deque[tuple[float, Any, str]] = deque(maxlen=queue_size)

    def add_rgb(self, *, stamp: float | None, value: Any, frame_id: str) -> SynchronizedRgbd | None:
        return self._add(self._rgb, stamp, value, frame_id)

    def add_depth(self, *, stamp: float | None, value: Any, frame_id: str) -> SynchronizedRgbd | None:
        return self._add(self._depth, stamp, value, frame_id)

    def _add(self, queue: deque[tuple[float, Any, str]], stamp: float | None, value: Any, frame_id: str) -> SynchronizedRgbd | None:
        if stamp is None:
            return None
        queue.append((float(stamp), value, frame_id))
        return self._match()

    def _match(self) -> SynchronizedRgbd | None:
        if not self._rgb or not self._depth:
            return None
        delta, rgb_index, depth_index = min(
            (abs(rgb[0] - depth[0]), rgb_index, depth_index)
            for rgb_index, rgb in enumerate(self._rgb)
            for depth_index, depth in enumerate(self._depth)
        )
        if delta > self.max_delta_s:
            return None
        rgb_stamp, rgb, rgb_frame = self._rgb[rgb_index]
        depth_stamp, depth, depth_frame = self._depth[depth_index]
        for _ in range(rgb_index + 1):
            self._rgb.popleft()
        for _ in range(depth_index + 1):
            self._depth.popleft()
        return SynchronizedRgbd(
            rgb=rgb,
            depth=depth,
            rgb_stamp=rgb_stamp,
            depth_stamp=depth_stamp,
            frame_id=rgb_frame or depth_frame,
        )
