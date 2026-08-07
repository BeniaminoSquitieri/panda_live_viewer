"""Configurable segmentation backends for perception."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .pipeline import Detection, NoopSegmenter, RegistryMapper, Segmenter


@dataclass(frozen=True)
class SegmenterConfig:
    backend: str
    model_path: str
    score_threshold: float
    mask_mode: str
    image_color_order: str


def labels_from_registry(registry: Mapping[str, Any] | None) -> list[str]:
    mapper = RegistryMapper.from_registry(registry)
    # Feed OWL-ViT the natural-language aliases (e.g. "coffee capsule") instead of
    # only the canonical names with underscores ("coffee_capsule"), which the
    # zero-shot detector handles poorly. canonicalize() maps every alias back.
    return sorted(set(mapper.aliases) | mapper.canonical_names)


class OwlVitSegmenter:
    """Zero-shot detector using transformers OWL-ViT, converted to instance masks."""

    def __init__(
        self,
        *,
        labels: list[str],
        model_path: str,
        score_threshold: float,
        mask_mode: str = "grabcut",
        image_color_order: str = "bgr",
    ) -> None:
        self.labels = labels
        self.model_path = model_path
        self.score_threshold = float(score_threshold)
        self.mask_mode = mask_mode
        self.image_color_order = image_color_order
        self._pipeline = None
        self.status_warning = ""

    def _load(self) -> Any | None:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except Exception as exc:
            self.status_warning = f"segmenter_backend_unavailable:transformers:{exc}"
            return None
        try:
            self._pipeline = pipeline(task="zero-shot-object-detection", model=self.model_path)
        except Exception as exc:
            self.status_warning = f"segmenter_backend_unavailable:{self.model_path}:{exc}"
            return None
        self.status_warning = ""
        return self._pipeline

    def _rgb_for_model(self, image: np.ndarray) -> np.ndarray:
        if self.image_color_order.lower() == "bgr" and image.ndim == 3 and image.shape[2] == 3:
            return image[:, :, ::-1]
        return image

    @staticmethod
    def _to_pil(image: np.ndarray) -> Any:
        """Return a contiguous uint8 PIL.Image; the transformers pipeline rejects
        numpy arrays (and reversed-stride views) for zero-shot object detection."""
        from PIL import Image

        array = np.ascontiguousarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def _box_from_detection(detection: Mapping[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
        raw_box = detection.get("box") or {}
        xmin = int(max(0, min(width - 1, round(float(raw_box.get("xmin", raw_box.get("x_min", 0)))))))
        ymin = int(max(0, min(height - 1, round(float(raw_box.get("ymin", raw_box.get("y_min", 0)))))))
        xmax = int(max(xmin + 1, min(width, round(float(raw_box.get("xmax", raw_box.get("x_max", width)))))))
        ymax = int(max(ymin + 1, min(height, round(float(raw_box.get("ymax", raw_box.get("y_max", height)))))))
        return xmin, ymin, xmax, ymax

    @staticmethod
    def _box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
        xmin, ymin, xmax, ymax = box
        mask = np.zeros(shape, dtype=bool)
        mask[ymin:ymax, xmin:xmax] = True
        return mask

    def _grabcut_mask(self, image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        try:
            import cv2
        except Exception:
            return self._box_mask(image.shape[:2], box)

        xmin, ymin, xmax, ymax = box
        rect = (xmin, ymin, max(1, xmax - xmin), max(1, ymax - ymin))
        grabcut_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(image, grabcut_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        except Exception:
            return self._box_mask(image.shape[:2], box)
        return np.isin(grabcut_mask, [cv2.GC_FGD, cv2.GC_PR_FGD])

    def _mask_for_detection(self, image: np.ndarray, detection: Mapping[str, Any]) -> np.ndarray:
        height, width = image.shape[:2]
        box = self._box_from_detection(detection, width, height)
        if self.mask_mode == "box":
            return self._box_mask((height, width), box)
        return self._grabcut_mask(image, box)

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        if not self.labels:
            self.status_warning = "segmenter_no_registry_labels"
            return []
        model = self._load()
        if model is None:
            return []
        try:
            model_image = self._rgb_for_model(rgb)
            pil_image = self._to_pil(model_image)
            outputs = model(pil_image, candidate_labels=self.labels, threshold=self.score_threshold)
        except Exception as exc:
            self.status_warning = f"segmenter_inference_error:{exc}"
            return []

        detections: list[Detection] = []
        for output in outputs or []:
            label = str(output.get("label") or "")
            score = float(output.get("score") or 0.0)
            if not label or score < self.score_threshold:
                continue
            detections.append(Detection(label=label, mask=self._mask_for_detection(rgb, output), score=score))
        self.status_warning = ""
        return detections


def build_segmenter(config: SegmenterConfig, registry: Mapping[str, Any] | None) -> Segmenter:
    backend = config.backend.strip().lower()
    if backend in {"", "noop", "none"}:
        return NoopSegmenter()
    labels = labels_from_registry(registry)
    if backend in {"owlvit", "owl_vit", "zero_shot"}:
        return OwlVitSegmenter(
            labels=labels,
            model_path=config.model_path,
            score_threshold=config.score_threshold,
            mask_mode=config.mask_mode,
            image_color_order=config.image_color_order,
        )
    raise ValueError(f"Unsupported segmenter_backend {config.backend!r}.")
