"""Perception pipeline primitives for scene facts generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

import numpy as np

from bt_planning.scene_facts import build_object_pose_fact

from .geometry import CameraIntrinsics, back_project_mask, estimate_pose_pca


@dataclass(frozen=True)
class Detection:
    """One class-aware instance mask from a segmenter."""

    label: str
    mask: np.ndarray
    score: float


class Segmenter(Protocol):
    """Protocol for pluggable segmentation backends."""

    def detect(self, rgb: np.ndarray) -> Iterable[Detection]:
        ...


class NoopSegmenter:
    """Fail-closed segmenter used until a real model is configured."""

    def detect(self, rgb: np.ndarray) -> Iterable[Detection]:
        _ = rgb
        return []


@dataclass(frozen=True)
class RegistryMapper:
    """Map detector labels onto the canonical planner registry vocabulary."""

    canonical_names: set[str]
    aliases: dict[str, str]

    @classmethod
    def from_registry(cls, registry: Mapping[str, Any] | None) -> "RegistryMapper":
        registry = registry or {}
        aliases: dict[str, str] = {}
        canonical_names: set[str] = set()

        raw_objects = registry.get("objects", [])
        if isinstance(raw_objects, list):
            for raw_object in raw_objects:
                if isinstance(raw_object, Mapping):
                    canonical_name = str(raw_object.get("canonical_name") or raw_object.get("name") or "").strip()
                    if not canonical_name:
                        continue
                    canonical_names.add(canonical_name)
                    aliases[canonical_name.lower()] = canonical_name
                    for alias in raw_object.get("aliases", []) or []:
                        aliases[str(alias).lower()] = canonical_name
                else:
                    canonical_name = str(raw_object).strip()
                    if canonical_name:
                        canonical_names.add(canonical_name)
                        aliases[canonical_name.lower()] = canonical_name

        for key in ("aliases", "object_aliases"):
            raw_aliases = registry.get(key, {})
            if isinstance(raw_aliases, Mapping):
                for alias, canonical in raw_aliases.items():
                    aliases[str(alias).lower()] = str(canonical)

        for name in canonical_names:
            aliases[name.lower()] = name
        return cls(canonical_names=canonical_names, aliases=aliases)

    def canonicalize(self, label: str) -> str | None:
        candidate = self.aliases.get(label.lower())
        if candidate in self.canonical_names:
            return candidate
        if not self.canonical_names and label:
            return label
        return None


@dataclass
class PipelineResult:
    facts: dict[str, Any]
    warnings: list[str]
    detections_seen: int


class PerceptionPipeline:
    """RGB-D perception pipeline with explicit uncertainty and safe fallbacks."""

    def __init__(self, segmenter: Segmenter | None = None, min_depth_points: int = 25):
        self.segmenter = segmenter or NoopSegmenter()
        self.min_depth_points = int(min_depth_points)

    def run(
        self,
        *,
        rgb: np.ndarray | None,
        depth: np.ndarray | None,
        intrinsics: CameraIntrinsics | None,
        registry: Mapping[str, Any] | None,
        frame_id: str,
        stamp: Any,
    ) -> PipelineResult:
        if rgb is None:
            return PipelineResult(facts={}, warnings=["rgb_unavailable"], detections_seen=0)

        mapper = RegistryMapper.from_registry(registry)
        detections = list(self.segmenter.detect(rgb))
        facts: dict[str, Any] = {}
        warnings: list[str] = []
        seen_by_name: dict[str, int] = {}

        for detection in detections:
            canonical_name = mapper.canonicalize(detection.label)
            if canonical_name is None:
                warnings.append(f"unknown_class:{detection.label}")
                continue

            seen_by_name[canonical_name] = seen_by_name.get(canonical_name, 0) + 1
            object_warnings: list[str] = []
            if seen_by_name[canonical_name] > 1:
                object_warnings.append("ambiguous_multiple_instances")

            if depth is None or intrinsics is None:
                object_warnings.append("insufficient_depth_or_camera_info")
                facts[canonical_name] = build_object_pose_fact(
                    name=canonical_name,
                    present=True,
                    frame_id=frame_id,
                    stamp=stamp,
                    pose_confidence=0.0,
                    seg_score=detection.score,
                    warnings=object_warnings,
                )
                continue

            try:
                points = back_project_mask(depth=depth, mask=detection.mask, intrinsics=intrinsics)
            except ValueError as exc:
                object_warnings.append(f"back_projection_error:{exc}")
                facts[canonical_name] = build_object_pose_fact(
                    name=canonical_name,
                    present=True,
                    frame_id=frame_id,
                    stamp=stamp,
                    pose_confidence=0.0,
                    seg_score=detection.score,
                    warnings=object_warnings,
                )
                continue
            if points.shape[0] < self.min_depth_points:
                object_warnings.append("insufficient_depth")
                confidence = 0.0
                translation = None
                quaternion = None
                covariance = None
                pose_residual_m = None
                inlier_ratio = None
            else:
                pose = estimate_pose_pca(points)
                translation = pose.translation
                quaternion = pose.quaternion_xyzw
                covariance = pose.covariance
                density_score = min(points.shape[0] / float(self.min_depth_points * 4), 1.0)
                confidence = max(0.0, min(float(detection.score) * density_score * pose.confidence, 1.0))
                pose_residual_m = pose.residual_m
                inlier_ratio = pose.inlier_ratio
                object_warnings.extend(pose.warnings)
                object_warnings.append("orientation_estimated_pca")

            facts[canonical_name] = build_object_pose_fact(
                name=canonical_name,
                present=True,
                frame_id=frame_id,
                stamp=stamp,
                translation=translation,
                quaternion_xyzw=quaternion,
                pose_confidence=confidence,
                covariance=covariance,
                seg_score=detection.score,
                pose_residual_m=pose_residual_m,
                inlier_ratio=inlier_ratio,
                warnings=object_warnings,
            )

        if not detections:
            segmenter_warning = str(getattr(self.segmenter, "status_warning", "") or "")
            warnings.append(segmenter_warning or "segmenter_no_detections")

        return PipelineResult(facts=facts, warnings=warnings, detections_seen=len(detections))
