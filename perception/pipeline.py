"""Perception pipeline primitives for scene facts generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from bt_planning.scene_facts import build_object_pose_fact

from .contracts import ObservationStatus, enrich_fact_contract
from .geometry import CameraIntrinsics, back_project_mask, estimate_pose_pca, robust_filter_points


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
        return []


@dataclass(frozen=True)
class RegistryMapper:
    """Map detector labels onto the canonical planner registry vocabulary."""

    canonical_names: set[str]
    aliases: dict[str, str]

    @classmethod
    def from_registry(cls, registry: Mapping[str, Any] | None) -> RegistryMapper:
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


@dataclass
class _Candidate:
    """One detection retained for per-object resolution."""

    fact: dict[str, Any]
    centroid: tuple[float, float, float] | None
    quality: tuple[int, float, float]


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.sqrt(sum((left - right) ** 2 for left, right in zip(a, b, strict=True))))


class PerceptionPipeline:
    """RGB-D perception pipeline with explicit uncertainty and safe fallbacks."""

    def __init__(
        self,
        segmenter: Segmenter | None = None,
        min_depth_points: int = 25,
        *,
        pose_max_depth_m: float = 0.0,
        pose_depth_band_m: float = 0.0,
        depth_scale_m: float = 0.001,
        support_preferences: Mapping[str, str] | None = None,
        support_radius_m: float = 0.0,
    ):
        self.segmenter = segmenter or NoopSegmenter()
        self.min_depth_points = int(min_depth_points)
        self.pose_max_depth_m = float(pose_max_depth_m)
        self.pose_depth_band_m = float(pose_depth_band_m)
        self.depth_scale_m = float(depth_scale_m)
        # Map of object -> support object (e.g. coffee_capsule -> plate). When the
        # support object is detected, the instance of the target object nearest to
        # the support centroid is preferred over the most-confident one.
        self.support_preferences = {str(key): str(value) for key, value in (support_preferences or {}).items()}
        self.support_radius_m = float(support_radius_m)

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
        warnings: list[str] = []
        seen_by_name: dict[str, int] = {}
        # Collect every canonicalized detection as a candidate so we can resolve
        # which instance to expose per object (most-confident, or the instance
        # nearest a support object when a spatial preference is configured).
        candidates: dict[str, list[_Candidate]] = {}

        for detection in detections:
            canonical_name = mapper.canonicalize(detection.label)
            if canonical_name is None:
                warnings.append(f"unknown_class:{detection.label}")
                continue

            seen_by_name[canonical_name] = seen_by_name.get(canonical_name, 0) + 1
            object_warnings: list[str] = []

            translation = quaternion = covariance = pose_residual_m = inlier_ratio = None
            confidence = 0.0

            if depth is None or intrinsics is None:
                object_warnings.append("insufficient_depth_or_camera_info")
            else:
                try:
                    points = back_project_mask(
                        depth=depth,
                        mask=detection.mask,
                        intrinsics=intrinsics,
                        max_depth_m=self.pose_max_depth_m,
                        depth_band_m=self.pose_depth_band_m,
                        depth_scale_m=self.depth_scale_m,
                    )
                    points = robust_filter_points(points)
                except ValueError as exc:
                    object_warnings.append(f"back_projection_error:{exc}")
                    points = np.empty((0, 3), dtype=np.float32)

                if points.shape[0] < self.min_depth_points:
                    if not any(w.startswith("back_projection_error") for w in object_warnings):
                        object_warnings.append("insufficient_depth")
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

            has_pose = translation is not None and quaternion is not None
            centroid = tuple(float(translation[key]) for key in "xyz") if has_pose else None
            fact = build_object_pose_fact(
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
            fact["observation_status"] = ObservationStatus.DETECTED_WITH_POSE if has_pose else ObservationStatus.DETECTED_NO_POSE
            fact["geometry_quality"] = confidence
            enrich_fact_contract(fact)
            quality = (1 if has_pose else 0, confidence, float(detection.score))
            candidates.setdefault(canonical_name, []).append(_Candidate(fact=fact, centroid=centroid, quality=quality))

        facts = self._resolve_candidates(candidates)

        # Missing detections are explicit observations, not proof of absence.
        # This distinction lets semantic consumers render UNKNOWN/NOT_DETECTED
        # without inventing a metric pose.
        for canonical_name in mapper.canonical_names:
            if canonical_name in facts:
                continue
            missing_fact = build_object_pose_fact(
                name=canonical_name,
                present=False,
                frame_id=frame_id,
                stamp=stamp,
                pose_confidence=0.0,
                warnings=["not_detected_in_frame"],
            )
            missing_fact["observation_status"] = ObservationStatus.NOT_DETECTED
            missing_fact["geometry_quality"] = 0.0
            facts[canonical_name] = enrich_fact_contract(missing_fact)

        # Flag objects with multiple detected instances on the retained fact.
        for canonical_name, count in seen_by_name.items():
            if count > 1 and canonical_name in facts:
                fact_warnings = facts[canonical_name].setdefault("warnings", [])
                if "ambiguous_multiple_instances" not in fact_warnings:
                    fact_warnings.append("ambiguous_multiple_instances")

        if not detections:
            segmenter_warning = str(getattr(self.segmenter, "status_warning", "") or "")
            warnings.append(segmenter_warning or "segmenter_no_detections")

        return PipelineResult(facts=facts, warnings=warnings, detections_seen=len(detections))

    def _resolve_candidates(
        self,
        candidates: dict[str, list[_Candidate]],
    ) -> dict[str, Any]:
        """Select one fact per object, applying spatial support preferences."""
        # Preliminary pick: most-confident candidate (valid pose first).
        chosen = {name: max(items, key=lambda candidate: candidate.quality) for name, items in candidates.items()}

        support_centroids = {
            name: candidate.centroid
            for name, candidate in chosen.items()
            if candidate.centroid is not None
        }

        for name, support_name in self.support_preferences.items():
            items = candidates.get(name)
            if not items:
                continue
            support_centroid = support_centroids.get(support_name)
            if support_centroid is None:
                chosen[name].fact.setdefault("warnings", []).append(
                    f"support_unavailable:{support_name}"
                )
                continue
            located = [c for c in items if c.centroid is not None]
            if not located:
                continue
            nearest = min(located, key=lambda candidate: _distance(candidate.centroid, support_centroid))
            nearest_distance = _distance(nearest.centroid, support_centroid)
            if self.support_radius_m > 0.0 and nearest_distance > self.support_radius_m:
                chosen[name].fact.setdefault("warnings", []).append(
                    f"no_instance_near_support:{support_name}"
                )
                continue
            nearest.fact.setdefault("warnings", []).append(
                f"selected_near_support:{support_name}"
            )
            chosen[name] = nearest

        return {name: candidate.fact for name, candidate in chosen.items()}
