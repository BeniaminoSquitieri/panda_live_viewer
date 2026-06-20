"""Covariance-aware multi-camera fusion for object scene facts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .contracts import ObservationStatus, enrich_fact_contract


def _translation(fact: Mapping[str, Any]) -> np.ndarray | None:
    pose = fact.get("pose")
    if not isinstance(pose, Mapping):
        return None
    raw = pose.get("translation")
    if not isinstance(raw, Mapping):
        return None
    try:
        value = np.array([raw["x"], raw["y"], raw["z"]], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.all(np.isfinite(value)) else None


def _translation_covariance(fact: Mapping[str, Any]) -> np.ndarray | None:
    values = fact.get("covariance", []) or []
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size == 36:
        covariance = array.reshape(6, 6)[:3, :3]
    elif array.size == 9:
        covariance = array.reshape(3, 3)
    else:
        return None
    if not np.all(np.isfinite(covariance)):
        return None
    return covariance + np.eye(3) * 1e-8


def _embed_translation_covariance(base: Iterable[float], covariance: np.ndarray) -> list[float]:
    raw = [float(value) for value in base]
    output = np.zeros((6, 6), dtype=float)
    if len(raw) == 36:
        output = np.asarray(raw, dtype=float).reshape(6, 6)
    output[:3, :3] = covariance
    return output.reshape(-1).tolist()


def fuse_object_facts(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Fuse same-object observations in a common frame, falling back conservatively."""
    if not observations:
        raise ValueError("observations must not be empty.")
    observations = [dict(item) for item in observations]
    posed = [item for item in observations if _translation(item) is not None]
    if not posed:
        chosen = max(observations, key=lambda item: float(item.get("pose_confidence", 0.0)))
        sources = sorted({source for item in observations for source in item.get("source_cameras", []) or []})
        chosen["source_cameras"] = sources
        return enrich_fact_contract(chosen)

    frames = {str(item.get("frame_id") or "") for item in posed}
    if len(frames) != 1:
        chosen = max(posed, key=lambda item: float(item.get("pose_confidence", 0.0)))
        chosen.setdefault("warnings", []).append("fusion_frame_mismatch")
        return enrich_fact_contract(chosen)

    weighted: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    for item in posed:
        covariance = _translation_covariance(item)
        translation = _translation(item)
        if covariance is None or translation is None:
            continue
        try:
            information = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            continue
        weighted.append((translation, information, item))

    best = max(posed, key=lambda item: float(item.get("pose_confidence", 0.0)))
    if len(weighted) < 2:
        result = dict(best)
        result.setdefault("warnings", []).append("fusion_fallback_best_observation")
    else:
        information_sum = sum((entry[1] for entry in weighted), np.zeros((3, 3), dtype=float))
        fused_covariance = np.linalg.inv(information_sum)
        rhs = sum((information @ translation for translation, information, _ in weighted), np.zeros(3))
        fused_translation = fused_covariance @ rhs
        result = dict(best)
        result["pose"] = dict(best["pose"])
        result["pose"]["translation"] = {
            "x": float(fused_translation[0]),
            "y": float(fused_translation[1]),
            "z": float(fused_translation[2]),
        }
        result["covariance"] = _embed_translation_covariance(
            best.get("covariance", []), fused_covariance
        )
        confidences = [max(0.0, min(float(item.get("pose_confidence", 0.0)), 1.0)) for item in posed]
        result["pose_confidence"] = 1.0 - float(np.prod([1.0 - value for value in confidences]))
        result["geometry_quality"] = result["pose_confidence"]
        result.setdefault("warnings", []).append("fused_multi_camera")
        result["observation_status"] = ObservationStatus.DETECTED_WITH_POSE

    result["source_cameras"] = sorted(
        {source for item in observations for source in item.get("source_cameras", []) or []}
    )
    result["stamp"] = max(float(item.get("stamp", 0.0) or 0.0) for item in observations)
    return enrich_fact_contract(result)


def fuse_camera_fact_sets(fact_sets: Iterable[Mapping[str, dict[str, Any]]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact_set in fact_sets:
        for name, fact in fact_set.items():
            grouped[str(name)].append(dict(fact))
    return {name: fuse_object_facts(items) for name, items in grouped.items()}
