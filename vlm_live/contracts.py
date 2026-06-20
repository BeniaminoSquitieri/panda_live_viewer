"""Semantic-verdict and control-action contracts for the VLM boundary."""

from __future__ import annotations

from enum import StrEnum

from .const import STATUS_FAILURE, STATUS_RUNNING, STATUS_SUCCESS, STATUS_WAIT_HUMAN


class SemanticStatus(StrEnum):
    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    UNKNOWN = "UNKNOWN"
    ANOMALY = "ANOMALY"


class ControlAction(StrEnum):
    KEEP_RUNNING = "KEEP_RUNNING"
    STOP_SUCCESS = "STOP_SUCCESS"
    STOP_UNSAFE = "STOP_UNSAFE"
    REQUEST_HUMAN = "REQUEST_HUMAN"


def semantic_contract_for_status(status: str) -> tuple[SemanticStatus, ControlAction]:
    """Map the model vocabulary to task semantics and explicit control intent."""
    if status == STATUS_SUCCESS:
        return SemanticStatus.SATISFIED, ControlAction.STOP_SUCCESS
    if status == STATUS_FAILURE:
        return SemanticStatus.ANOMALY, ControlAction.STOP_UNSAFE
    if status == STATUS_WAIT_HUMAN:
        return SemanticStatus.UNKNOWN, ControlAction.REQUEST_HUMAN
    if status == STATUS_RUNNING:
        return SemanticStatus.NOT_SATISFIED, ControlAction.KEEP_RUNNING
    return SemanticStatus.UNKNOWN, ControlAction.KEEP_RUNNING
