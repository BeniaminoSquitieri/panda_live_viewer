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
    return {
        STATUS_SUCCESS: (SemanticStatus.SATISFIED, ControlAction.STOP_SUCCESS),
        STATUS_FAILURE: (SemanticStatus.ANOMALY, ControlAction.STOP_UNSAFE),
        STATUS_WAIT_HUMAN: (SemanticStatus.UNKNOWN, ControlAction.REQUEST_HUMAN),
        STATUS_RUNNING: (SemanticStatus.NOT_SATISFIED, ControlAction.KEEP_RUNNING),
    }.get(status, (SemanticStatus.UNKNOWN, ControlAction.KEEP_RUNNING))
