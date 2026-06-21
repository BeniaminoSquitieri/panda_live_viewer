"""Unit tests for the pure VLM decision rules (no ROS node required)."""

from __future__ import annotations

import pytest

from vlm_live.const import STATUS_FAILURE, STATUS_RUNNING, STATUS_SUCCESS
from vlm_live.decision import (
    UNCERTAINTY_MARKERS,
    downgrade_uncertain_failure,
    is_uncertain_reason,
)


@pytest.mark.parametrize("marker", UNCERTAINTY_MARKERS)
def test_each_marker_is_detected(marker: str) -> None:
    assert is_uncertain_reason(f"The object is {marker} right now.")


def test_uncertainty_detection_is_case_insensitive() -> None:
    assert is_uncertain_reason("Object is OCCLUDED by the gripper")


def test_confident_reason_is_not_uncertain() -> None:
    assert not is_uncertain_reason("The capsule is clearly inside the holder.")


def test_empty_reason_is_not_uncertain() -> None:
    assert not is_uncertain_reason("")


def test_failure_with_uncertain_reason_is_downgraded() -> None:
    status, reason = downgrade_uncertain_failure(STATUS_FAILURE, "Capsule not visible in current frame")
    assert status == STATUS_RUNNING
    assert reason == "Capsule not visible in current frame"


def test_failure_with_confident_reason_stays_failure() -> None:
    status, reason = downgrade_uncertain_failure(STATUS_FAILURE, "Capsule is on the floor")
    assert status == STATUS_FAILURE
    assert reason == "Capsule is on the floor"


def test_non_failure_statuses_pass_through_untouched() -> None:
    # Even an uncertain reason must not change a non-FAILURE status.
    assert downgrade_uncertain_failure(STATUS_SUCCESS, "appears done") == (STATUS_SUCCESS, "appears done")
    assert downgrade_uncertain_failure(STATUS_RUNNING, "occluded") == (STATUS_RUNNING, "occluded")
