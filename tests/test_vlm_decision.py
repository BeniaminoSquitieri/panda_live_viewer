"""Unit tests for the pure VLM decision rules (no ROS node required)."""

from __future__ import annotations

import unittest
from vlm_live.const import STATUS_FAILURE, STATUS_RUNNING, STATUS_SUCCESS
from vlm_live.decision import (
    UNCERTAINTY_MARKERS,
    downgrade_uncertain_failure,
    is_terminal_gate_status,
    is_uncertain_reason,
)


class UncertaintyMarkerTests(unittest.TestCase):
    def test_each_marker_is_detected(self) -> None:
        for marker in UNCERTAINTY_MARKERS:
            with self.subTest(marker=marker):
                self.assertTrue(is_uncertain_reason(f"The object is {marker} right now."))

    def test_uncertainty_detection_is_case_insensitive(self) -> None:
        self.assertTrue(is_uncertain_reason("Object is OCCLUDED by the gripper"))

    def test_confident_reason_is_not_uncertain(self) -> None:
        self.assertFalse(is_uncertain_reason("The capsule is clearly inside the holder."))

    def test_empty_reason_is_not_uncertain(self) -> None:
        self.assertFalse(is_uncertain_reason(""))

    def test_failure_with_uncertain_reason_is_downgraded(self) -> None:
        reason = "Capsule not visible in current frame"
        self.assertEqual(downgrade_uncertain_failure(STATUS_FAILURE, reason), (STATUS_RUNNING, reason))

    def test_failure_with_confident_reason_stays_failure(self) -> None:
        reason = "Capsule is on the floor"
        self.assertEqual(downgrade_uncertain_failure(STATUS_FAILURE, reason), (STATUS_FAILURE, reason))

    def test_non_failure_statuses_pass_through_untouched(self) -> None:
        for status, reason in ((STATUS_SUCCESS, "appears done"), (STATUS_RUNNING, "occluded")):
            self.assertEqual(downgrade_uncertain_failure(status, reason), (status, reason))

    def test_only_success_closes_the_scene_gate(self) -> None:
        for status in ("RUNNING", "FAILURE", "WAIT_HUMAN"):
            self.assertFalse(is_terminal_gate_status(status))
        self.assertTrue(is_terminal_gate_status("SUCCESS"))
