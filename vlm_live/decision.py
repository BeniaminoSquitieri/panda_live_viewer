"""Pure decision rules for the VLM live verifier (no rclpy, no I/O).

Extracted from ``node.py`` so the rules can be unit-tested with plain
input/output, without constructing a ROS node. The node keeps the side effects
(logging, publishing); this module only computes status transitions.
"""

from __future__ import annotations

from .const import STATUS_FAILURE, STATUS_RUNNING

# Phrases in a VLM reason that indicate the model is *uncertain* rather than
# observing a genuine anomaly. A FAILURE justified only by uncertainty is
# downgraded to RUNNING so the BT keeps trying instead of stopping unsafely.
UNCERTAINTY_MARKERS: tuple[str, ...] = (
    "not visible",
    "not clearly visible",
    "no clear evidence",
    "not clear",
    "unclear",
    "cannot confirm",
    "does not confirm",
    "no definitive",
    "insufficient",
    "hard to see",
    "occluded",
    "possibly",
    "appears",
    "current frame",
)


def is_uncertain_reason(reason: str) -> bool:
    """Return True if ``reason`` reads as uncertainty rather than a hard verdict."""
    reason_lower = (reason or "").lower()
    return any(marker in reason_lower for marker in UNCERTAINTY_MARKERS)


def downgrade_uncertain_failure(status: str, reason: str) -> tuple[str, str]:
    """Map an uncertainty-only FAILURE to RUNNING; pass everything else through.

    Returns the (possibly unchanged) ``(status, reason)``. The caller is
    responsible for logging when a downgrade happens (compare the returned
    status with the input to detect it).
    """
    if status != STATUS_FAILURE:
        return status, reason
    if is_uncertain_reason(reason):
        return STATUS_RUNNING, reason
    return status, reason
