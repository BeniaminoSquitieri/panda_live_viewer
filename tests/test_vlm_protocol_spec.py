"""Coherence guard: the viewer VLM contracts must match the shared spec.

``vlm_live/vlm_protocol.json`` is the single source of truth for the VLM
boundary, kept byte-for-byte identical in the VLM-BT-BC repo. These tests assert
that ``contracts.semantic_contract_for_status`` and the ``const.py`` vocabulary
agree with that data, so the two repos cannot drift apart silently.
"""

from __future__ import annotations

from vlm_live.const import SUPPORTED_STATUSES
from vlm_live.contracts import ControlAction, SemanticStatus, semantic_contract_for_status
from vlm_live.protocol_spec import load_protocol_spec


def test_protocol_schema_version_is_current() -> None:
    assert load_protocol_spec()["schema_version"] == 2


def test_supported_statuses_match_spec() -> None:
    spec = load_protocol_spec()
    assert set(SUPPORTED_STATUSES) == set(spec["statuses"])


def test_control_action_enum_matches_spec() -> None:
    spec = load_protocol_spec()
    assert {a.value for a in ControlAction} == set(spec["control_actions"])


def test_semantic_status_enum_matches_spec() -> None:
    spec = load_protocol_spec()
    assert {s.value for s in SemanticStatus} == set(spec["semantic_statuses"])


def test_status_to_semantic_contract_matches_spec() -> None:
    spec = load_protocol_spec()
    for status, expected in spec["status_to_semantic"].items():
        if status == "_default":
            continue
        semantic, control = semantic_contract_for_status(status)
        assert semantic.value == expected["semantic"], status
        assert control.value == expected["control_action"], status


def test_default_contract_matches_spec() -> None:
    spec = load_protocol_spec()
    default = spec["status_to_semantic"]["_default"]
    semantic, control = semantic_contract_for_status("SOMETHING_UNKNOWN")
    assert semantic.value == default["semantic"]
    assert control.value == default["control_action"]
