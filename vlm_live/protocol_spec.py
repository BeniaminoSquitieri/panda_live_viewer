"""Loader for the shared VLM protocol specification (``vlm_protocol.json``).

The JSON file next to this module is the single source of truth for the VLM
boundary and is kept byte-for-byte identical in the VLM-BT-BC repo. This loader
exposes it as data; a coherence test asserts that the hand-written contracts in
``contracts.py`` / ``const.py`` match the spec, so the two repos cannot drift
apart silently. See the VLM-BT-BC docs/ros_contract.md.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).with_name("vlm_protocol.json")


@lru_cache(maxsize=1)
def load_protocol_spec() -> dict[str, Any]:
    """Return the parsed shared VLM protocol specification."""
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))
