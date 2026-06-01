import json
import unittest
from pathlib import Path

from bt_planning.baseline_prompts import (
    BASELINE_UNSAFE_MARKER,
    build_constrained_linear_ir_prompt,
    build_direct_xml_prompt,
    build_unconstrained_linear_ir_prompt,
)


def registry():
    return {
        "robot_skills": ["place_first_toast"],
        "human_steps": ["pour_ingredient"],
        "vlm_gates": ["initial_scene_ready"],
        "objects": [{"canonical_name": "toast"}],
        "canonical_task_sequence": [
            {"kind": "vlm_gate", "name": "initial_scene_ready"},
            {"kind": "robot_skill", "name": "place_first_toast"},
            {"kind": "human_step", "name": "pour_ingredient"},
        ],
    }


class BaselinePromptsTests(unittest.TestCase):
    def test_direct_xml_is_marked_unsafe_offline_only(self):
        prompt = build_direct_xml_prompt("make_sandwich", registry())
        self.assertIn(BASELINE_UNSAFE_MARKER, prompt)
        self.assertEqual(BASELINE_UNSAFE_MARKER, "BASELINE_UNSAFE_OFFLINE_ONLY")

    def test_constrained_includes_canonical_task_sequence(self):
        prompt = build_constrained_linear_ir_prompt("make_sandwich", registry())
        self.assertIn("canonical_task_sequence", prompt)

    def test_unconstrained_omits_follow_canonical_instruction(self):
        prompt = build_unconstrained_linear_ir_prompt("make_sandwich", registry())
        self.assertNotIn("Follow canonical_task_sequence exactly", prompt)
        # Still a Linear IR prompt that forbids XML.
        self.assertIn("Do NOT return XML.", prompt)

    def test_direct_xml_not_imported_by_service_logic(self):
        source = Path(__file__).resolve().parents[1] / "bt_planning" / "service_logic.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("baseline_prompts", text)
        self.assertNotIn("build_direct_xml_prompt", text)


if __name__ == "__main__":
    unittest.main()
