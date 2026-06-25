import json
import unittest

from bt_planning.grounding_prompt_builder import build_grounding_prompt
from bt_planning.grounding_service_logic import build_ground_instruction_response


def grounding_payload():
    # Mirrors build_grounding_payload's shape WITHOUT the canonical sequence.
    return {
        "grounding_schema_version": 1,
        "nl_instruction": "make a sandwich",
        "available_tasks": [
            {"name": "make_sandwich", "description": "Assemble a sandwich with a person."},
            {"name": "make_coffee", "description": "Prepare a capsule coffee."},
        ],
        "robot_skills": [{"name": "place_first_toast", "description": "Place the first toast."}],
        "human_steps": [{"name": "pour_ingredient", "description": "Pour the filling."}],
        "vlm_gates": [{"name": "initial_scene_ready", "description": "Scene ready."}],
        "allowed_objects": [{"canonical_name": "first_toast", "aliases": ["bottom toast"]}],
        "rules": {"no_xml": True},
    }


CANONICAL_RAW = json.dumps(
    {"task_name": "make_sandwich", "steps": [{"kind": "vlm_gate", "name": "initial_scene_ready"}]}
)


class GroundingServiceLogicTests(unittest.TestCase):
    def test_returns_raw_response_unparsed(self):
        captured = {}

        def backend(prompt):
            captured["prompt"] = prompt
            return CANONICAL_RAW

        result = build_ground_instruction_response(
            "make a sandwich",
            json.dumps(grounding_payload()),
            "",
            vlm_backend=backend,
        )
        self.assertTrue(result.success)
        # The transport boundary returns the VLM text unchanged.
        self.assertEqual(result.raw_response, CANONICAL_RAW)
        self.assertEqual(result.error_message, "")

    def test_prompt_never_contains_canonical_sequence(self):
        prompt = build_grounding_prompt("make a sandwich", grounding_payload(), None)
        self.assertNotIn("canonical_task_sequence", prompt)
        self.assertNotIn("ordering_constraints", prompt)

    def test_last_error_is_forwarded_into_prompt_after_rules(self):
        last_error = {"stage": "validation", "reason": "Generated plan does not match canonical task sequence"}
        prompt = build_grounding_prompt("make a sandwich", grounding_payload(), last_error)

        # The repair feedback reaches the model...
        self.assertIn("validation", prompt)
        self.assertIn("does not match canonical task sequence", prompt)
        # ...placed AFTER the rules block...
        self.assertGreater(prompt.find("previous attempt"), prompt.find("Rules:"))
        # ...and the very last line the model reads is still a format rule.
        last_line = [line for line in prompt.splitlines() if line.strip()][-1]
        self.assertIn("No XML", last_line)

    def test_backend_error_reports_failure(self):
        def backend(prompt):
            raise RuntimeError("model unreachable")

        result = build_ground_instruction_response(
            "make a sandwich",
            json.dumps(grounding_payload()),
            "",
            vlm_backend=backend,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.raw_response, "")
        self.assertIn("model unreachable", result.error_message)

    def test_missing_instruction_is_rejected(self):
        result = build_ground_instruction_response(
            "", json.dumps(grounding_payload()), "", vlm_backend=lambda p: CANONICAL_RAW
        )
        self.assertFalse(result.success)
        self.assertIn("nl_instruction is required", result.error_message)


if __name__ == "__main__":
    unittest.main()
