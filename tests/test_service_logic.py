import json
import unittest

from bt_planning.service_logic import build_generate_plan_response


def registry():
    return {
        "robot_skills": ["place_first_toast"],
        "human_steps": ["pour_ingredient"],
        "vlm_gates": ["initial_scene_ready"],
        "canonical_task_sequence": [
            {"kind": "vlm_gate", "name": "initial_scene_ready"},
            {"kind": "robot_skill", "name": "place_first_toast"},
            {"kind": "human_step", "name": "pour_ingredient"},
        ],
    }


class GeneratePlanServiceLogicTests(unittest.TestCase):
    def test_accepts_matching_vlm_plan(self):
        raw_plan = {
            "task_name": "make_sandwich",
            "steps": registry()["canonical_task_sequence"],
        }

        result = build_generate_plan_response(
            task_name=" make_sandwich ",
            planner_registry_json=json.dumps(registry()),
            dry_run=False,
            vlm_backend=lambda _prompt: json.dumps(raw_plan),
        )

        self.assertTrue(result.success, result.error_message)

    def test_rejects_task_name_mismatch(self):
        raw_plan = {
            "task_name": "other_task",
            "steps": registry()["canonical_task_sequence"],
        }

        result = build_generate_plan_response(
            task_name="make_sandwich",
            planner_registry_json=json.dumps(registry()),
            dry_run=False,
            vlm_backend=lambda _prompt: json.dumps(raw_plan),
        )

        self.assertFalse(result.success)
        self.assertIn("does not match request task_name", result.error_message)

    def test_rejects_canonical_sequence_mismatch(self):
        raw_plan = {
            "task_name": "make_sandwich",
            "steps": list(reversed(registry()["canonical_task_sequence"])),
        }

        result = build_generate_plan_response(
            task_name="make_sandwich",
            planner_registry_json=json.dumps(registry()),
            dry_run=False,
            vlm_backend=lambda _prompt: json.dumps(raw_plan),
        )

        self.assertFalse(result.success)
        self.assertIn("canonical_task_sequence", result.error_message)


if __name__ == "__main__":
    unittest.main()
