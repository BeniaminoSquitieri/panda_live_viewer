import json
import unittest

from bt_planning.plan_parser import PlanParseError, parse_linear_ir_plan


def valid_plan():
    return {
        "task_name": "make_sandwich",
        "steps": [
            {"kind": "vlm_gate", "name": "initial_scene_ready"},
            {"kind": "robot_skill", "name": "place_first_toast"},
            {"kind": "human_step", "name": "pour_ingredient"},
        ],
    }


class LinearIrParserTests(unittest.TestCase):
    def test_valid_json_object(self):
        plan = parse_linear_ir_plan(json.dumps(valid_plan()))

        self.assertEqual(plan["task_name"], "make_sandwich")
        self.assertEqual(len(plan["steps"]), 3)

    def test_valid_fenced_json_block(self):
        raw = "```json\n" + json.dumps(valid_plan()) + "\n```"

        plan = parse_linear_ir_plan(raw)

        self.assertEqual(plan["steps"][0]["kind"], "vlm_gate")

    def test_rejects_pure_xml(self):
        with self.assertRaises(PlanParseError):
            parse_linear_ir_plan("<root><step kind=\"robot_skill\" name=\"foo\"/></root>")

    def test_rejects_xml_inside_prose(self):
        with self.assertRaises(PlanParseError):
            parse_linear_ir_plan("Here is a plan: <BehaviorTree><Action ID=\"foo\"/></BehaviorTree>")

    def test_rejects_markdown_or_prose_outside_json(self):
        raw = "Here is the plan:\n```json\n" + json.dumps(valid_plan()) + "\n```"

        with self.assertRaises(PlanParseError):
            parse_linear_ir_plan(raw)

    def test_rejects_forbidden_fields(self):
        for field in (
            "raw_xml",
            "xml",
            "action",
            "condition",
            "explanation",
            "free_text",
            "fallback",
            "parallel",
            "human_fallback",
        ):
            plan = valid_plan()
            plan["steps"][0][field] = "not allowed"
            with self.subTest(field=field):
                with self.assertRaises(PlanParseError):
                    parse_linear_ir_plan(json.dumps(plan))

    def test_rejects_confidence_step_field(self):
        plan = valid_plan()
        plan["steps"][0]["confidence"] = 0.92

        with self.assertRaisesRegex(PlanParseError, "confidence"):
            parse_linear_ir_plan(json.dumps(plan))

    def test_rejects_reason_step_field(self):
        plan = valid_plan()
        plan["steps"][0]["reason"] = "Looks ready."

        with self.assertRaisesRegex(PlanParseError, "reason"):
            parse_linear_ir_plan(json.dumps(plan))

    def test_accepts_valid_object_step_field(self):
        plan = valid_plan()
        plan["steps"][1]["object"] = "toast"

        parsed = parse_linear_ir_plan(json.dumps(plan))

        self.assertEqual(parsed["steps"][1]["object"], "toast")

    def test_accepts_valid_objects_step_field(self):
        plan = valid_plan()
        plan["steps"][1]["objects"] = ["toast", "plate"]

        parsed = parse_linear_ir_plan(json.dumps(plan))

        self.assertEqual(parsed["steps"][1]["objects"], ["toast", "plate"])

    def test_rejects_wrong_kind(self):
        plan = valid_plan()
        plan["steps"][0]["kind"] = "condition"

        with self.assertRaises(PlanParseError):
            parse_linear_ir_plan(json.dumps(plan))

    def test_rejects_type_instead_of_kind(self):
        plan = valid_plan()
        plan["steps"][0].pop("kind")
        plan["steps"][0]["type"] = "vlm_gate"

        with self.assertRaises(PlanParseError):
            parse_linear_ir_plan(json.dumps(plan))


if __name__ == "__main__":
    unittest.main()
