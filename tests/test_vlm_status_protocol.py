import json
import unittest
from types import SimpleNamespace

from vlm_live.const import STATUS_FAILURE, STATUS_RUNNING, STATUS_SUCCESS, STATUS_WAIT_HUMAN
from vlm_live.prompt import build_prompt, extract_reason, fit_status, format_scene_context, parse_status
from vlm_live.protocol import PROTOCOL_SCHEMA_VERSION, build_result_payload, clean_statuses, parse_request


class _Logger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)


class VlmStatusProtocolTests(unittest.TestCase):
    def test_request_and_result_carry_current_protocol_version(self):
        logger = _Logger()
        request = parse_request(
            SimpleNamespace(
                data=json.dumps(
                    {
                        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
                        "skill_name": "ingredient_poured",
                        "attempt_id": 4,
                    }
                )
            ),
            logger,
        )

        self.assertIsNotNone(request)
        self.assertEqual(request["protocol_schema_version"], PROTOCOL_SCHEMA_VERSION)
        result = build_result_payload(request, STATUS_WAIT_HUMAN, "Please adjust the scene.")
        self.assertEqual(result["protocol_schema_version"], PROTOCOL_SCHEMA_VERSION)

    def test_explicit_incompatible_protocol_version_is_rejected(self):
        logger = _Logger()
        request = parse_request(
            SimpleNamespace(
                data=json.dumps(
                    {
                        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION + 1,
                        "skill_name": "ingredient_poured",
                    }
                )
            ),
            logger,
        )

        self.assertIsNone(request)
        self.assertTrue(any("expected" in message for message in logger.messages))

    def test_reasoning_prompt_mentions_wait_human(self):
        prompt = build_prompt(
            {
                "skill_name": "ingredient_poured",
                "attempt_id": 1,
                "message": "Check whether the ingredient has been poured.",
                "task": "Verify ingredient state.",
                "allowed_statuses": [STATUS_RUNNING, STATUS_SUCCESS, STATUS_FAILURE, STATUS_WAIT_HUMAN],
            },
            reasoning=True,
        )

        self.assertIn("STATUS=<RUNNING|SUCCESS|FAILURE|WAIT_HUMAN>", prompt)
        self.assertIn("human intervention", prompt)

    def test_prompt_omits_wait_human_when_not_allowed(self):
        prompt = build_prompt(
            {
                "skill_name": "ingredient_poured",
                "attempt_id": 1,
                "allowed_statuses": [STATUS_RUNNING, STATUS_SUCCESS, STATUS_FAILURE],
            },
            reasoning=True,
        )

        self.assertIn("STATUS=<RUNNING|SUCCESS|FAILURE>", prompt)
        self.assertNotIn("WAIT_HUMAN", prompt)

    def test_parse_status_wait_human(self):
        self.assertEqual(parse_status("STATUS=WAIT_HUMAN\nREASON=Need a human decision."), STATUS_WAIT_HUMAN)
        self.assertEqual(parse_status("Need to wait human action"), STATUS_WAIT_HUMAN)

    def test_extract_reason_handles_wait_human(self):
        reason = extract_reason("STATUS=WAIT_HUMAN\nREASON=User must pour the ingredient.")

        self.assertEqual(reason, "User must pour the ingredient.")

    def test_clean_statuses_accepts_wait_human(self):
        statuses = clean_statuses(["running", "WAIT_HUMAN", "wat"])

        self.assertEqual(statuses, ["RUNNING", "WAIT_HUMAN"])

    def test_missing_allowed_statuses_default_to_lerobot_compatible_statuses(self):
        self.assertEqual(clean_statuses(None), ["RUNNING", "SUCCESS", "FAILURE", "WAIT_HUMAN"])

    def test_fit_status_can_return_wait_human_when_it_is_only_allowed_status(self):
        self.assertEqual(fit_status("RUNNING", ["WAIT_HUMAN"]), "WAIT_HUMAN")

    def test_wait_human_maps_to_running_when_not_allowed(self):
        payload = build_result_payload(
            {
                "skill_name": "ingredient_poured",
                "attempt_id": 4,
                "allowed_statuses": [STATUS_RUNNING, STATUS_SUCCESS, STATUS_FAILURE],
            },
            STATUS_WAIT_HUMAN,
            "User must pour the ingredient.",
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["status"], STATUS_RUNNING)
        self.assertEqual(payload["message"], "WAIT_HUMAN: User must pour the ingredient.")
        self.assertEqual(payload["semantic_status"], "NOT_SATISFIED")
        self.assertEqual(payload["control_action"], "KEEP_RUNNING")

    def test_success_has_explicit_semantic_and_control_contract(self):
        payload = build_result_payload(
            {"skill_name": "pick", "attempt_id": 2},
            STATUS_SUCCESS,
            "goal reached",
        )
        self.assertEqual(payload["semantic_status"], "SATISFIED")
        self.assertEqual(payload["control_action"], "STOP_SUCCESS")

    def test_wait_human_maps_to_running_even_if_running_was_not_listed(self):
        payload = build_result_payload(
            {
                "skill_name": "ingredient_poured",
                "attempt_id": 4,
                "allowed_statuses": [STATUS_SUCCESS, STATUS_FAILURE],
            },
            STATUS_WAIT_HUMAN,
            "User must pour the ingredient.",
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["status"], STATUS_RUNNING)

    def test_wait_human_is_published_when_allowed(self):
        payload = build_result_payload(
            {
                "skill_name": "ingredient_poured",
                "attempt_id": 4,
                "allowed_statuses": [STATUS_RUNNING, STATUS_SUCCESS, STATUS_FAILURE, STATUS_WAIT_HUMAN],
            },
            STATUS_WAIT_HUMAN,
            "User must pour the ingredient.",
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["status"], STATUS_WAIT_HUMAN)
        self.assertEqual(payload["message"], "User must pour the ingredient.")


class SceneContextEnrichmentTests(unittest.TestCase):
    _FACTS = (
        '{"facts": {"coffee_capsule": {"present": true, "frame_id": "base_link", '
        '"pose": {"translation": {"x": 0.684, "y": -0.26, "z": 0.15}}, '
        '"pose_confidence": 0.62}}}'
    )

    def test_format_scene_context_renders_present_pose(self):
        context = format_scene_context(self._FACTS)

        self.assertIn("coffee_capsule", context)
        self.assertIn("0.684", context)
        self.assertIn("base_link", context)
        self.assertIn("confidence 0.62", context)

    def test_format_scene_context_empty_for_blank_or_invalid(self):
        self.assertEqual(format_scene_context(""), "")
        self.assertEqual(format_scene_context("not-json"), "")
        self.assertEqual(format_scene_context('{"facts": {}}'), "")

    def test_format_scene_context_skips_absent_objects(self):
        facts = '{"facts": {"cup": {"present": false, "frame_id": "base_link"}}}'
        self.assertEqual(format_scene_context(facts), "")

    def test_build_prompt_includes_scene_context_when_present(self):
        context = format_scene_context(self._FACTS)
        prompt = build_prompt(
            {
                "skill_name": "pick_and_insert_capsule",
                "attempt_id": 1,
                "message": "Check capsule inserted.",
                "scene_context": context,
            },
            reasoning=False,
        )

        self.assertIn("Perception scene facts", prompt)
        self.assertIn("coffee_capsule", prompt)

    def test_build_prompt_unchanged_without_scene_context(self):
        prompt = build_prompt(
            {
                "skill_name": "pick_and_insert_capsule",
                "attempt_id": 1,
                "message": "Check capsule inserted.",
            },
            reasoning=False,
        )

        self.assertNotIn("Perception scene facts", prompt)


if __name__ == "__main__":
    unittest.main()
