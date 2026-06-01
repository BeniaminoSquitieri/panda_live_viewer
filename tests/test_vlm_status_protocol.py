import unittest

from vlm_live.const import STATUS_FAILURE, STATUS_RUNNING, STATUS_SUCCESS, STATUS_WAIT_HUMAN
from vlm_live.prompt import build_prompt, extract_reason, fit_status, parse_status
from vlm_live.protocol import build_result_payload, clean_statuses


class VlmStatusProtocolTests(unittest.TestCase):
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
        self.assertEqual(clean_statuses(None), ["RUNNING", "SUCCESS", "FAILURE"])

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


if __name__ == "__main__":
    unittest.main()
