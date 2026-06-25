import unittest

from vlm_live.const import GENERATE_PLAN_SERVICE, GROUND_INSTRUCTION_SERVICE


class RosContractGroundingTests(unittest.TestCase):
    """Pin this repo's side of the grounding ROS contract.

    The wire string is shared by VLM-BT-BC (bt_generation/nl_grounding_client.py
    GROUND_INSTRUCTION_SERVICE) and documented in docs/VLM_BT_BC_ARCHITETTURA.md. A rename
    here fails silently at runtime, so this test must go red on drift; update both
    repos and the contract doc together.
    """

    def test_ground_instruction_service_name_matches_contract(self):
        self.assertEqual(GROUND_INSTRUCTION_SERVICE, "/lerobot_bt/ground_instruction")

    def test_generate_plan_service_name_unchanged(self):
        self.assertEqual(GENERATE_PLAN_SERVICE, "/lerobot_bt/generate_plan")


if __name__ == "__main__":
    unittest.main()
