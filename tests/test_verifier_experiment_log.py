import ast
import json
import tempfile
import unittest
from pathlib import Path

from vlm_live.experiment_log import (
    EVENT_TYPE_VERIFIER,
    append_verifier_event,
    build_verifier_event,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = REPO_ROOT / "vlm_live/node.py"


class VerifierExperimentLogTests(unittest.TestCase):
    def test_append_writes_one_json_line(self):
        event = build_verifier_event(
            skill_name="pick_bread",
            attempt_id=1,
            allowed_statuses=["running", "success"],
            raw_status="success",
            published_status="success",
            reason="bread is on the plate",
            was_wait_human_coerced=False,
            front_frame_available=True,
            wrist_frame_available=True,
            duration_s=0.42,
            model_path="/models/qwen",
            dry_run_planner=False,
            error_message=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verifier_events.jsonl"
            append_verifier_event(path, event)
            append_verifier_event(path, event)
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["event_type"], EVENT_TYPE_VERIFIER)
        self.assertEqual(first["skill_name"], "pick_bread")
        self.assertEqual(first["published_status"], "success")
        self.assertFalse(first["was_wait_human_coerced"])

    def test_wait_human_coercion_is_representable(self):
        event = build_verifier_event(
            skill_name="pour_coffee",
            attempt_id=2,
            allowed_statuses=["running", "success", "failure"],
            raw_status="wait_human",
            published_status="running",
            reason="needs a human step",
            was_wait_human_coerced=True,
            front_frame_available=True,
            wrist_frame_available=False,
            duration_s=0.1,
        )
        self.assertEqual(event["raw_status"], "wait_human")
        self.assertEqual(event["published_status"], "running")
        self.assertTrue(event["was_wait_human_coerced"])
        # Verifier logs never carry image data, only availability flags.
        self.assertNotIn("front_frame", event)
        self.assertNotIn("image", event)

    def test_node_declares_verifier_log_parameter(self):
        module = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
        source = NODE_PATH.read_text(encoding="utf-8")
        self.assertIn("verifier_experiment_log_path", source)
        # Parameter is declared with a safe disabled-by-default empty string.
        declared = any(
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "declare_parameter"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "verifier_experiment_log_path"
            for node in ast.walk(module)
        )
        self.assertTrue(declared, "node.py must declare verifier_experiment_log_path parameter")


if __name__ == "__main__":
    unittest.main()
