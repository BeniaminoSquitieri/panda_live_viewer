"""
Manual check for dry-run planner and parser.
Run: python3 -m bt_planning.manual_check
"""
import json
import tempfile
import os
from bt_planning.dry_run_plan import make_dummy_plan, extract_names
from bt_planning.plan_parser import parse_linear_ir_plan, PlanParseError

# Minimal valid registry for make_sandwich
registry = {
    "robot_skills": ["place_first_toast", "place_second_toast"],
    "human_steps": ["pour_ingredient"],
    "vlm_gates": ["initial_scene_ready", "ingredient_poured", "second_toast_ready", "make_sandwich.task_complete"]
}

def test_dry_run_valid():
    plan = make_dummy_plan("make_sandwich", registry)
    assert all("kind" in s and "name" in s for s in plan["steps"]), "All steps must have kind and name"
    assert all("type" not in s for s in plan["steps"]), "No step should have 'type'"
    assert plan["steps"][0]["kind"] == "vlm_gate"
    print("dry-run valid plan output:\n", json.dumps(plan, indent=2))
    # Parser should accept
    parsed = parse_linear_ir_plan(json.dumps(plan))
    assert parsed["task_name"] == "make_sandwich"
    print("Parser accepted valid plan.")

def test_dry_run_missing():
    bad_registry = {"robot_skills": ["place_first_toast"], "human_steps": [], "vlm_gates": []}
    try:
        make_dummy_plan("make_sandwich", bad_registry)
        assert False, "Should fail if required names missing"
    except RuntimeError as e:
        print("Correctly failed on missing registry entries:", e)

def test_parser_xml():
    xml = "<root><step kind=\"robot_skill\" name=\"foo\"/></root>"
    try:
        parse_linear_ir_plan(xml)
        assert False, "Should reject XML"
    except PlanParseError as e:
        print("Correctly rejected XML:", e)

def test_parser_type():
    plan = {"task_name": "foo", "steps": [{"type": "robot_skill", "name": "foo"}]}
    try:
        parse_linear_ir_plan(json.dumps(plan))
        assert False, "Should reject step with 'type'"
    except PlanParseError as e:
        print("Correctly rejected 'type' field:", e)

def test_parser_missing_name():
    plan = {"task_name": "foo", "steps": [{"kind": "robot_skill"}]}
    try:
        parse_linear_ir_plan(json.dumps(plan))
        assert False, "Should reject step missing 'name'"
    except PlanParseError as e:
        print("Correctly rejected missing name:", e)

if __name__ == "__main__":
    test_dry_run_valid()
    test_dry_run_missing()
    test_parser_xml()
    test_parser_type()
    test_parser_missing_name()
    print("Manual checks completed.")
