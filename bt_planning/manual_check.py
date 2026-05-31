"""
Manual check for dry-run planner and parser.
Run: python3 -m bt_planning.manual_check
"""
import json
from bt_planning.dry_run_plan import make_dummy_plan
from bt_planning.plan_parser import parse_linear_ir_plan, PlanParseError
from bt_planning.prompt_builder import build_planner_prompt
from bt_planning.service_logic import build_generate_plan_response

registry = {
    "robot_skills": ["place_first_toast", "place_second_toast"],
    "human_steps": ["pour_ingredient"],
    "vlm_gates": [
        "initial_scene_ready",
        "ingredient_poured",
        "second_toast_ready",
        "make_sandwich.task_complete",
    ],
    "canonical_task_sequence": [
        {"kind": "vlm_gate", "name": "initial_scene_ready"},
        {"kind": "robot_skill", "name": "place_first_toast"},
        {"kind": "human_step", "name": "pour_ingredient"},
        {"kind": "vlm_gate", "name": "ingredient_poured"},
        {"kind": "vlm_gate", "name": "second_toast_ready"},
        {"kind": "robot_skill", "name": "place_second_toast"},
        {"kind": "vlm_gate", "name": "make_sandwich.task_complete"},
    ],
    "ordering_constraints": [
        {"before": "place_first_toast", "after": "initial_scene_ready"},
    ],
}


def test_prompt_canonical_sequence():
    prompt = build_planner_prompt("make_sandwich", registry, {"front_camera": "toast visible"})
    assert "canonical_task_sequence" in prompt
    assert "authoritative BT leaf order" in prompt
    assert "Follow canonical_task_sequence exactly" in prompt
    assert "Do not reorder steps" in prompt
    assert "Do not add steps" in prompt
    assert "Do not remove steps" in prompt
    assert "Use only kind/name pairs from canonical_task_sequence" in prompt
    assert "Do not infer order from object names" in prompt
    assert "ordering_constraints" in prompt
    print("Prompt includes canonical sequence and ordering constraints.")


def test_dry_run_valid():
    plan = make_dummy_plan("make_sandwich", registry)
    assert all("kind" in s and "name" in s for s in plan["steps"]), "All steps must have kind and name"
    assert all("type" not in s for s in plan["steps"]), "No step should have 'type'"
    assert plan["steps"] == registry["canonical_task_sequence"]
    print("dry-run valid plan output:\n", json.dumps(plan, indent=2))
    parsed = parse_linear_ir_plan(json.dumps(plan))
    assert parsed["task_name"] == "make_sandwich"
    print("Parser accepted valid plan.")


def test_dry_run_missing_canonical_sequence():
    bad_registry = {
        "robot_skills": ["place_first_toast"],
        "human_steps": [],
        "vlm_gates": [],
    }
    try:
        make_dummy_plan("make_sandwich", bad_registry)
        assert False, "Should fail if canonical_task_sequence is missing"
    except RuntimeError as e:
        print("Correctly failed on missing canonical sequence:", e)


def test_dry_run_unlisted_step():
    bad_registry = dict(registry)
    bad_registry["robot_skills"] = ["place_first_toast"]
    try:
        make_dummy_plan("make_sandwich", bad_registry)
        assert False, "Should fail if canonical step is not listed in allowed vocabulary"
    except RuntimeError as e:
        print("Correctly failed on unlisted canonical step:", e)


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


def test_service_dry_run():
    result = build_generate_plan_response(
        task_name="make_sandwich",
        planner_registry_json=json.dumps(registry),
        scene_facts_json=json.dumps({"front_camera": "toast visible"}),
        dry_run=True,
    )
    assert result.success, result.error_message
    parsed = parse_linear_ir_plan(result.plan_json)
    assert parsed["steps"] == registry["canonical_task_sequence"]
    print("Service dry-run returned success=true and valid plan_json.")


def test_service_malformed_registry():
    result = build_generate_plan_response(
        task_name="make_sandwich",
        planner_registry_json="{bad json",
        dry_run=True,
    )
    assert not result.success
    assert result.plan_json == ""
    assert "Malformed planner_registry_json" in result.error_message
    print("Service correctly failed malformed planner_registry_json.")


if __name__ == "__main__":
    test_prompt_canonical_sequence()
    test_dry_run_valid()
    test_dry_run_missing_canonical_sequence()
    test_dry_run_unlisted_step()
    test_parser_xml()
    test_parser_type()
    test_parser_missing_name()
    test_service_dry_run()
    test_service_malformed_registry()
    print("Manual checks completed.")
