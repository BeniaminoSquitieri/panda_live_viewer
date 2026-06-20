import ast
import builtins
import importlib
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = REPO_ROOT / "vlm_live/node.py"
CLI_PATH = REPO_ROOT / "vlm_live/cli.py"


class VlmNodeStaticTests(unittest.TestCase):
    def test_node_module_has_no_top_level_model_import(self):
        module = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
        top_level_model_imports = [
            node
            for node in module.body
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "model"
        ]

        self.assertFalse(
            top_level_model_imports,
            "vlm_live.node must not import vlm_live.model at module-load time",
        )

    def test_lazy_load_model_defers_model_loading_in_init(self):
        module = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
        init_fn = _class_method(module, "VlmNode", "__init__")
        load_guards = [
            node
            for node in ast.walk(init_fn)
            if isinstance(node, ast.If) and _is_not_self_lazy_load_model(node.test)
        ]

        self.assertTrue(load_guards, "VlmNode.__init__ must guard eager loading with lazy_load_model")
        self.assertTrue(
            any(_calls_self_method(node, "_ensure_vlm_loaded") for node in load_guards),
            "lazy_load_model=False should call _ensure_vlm_loaded in __init__",
        )
        self.assertFalse(
            any(_calls_name(node, "load_model") for node in ast.walk(init_fn)),
            "VlmNode.__init__ should not call load_model directly",
        )

    def test_import_node_succeeds_when_qwen_dependency_is_missing(self):
        module_name = "vlm_live.node"
        previous_module = sys.modules.pop(module_name, None)
        original_import = builtins.__import__
        stubbed_modules = _install_minimal_ros_stubs()

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "qwen_vl_utils" or name.startswith("qwen_vl_utils."):
                raise ModuleNotFoundError("No module named 'qwen_vl_utils'")
            return original_import(name, globals, locals, fromlist, level)

        try:
            builtins.__import__ = guarded_import
            module = importlib.import_module(module_name)
            self.assertTrue(hasattr(module, "VlmNode"))
        finally:
            builtins.__import__ = original_import
            sys.modules.pop(module_name, None)
            for stub_name in reversed(stubbed_modules):
                sys.modules.pop(stub_name, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module

    def test_cli_has_main_guard(self):
        source = CLI_PATH.read_text(encoding="utf-8")
        self.assertIn("__main__", source, "vlm_live/cli.py must contain an __main__ guard")

    def test_cli_uses_multithreaded_executor(self):
        source = CLI_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "MultiThreadedExecutor",
            source,
            "vlm_live/cli.py must use MultiThreadedExecutor",
        )

    def test_cli_calls_spin(self):
        source = CLI_PATH.read_text(encoding="utf-8")
        self.assertTrue(
            ".spin(" in source or "executor.spin(" in source,
            "vlm_live/cli.py must call executor.spin()",
        )


def _install_minimal_ros_stubs() -> list[str]:
    """Install minimal ROS-related modules so node import can run in unit tests."""
    created: list[str] = []

    def ensure_module(name: str) -> types.ModuleType:
        existing = sys.modules.get(name)
        if existing is not None:
            return existing
        module = types.ModuleType(name)
        sys.modules[name] = module
        created.append(name)
        return module

    rclpy_mod = ensure_module("rclpy")
    if not hasattr(rclpy_mod, "ok"):
        setattr(rclpy_mod, "ok", lambda: False)

    rclpy_node_mod = ensure_module("rclpy.node")
    if not hasattr(rclpy_node_mod, "Node"):
        class _DummyNode:  # pragma: no cover - constructor behavior is not exercised.
            pass

        setattr(rclpy_node_mod, "Node", _DummyNode)
    setattr(rclpy_mod, "node", rclpy_node_mod)

    callback_groups_mod = ensure_module("rclpy.callback_groups")
    if not hasattr(callback_groups_mod, "ReentrantCallbackGroup"):
        class _DummyReentrantCallbackGroup:  # pragma: no cover
            pass

        setattr(callback_groups_mod, "ReentrantCallbackGroup", _DummyReentrantCallbackGroup)
    setattr(rclpy_mod, "callback_groups", callback_groups_mod)

    qos_mod = ensure_module("rclpy.qos")
    if not hasattr(qos_mod, "qos_profile_sensor_data"):
        setattr(qos_mod, "qos_profile_sensor_data", object())
    setattr(rclpy_mod, "qos", qos_mod)

    sensor_msgs_mod = ensure_module("sensor_msgs")
    sensor_msgs_msg_mod = ensure_module("sensor_msgs.msg")
    if not hasattr(sensor_msgs_msg_mod, "CompressedImage"):
        class _DummyCompressedImage:  # pragma: no cover
            data = b""

        setattr(sensor_msgs_msg_mod, "CompressedImage", _DummyCompressedImage)
    setattr(sensor_msgs_mod, "msg", sensor_msgs_msg_mod)

    std_msgs_mod = ensure_module("std_msgs")
    std_msgs_msg_mod = ensure_module("std_msgs.msg")
    if not hasattr(std_msgs_msg_mod, "String"):
        class _DummyString:  # pragma: no cover
            data = ""

        setattr(std_msgs_msg_mod, "String", _DummyString)
    setattr(std_msgs_mod, "msg", std_msgs_msg_mod)

    return created


def _class_method(module: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"Method {class_name}.{method_name} not found")


def _is_not_self_lazy_load_model(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Attribute)
        and node.operand.attr == "lazy_load_model"
        and isinstance(node.operand.value, ast.Name)
        and node.operand.value.id == "self"
    )


def _calls_self_method(node: ast.AST, method_name: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == method_name
        and isinstance(child.func.value, ast.Name)
        and child.func.value.id == "self"
        for child in ast.walk(node)
    )


def _calls_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == name
        for child in ast.walk(node)
    )


if __name__ == "__main__":
    unittest.main()
