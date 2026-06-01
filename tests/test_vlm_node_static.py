import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = REPO_ROOT / "vlm_live/node.py"


class VlmNodeStaticTests(unittest.TestCase):
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
