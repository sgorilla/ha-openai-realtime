import ast
from pathlib import Path
import unittest


APP_DIR = Path(__file__).resolve().parents[1] / "app"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _is_call(node: ast.AST, owner: str, method: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
    )


class PipelineLifecycleTests(unittest.TestCase):
    def test_pipeline_builder_does_not_start_runner(self):
        tree = ast.parse((APP_DIR / "websocket_handler.py").read_text())
        build_pipeline = _function(tree, "build_pipeline")

        self.assertFalse(
            any(_is_call(node, "asyncio", "create_task") for node in ast.walk(build_pipeline))
        )

    def test_application_owns_exactly_one_runner_start(self):
        tree = ast.parse((APP_DIR / "main.py").read_text())
        run = _function(tree, "run")
        runner_starts = [
            node
            for node in ast.walk(run)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "runner"
        ]

        self.assertEqual(len(runner_starts), 1)

    def test_client_connect_reuses_pipeline_service(self):
        tree = ast.parse((APP_DIR / "main.py").read_text())
        on_client_connected = _function(tree, "on_client_connected")

        called_methods = {
            node.func.attr
            for node in ast.walk(on_client_connected)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("_ensure_openai_service", called_methods)
        self.assertIn("set_current_service", called_methods)


if __name__ == "__main__":
    unittest.main()
