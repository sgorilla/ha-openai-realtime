import asyncio
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

from app.memory_store import MemoryStore
from app.memory_tool import create_memory_tool_handlers, get_memory_tool_definitions


@dataclass
class _Params:
    arguments: dict
    result_callback: object


class MemoryToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp_dir.name) / "memory.sqlite3")
        self.handlers = create_memory_tool_handlers(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _invoke(handler, arguments):
        results = []

        async def result_callback(value):
            results.append(value)

        asyncio.run(handler(_Params(arguments, result_callback)))
        return results

    def test_definitions_require_explicit_save_and_exact_delete(self):
        definitions = {item["name"]: item for item in get_memory_tool_definitions()}
        self.assertEqual(set(definitions), {"save_memory", "recall_memories", "forget_memory"})
        self.assertIn("ONLY when the user explicitly asks", definitions["save_memory"]["description"])
        self.assertEqual(
            definitions["forget_memory"]["parameters"]["required"],
            ["memory_id"],
        )

    def test_save_recall_forget_round_trip_without_logging_content(self):
        with self.assertLogs("app.memory_tool", level="INFO") as captured:
            saved_result = self._invoke(
                self.handlers["save_memory"],
                {"content": "Eric likes jasmine tea", "category": "preference"},
            )
        self.assertIn("Saved as memory #", saved_result[0])
        self.assertNotIn("jasmine", " ".join(captured.output))

        recalled = self._invoke(
            self.handlers["recall_memories"], {"query": "jasmine", "limit": 10}
        )
        self.assertIn("Eric likes jasmine tea", recalled[0])

        memory_id = self.store.recall("jasmine")[0].id
        forgotten = self._invoke(
            self.handlers["forget_memory"], {"memory_id": memory_id}
        )
        self.assertIn("was deleted", forgotten[0])
        self.assertEqual(self.store.count(), 0)

    def test_store_failure_is_reported_without_leaking_exception_text(self):
        class _BrokenStore:
            def save(self, content, category):
                raise RuntimeError("private-memory-content")

            def recall(self, query, limit):
                raise RuntimeError("private-memory-content")

            def forget(self, memory_id):
                raise RuntimeError("private-memory-content")

        handlers = create_memory_tool_handlers(_BrokenStore())
        with self.assertLogs("app.memory_tool", level="ERROR") as captured:
            result = self._invoke(
                handlers["save_memory"],
                {"content": "not logged", "category": "other"},
            )

        self.assertIn("temporarily unavailable", result[0])
        self.assertNotIn("private-memory-content", " ".join(captured.output))


if __name__ == "__main__":
    unittest.main()
