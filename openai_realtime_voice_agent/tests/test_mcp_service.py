import asyncio
from dataclasses import dataclass
import sys
import types
import unittest


# The production image supplies Pipecat. Keep these adapter unit tests runnable
# in a plain checkout without installing the large audio/runtime dependency.
try:
    from app.mcp_service import HomeAssistantMCPService
except ModuleNotFoundError as exc:
    if exc.name != "pipecat":
        raise

    mcp_module = types.ModuleType("pipecat.services.mcp_service")
    mcp_module.MCPClient = object
    mcp_module.StreamableHttpParameters = object
    sys.modules["pipecat"] = types.ModuleType("pipecat")
    sys.modules["pipecat.services"] = types.ModuleType("pipecat.services")
    sys.modules["pipecat.services.mcp_service"] = mcp_module

    from app.mcp_service import HomeAssistantMCPService


@dataclass
class _Schema:
    name: str
    description: str = "test tool"
    properties: dict | None = None
    required: list | None = None


@dataclass
class _Params:
    function_name: str
    tool_call_id: str = "call-1"
    arguments: dict | None = None
    llm: object | None = None
    context: object | None = None
    result_callback: object | None = None


class _FakeMCPClient:
    def __init__(self):
        self.calls = []

    async def _tool_wrapper(self, params):
        self.calls.append(params)


class _FakeLLM:
    def __init__(self):
        self.handlers = {}

    def register_function(self, name, handler):
        self.handlers[name] = handler


class HomeAssistantMCPServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = HomeAssistantMCPService("http://example.invalid", "token")
        self.client = _FakeMCPClient()
        self.service.mcp_client = self.client

    @staticmethod
    def _tools(*names):
        return types.SimpleNamespace(standard_tools=[_Schema(name) for name in names])

    def test_builds_clean_aliases_and_accepts_clean_allowlist(self):
        bindings = self.service.build_tool_bindings(
            self._tools("llm__GetDateTime", "homeassistant__GetLiveContext"),
            ["GetDateTime"],
        )

        self.assertEqual(
            [(binding.exposed_name, binding.source_name) for binding in bindings],
            [("GetDateTime", "llm__GetDateTime")],
        )

    def test_duplicate_short_names_keep_unambiguous_namespaces(self):
        bindings = self.service.build_tool_bindings(
            self._tools("one__Shared", "two__Shared")
        )

        self.assertEqual(
            [binding.exposed_name for binding in bindings],
            ["one_Shared", "two_Shared"],
        )

    def test_registered_alias_calls_exact_namespaced_mcp_tool(self):
        binding = self.service.build_tool_bindings(
            self._tools("llm__GetDateTime")
        )[0]
        llm = _FakeLLM()
        asyncio.run(self.service.register_tool_bindings([binding], llm))

        original = _Params(function_name="GetDateTime", arguments={})
        asyncio.run(llm.handlers["GetDateTime"](original))

        self.assertEqual(original.function_name, "GetDateTime")
        self.assertEqual(self.client.calls[0].function_name, "llm__GetDateTime")


if __name__ == "__main__":
    unittest.main()
