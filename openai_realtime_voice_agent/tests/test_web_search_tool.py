import asyncio
from dataclasses import dataclass
import sys
import types
import unittest


# The production image supplies the OpenAI SDK. Keep these focused adapter
# tests runnable in a plain checkout without installing the runtime stack.
try:
    from app.web_search_tool import (
        build_web_search_input,
        build_web_search_tool,
        create_web_search_tool_handler,
    )
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    openai_module = types.ModuleType("openai")
    openai_module.AsyncOpenAI = object
    sys.modules["openai"] = openai_module
    from app.web_search_tool import (
        build_web_search_input,
        build_web_search_tool,
        create_web_search_tool_handler,
    )

from app.home_location import HomeLocation


class _FakeResponses:
    def __init__(self):
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return types.SimpleNamespace(output_text="Three good options are nearby.")


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


@dataclass
class _Params:
    arguments: dict
    result_callback: object


class WebSearchLocationTests(unittest.TestCase):
    def setUp(self):
        self.location = HomeLocation(
            location_name="Los Angeles",
            country="US",
            timezone="America/Los_Angeles",
            latitude=33.864321,
            longitude=-118.399876,
        )

    def test_official_user_location_shape_uses_valid_approximate_fields(self):
        self.assertEqual(
            build_web_search_tool(self.location),
            {
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                    "timezone": "America/Los_Angeles",
                    "city": "Los Angeles",
                },
            },
        )
        self.assertEqual(build_web_search_tool(None), {"type": "web_search"})

    def test_local_search_input_gets_rounded_not_exact_coordinates(self):
        search_input = build_web_search_input(
            "What coffee is open near me?", self.location
        )

        self.assertIn("33.86, -118.40", search_input)
        self.assertNotIn("33.864321", search_input)
        self.assertNotIn("-118.399876", search_input)

    def test_nonlocal_search_does_not_disclose_coordinates(self):
        search_input = build_web_search_input(
            "Who won the World Cup in 2018?", self.location
        )

        self.assertNotIn("33.86", search_input)
        self.assertNotIn("-118.40", search_input)

    def test_handler_passes_location_to_responses_web_search(self):
        client = _FakeClient()
        results = []

        async def result_callback(result):
            results.append(result)

        handler = create_web_search_tool_handler(
            "unused-in-test",
            "gpt-test",
            home_location=self.location,
            client=client,
        )
        with self.assertLogs("app.web_search_tool", level="INFO") as captured:
            asyncio.run(
                handler(
                    _Params(
                        arguments={"query": "Coffee near me at 33.864321"},
                        result_callback=result_callback,
                    )
                )
            )

        self.assertEqual(results, ["Three good options are nearby."])
        request = client.responses.requests[0]
        self.assertEqual(request["model"], "gpt-test")
        self.assertEqual(
            request["tools"][0]["user_location"]["country"], "US"
        )
        self.assertIn("33.86, -118.40", request["input"])
        # User/location values stay out of logs even when present in a query.
        self.assertNotIn("33.864321", " ".join(captured.output))


if __name__ == "__main__":
    unittest.main()
