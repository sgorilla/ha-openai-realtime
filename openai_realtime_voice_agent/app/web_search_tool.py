"""Web search function-tool.

Lets the Realtime assistant look things up online (weather, news, facts, opening
hours, prices, recent events). The Realtime API has NO native web search, and
pipecat 0.0.97 only supports custom *function* tools — so this is wired exactly
like the disconnect tool: a `web_search` function tool whose handler runs a
SECOND, server-side OpenAI call (the Responses API `web_search` built-in tool),
then returns a short, spoken-friendly answer the Realtime model reads aloud.

Uses the add-on's existing OPENAI_API_KEY — no extra account/vendor. Default
model gpt-5.5 gives the best-quality search answers; it's configurable via
WEB_SEARCH_MODEL (the mini/nano models are cheaper) so a different price/quality
(or a renamed model) can be swapped in without a code change.
"""
import logging
from typing import Dict, Any, Callable, Awaitable, TYPE_CHECKING

from openai import AsyncOpenAI
from app.home_location import HomeLocation, query_needs_coordinates

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

logger = logging.getLogger(__name__)


def get_web_search_tool_definition() -> Dict[str, Any]:
    """OpenAI Realtime function-tool definition for web search."""
    return {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the public internet for current, real-time, or factual "
            "information the assistant does not already know — for example the "
            "weather, news, sports scores, opening hours, prices, travel info, or "
            "recent events. When home-location sharing is enabled, the tool "
            "automatically receives the Home Assistant home location, so preserve "
            "phrases such as 'near me' and 'around here' instead of asking the "
            "user for a city. Do NOT use this for "
            "controlling the smart home — use the Hass* tools for lights, "
            "switches, climate, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query, phrased as a clear natural-language "
                        "question in the user's language."
                    ),
                }
            },
            "required": ["query"],
        },
    }


def build_web_search_tool(home_location: HomeLocation | None) -> Dict[str, Any]:
    """Build the official Responses web-search tool with approximate location."""
    tool: Dict[str, Any] = {"type": "web_search"}
    if home_location:
        user_location = home_location.web_search_user_location()
        if user_location:
            tool["user_location"] = user_location
    return tool


def build_web_search_input(query: str, home_location: HomeLocation | None) -> str:
    """Build spoken-answer instructions, adding rounded coordinates only locally."""
    location_context = ""
    if home_location and query_needs_coordinates(query):
        coordinates = home_location.rounded_coordinates()
        if coordinates:
            location_context = (
                " This is a location-sensitive request. Resolve here/nearby/local "
                "against the user's Home Assistant home at approximate coordinates "
                f"{coordinates[0]:.2f}, {coordinates[1]:.2f}. Use the coordinates "
                "for search relevance but do not state or expose them in the answer."
            )

    return (
        "Answer in at most 2 short sentences suitable for being read aloud, "
        "in the same language as the question. Do not include URLs, citations, "
        "or markdown."
        + location_context
        + " Question: "
        + query
    )


def create_web_search_tool_handler(
    api_key: str,
    model: str,
    home_location: HomeLocation | None = None,
    *,
    client: Any | None = None,
) -> Callable[["FunctionCallParams"], Awaitable[None]]:
    """Create a web_search handler for pipecat's OpenAIRealtimeLLMService.

    The handler calls the OpenAI Responses API with the built-in `web_search`
    tool and returns a short answer via ``params.result_callback`` (which becomes
    the function_call_output the Realtime model speaks).
    """
    client = client or AsyncOpenAI(api_key=api_key)

    async def web_search_tool_handler(params: "FunctionCallParams") -> None:
        query = (params.arguments or {}).get("query", "").strip()
        logger.info(
            "🔎 web_search called (model=%s, location_sensitive=%s, query_chars=%d)",
            model,
            query_needs_coordinates(query),
            len(query),
        )

        if not query:
            await params.result_callback("Geen zoekopdracht ontvangen.")
            return

        try:
            response = await client.responses.create(
                model=model,
                tools=[build_web_search_tool(home_location)],
                input=build_web_search_input(query, home_location),
            )
            answer = (getattr(response, "output_text", None) or "").strip()
            logger.info("🔎 web_search answer received (%d chars)", len(answer))
            await params.result_callback(answer or "Ik kon hier online niets over vinden.")
        except Exception as e:
            # Some SDK errors contain request details. Do not risk placing
            # location context (or authorization data) in add-on logs.
            logger.error("❌ web_search failed (%s)", type(e).__name__)
            await params.result_callback("Het zoeken op internet lukte even niet.")

    return web_search_tool_handler
