"""MCP service integration using Pipecat's MCPClient with StreamableHTTP."""
from collections import Counter
from dataclasses import dataclass, replace
import logging
from typing import Any, Optional, Sequence

from pipecat.services.mcp_service import MCPClient, StreamableHttpParameters

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPToolBinding:
    """The model-facing name and the exact name understood by HA's MCP server."""

    exposed_name: str
    source_name: str
    schema: Any


class HomeAssistantMCPService:
    """Home Assistant MCP service using Pipecat's MCPClient."""
    
    def __init__(self, url: str, access_token: str):
        """
        Initialize Home Assistant MCP service.
        
        Args:
            url: Home Assistant MCP Server URL (e.g., http://supervisor/core/api/mcp)
            access_token: Long-lived access token for Home Assistant
        """
        self.url = url
        self.access_token = access_token
        self.mcp_client: Optional[MCPClient] = None

    @staticmethod
    def _short_tool_name(source_name: str) -> str:
        """Return HA's public tool name without its MCP namespace.

        Home Assistant publishes names such as ``llm__GetDateTime`` and
        ``homeassistant__GetLiveContext``. The Realtime API presents/calls those
        as ``GetDateTime`` and ``GetLiveContext``. Retaining the source name for
        the MCP call is therefore essential.
        """
        return source_name.rsplit("__", 1)[-1]

    def build_tool_bindings(
        self,
        tools_schema: Any,
        allowlist: Sequence[str] = (),
    ) -> list[MCPToolBinding]:
        """Build unambiguous model-name -> HA-name bindings.

        The allow-list accepts either the clean public name documented by the
        add-on (``GetDateTime``) or HA's exact namespaced MCP name
        (``llm__GetDateTime``).
        """
        schemas = list(tools_schema.standard_tools)
        short_names = [self._short_tool_name(schema.name) for schema in schemas]
        short_name_counts = Counter(short_names)
        allowed = set(allowlist)
        bindings: list[MCPToolBinding] = []

        for schema, short_name in zip(schemas, short_names):
            # Namespace removal is what the Realtime API already does. If HA
            # ever publishes two tools with the same public name, retain a
            # single-underscore namespace so both remain callable and valid as
            # OpenAI function names.
            exposed_name = (
                short_name
                if short_name_counts[short_name] == 1
                else schema.name.replace("__", "_")
            )
            if allowed and not ({schema.name, short_name, exposed_name} & allowed):
                continue
            bindings.append(
                MCPToolBinding(
                    exposed_name=exposed_name,
                    source_name=schema.name,
                    schema=schema,
                )
            )

        return bindings

    async def register_tool_bindings(
        self,
        bindings: Sequence[MCPToolBinding],
        llm: Any,
    ) -> None:
        """Register aliases that translate model calls back to exact HA names."""
        if self.mcp_client is None:
            raise RuntimeError("Home Assistant MCP client is not initialized")

        for binding in bindings:
            llm.register_function(
                binding.exposed_name,
                self._make_tool_handler(binding),
            )

    def _make_tool_handler(self, binding: MCPToolBinding):
        """Create a one-argument Pipecat handler for one namespaced HA tool."""
        if self.mcp_client is None:
            raise RuntimeError("Home Assistant MCP client is not initialized")
        mcp_client = self.mcp_client

        async def call_namespaced_tool(params):
            logger.info(
                "Mapping Home Assistant tool %r to MCP tool %r",
                binding.exposed_name,
                binding.source_name,
            )
            translated_params = replace(params, function_name=binding.source_name)
            await mcp_client._tool_wrapper(translated_params)

        return call_namespaced_tool
        
    async def initialize(self) -> MCPClient:
        """Initialize and return the MCP client."""
        try:
            logger.info(f"🔗 Initializing Home Assistant MCP Client at {self.url}")
            
            # Create StreamableHTTP parameters with authentication
            server_params = StreamableHttpParameters(
                url=self.url,
                headers={
                    "Authorization": f"Bearer {self.access_token}"
                }
            )
            
            # Create MCP client
            self.mcp_client = MCPClient(server_params=server_params)
            
            logger.info("✅ Home Assistant MCP Client initialized")
            return self.mcp_client
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Home Assistant MCP Client: {e}", exc_info=True)
            raise
    
    def get_client(self) -> Optional[MCPClient]:
        """Get the MCP client instance."""
        return self.mcp_client





