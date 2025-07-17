"""MCP Server wrapper class for integration with ART."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class McpServer(ABC):
    """Abstract base class for MCP server integration."""

    def __init__(self):
        """Initialize MCP server wrapper."""
        self._session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._stdio_context = None
        self._session_context = None

    @abstractmethod
    async def init_server(self) -> StdioServerParameters:
        """Initialize server configuration.

        Returns:
            StdioServerParameters for the specific server implementation
        """
        pass

    async def start(self) -> None:
        """Start the MCP server and establish connection."""
        server_params = await self.init_server()

        # Start stdio client
        self._stdio_context = stdio_client(server_params)
        self._read, self._write = await self._stdio_context.__aenter__()

        # Start session
        self._session_context = ClientSession(self._read, self._write)
        self._session = await self._session_context.__aenter__()

        # Initialize session
        await self._session.initialize()

    async def stop(self) -> None:
        """Stop the MCP server and clean up connections."""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
            self._session_context = None
            self._session = None

        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)
            self._stdio_context = None
            self._read = None
            self._write = None

    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools in OpenAI format.

        Returns:
            List of tool schemas in OpenAI format
        """
        if not self._session:
            raise RuntimeError("Server not started. Call start() first.")

        # Get available tools from MCP server
        tools_result = await self._session.list_tools()

        # Convert to OpenAI format
        tool_schemas = []
        for tool in tools_result.tools:
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"MCP tool: {tool.name}",
                    "parameters": tool.inputSchema
                    or {"type": "object", "properties": {}},
                },
            }
            tool_schemas.append(tool_schema)

        return tool_schemas

    async def apply_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Apply a tool with given arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as string

        Raises:
            RuntimeError: If server not started
            Exception: If tool call fails
        """
        if not self._session:
            raise RuntimeError("Server not started. Call start() first.")

        try:
            result = await self._session.call_tool(tool_name, arguments)
            # Extract text content from MCP result
            if hasattr(result, "content") and result.content:
                if isinstance(result.content, list):
                    # Handle list of content items
                    content_text = ""
                    for item in result.content:
                        if hasattr(item, "text"):
                            content_text += item.text
                        else:
                            content_text += str(item)
                else:
                    content_text = (
                        result.content.text
                        if hasattr(result.content, "text")
                        else str(result.content)
                    )
            else:
                content_text = str(result)

            return content_text
        except Exception as e:
            raise Exception(f"Tool '{tool_name}' failed: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class AlphaMcpServer(McpServer):
    """AlphaVantage MCP server implementation."""

    def __init__(self, api_key: str = "demo"):
        """Initialize AlphaVantage MCP server.

        Args:
            api_key: AlphaVantage API key (defaults to 'demo')
        """
        super().__init__()
        self.api_key = api_key

    async def init_server(self) -> StdioServerParameters:
        """Initialize AlphaVantage server configuration.

        Returns:
            StdioServerParameters for AlphaVantage server
        """
        return StdioServerParameters(
            command="node",
            args=["servers/mcp-alphavantage/build/index.js"],
            env={"ALPHAVANTAGE_API_KEY": self.api_key},
        )
