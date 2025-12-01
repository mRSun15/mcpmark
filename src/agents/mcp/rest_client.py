"""
REST API Client for Filesystem MCP Server
==========================================

Simple HTTP client for the filesystem REST API.
"""

import json
from typing import Any, Dict, List, Optional

import aiohttp


class MCPRestClient:
    """Simple REST client for filesystem MCP server."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30):
        self.base_url = url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    async def start(self):
        """Initialize HTTP session."""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )

    async def stop(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
        self.session = None
        self._tools_cache = None

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        if self._tools_cache is not None:
            return self._tools_cache
            
        if not self.session:
            raise RuntimeError("REST client not started")

        url = f"{self.base_url}/tools"
        async with self.session.get(url) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Handle response format
            if isinstance(data, dict) and "tools" in data:
                tools = data["tools"]
            elif isinstance(data, list):
                tools = data
            else:
                raise ValueError(f"Unexpected response format: {data}")
            
            self._tools_cache = tools
            return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool."""
        if not self.session:
            raise RuntimeError("REST client not started")

        url = f"{self.base_url}/mcp/tools/{name}"
        
        try:
            async with self.session.post(
                url,
                json=arguments,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                # Wrap result in MCP format
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result) if not isinstance(result, str) else result
                        }
                    ],
                    "isError": False
                }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True
            }


