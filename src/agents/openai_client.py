"""
Simple OpenAI Client for MCPMark
=================================

Direct OpenAI client for improved performance with ChatGPT models.
"""

from typing import Any, Dict, List, Optional

from src.logger import get_logger

logger = get_logger(__name__)


class SimpleOpenAIClient:
    """Simple wrapper around OpenAI chat completion API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        reasoning_effort: str = "default",
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        
        try:
            import openai
        except ModuleNotFoundError:
            raise ImportError("OpenAI package not installed. Please install with: pip install openai")
        
        # Initialize client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self._client = openai.AsyncOpenAI(**client_kwargs)

    async def acompletion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a chat completion."""
        model_lower = self.model.lower()
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
        }
        
        # Add tools if provided
        if tools:
            request_params["tools"] = tools
            if tool_choice:
                request_params["tool_choice"] = tool_choice
        
        # Handle token limits - newer models use max_completion_tokens
        if max_tokens:
            if "gpt-5" in model_lower or "gpt-4o" in model_lower or "o3" in model_lower or "o4" in model_lower:
                request_params["max_completion_tokens"] = max_tokens
            else:
                request_params["max_tokens"] = max_tokens
        
        # Add temperature if provided
        if temperature is not None:
            request_params["temperature"] = temperature
        
        # Add reasoning_effort for o-series models
        if ("o3" in model_lower or "o4" in model_lower) and self.reasoning_effort != "default":
            request_params["reasoning_effort"] = self.reasoning_effort
        
        # Make the API call
        response = await self._client.chat.completions.create(**request_params)
        return response

