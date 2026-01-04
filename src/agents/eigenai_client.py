"""
EigenAI Direct HTTP Client
===========================

Direct HTTP client for EigenAI API, bypassing LiteLLM to avoid Cloudflare blocks.
Uses the same interface as LiteLLM for easy integration.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import aiohttp

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    def model_dump(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Function:
    """Function call details."""
    name: str
    arguments: str
    
    def model_dump(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class ToolCall:
    """Tool call from the model."""
    id: str
    type: str
    function: Function
    
    def model_dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.model_dump(),
        }


@dataclass
class Message:
    """Chat message."""
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    
    def model_dump(self) -> Dict[str, Any]:
        result = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        return result


@dataclass
class Choice:
    """Completion choice."""
    index: int
    message: Message
    finish_reason: str
    
    def model_dump(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "message": self.message.model_dump(),
            "finish_reason": self.finish_reason,
        }


@dataclass
class CompletionResponse:
    """Completion response matching LiteLLM's format."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    
    def model_dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.model_dump() for c in self.choices],
            "usage": self.usage.model_dump(),
        }


class EigenAIClient:
    """Direct HTTP client for EigenAI API."""

    DEFAULT_BASE_URL = "https://train.eigenai.com/api/v1"
    DEFAULT_MODEL = "deepseek-v31-terminus"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 300,
    ):
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        

    async def acompletion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        **kwargs,
    ) -> CompletionResponse:
        """
        Async chat completion matching LiteLLM's interface.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            tool_choice: Optional tool choice setting
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters (ignored for compatibility)
        
        Returns:
            CompletionResponse matching LiteLLM's format
        """
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.4.0",  # Match curl to avoid Cloudflare
            "Accept": "*/*",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 1,
            "stream": False,
            "chat_template_kwargs": {
                "thinking": False,
            },
        }
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"EigenAI API error {response.status}: {error_text}")
                
                data = await response.json()
        
        
        # Parse response into LiteLLM-compatible format
        choices = []
        for i, choice in enumerate(data.get("choices", [])):
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls")
            
            # Parse tool calls if present - create proper ToolCall objects
            parsed_tool_calls = None
            if tool_calls:
                parsed_tool_calls = []
                for j, tc in enumerate(tool_calls):
                    func_data = tc.get("function", {})
                    parsed_tool_calls.append(ToolCall(
                        id=tc.get("id", f"call_{i}_{j}"),
                        type="function",
                        function=Function(
                            name=func_data.get("name", ""),
                            arguments=func_data.get("arguments", "{}"),
                        )
                    ))
            
            choices.append(Choice(
                index=i,
                message=Message(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                    tool_calls=parsed_tool_calls,
                ),
                finish_reason=choice.get("finish_reason", "stop"),
            ))
        
        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        
        return CompletionResponse(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", self.model),
            choices=choices,
            usage=usage,
        )

    def completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        **kwargs,
    ) -> CompletionResponse:
        """Sync wrapper for acompletion."""
        return asyncio.get_event_loop().run_until_complete(
            self.acompletion(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        )


def is_eigenai_model(model_name: str) -> bool:
    """Check if the model is an EigenAI model."""
    return "eigenai" in model_name.lower() or "deepseek-v31" in model_name.lower()

