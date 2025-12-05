"""
MCPMark Agent Implementation
============================

Unified agent using LiteLLM for all model interactions with minimal MCP support.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Callable

import httpx
import litellm
import nest_asyncio

from src.logger import get_logger
from .base_agent import BaseMCPAgent
from .mcp import MCPStdioServer, MCPHttpServer, MCPRestClient
from .openai_client import SimpleOpenAIClient

# Apply nested asyncio support
nest_asyncio.apply()

# Configure LiteLLM
litellm.suppress_debug_info = True

logger = get_logger(__name__)

class MCPMarkAgent(BaseMCPAgent):
    """
    Unified agent for LLM and MCP server management.

    - Claude models: Use LiteLLM
    - OpenAI models: Use direct OpenAI client for better performance
    """

    MAX_TURNS = 100
    # Critical tools that should trigger memory update when called
    CRITICAL_TOOLS_FOR_MEMORY = {
        "read_multiple_files",
        "directory_tree",
    }
    SYSTEM_PROMPT = (
        "You are a helpful agent that uses tools iteratively to complete the user's task, "
        "and when finished, provides the final answer or simply states \"Task completed\" without further tool calls. CRITICAL RULES: "
        "1. you should strictly follow the user's instructions, no extra inference or reasoning unless it is explicitly requested; "
        "2. memory is a compression summary of what you have done, key facts and taks plan, you may reference it for next tool calls. "
        "3. If memory contains VERIFIED CONSTRAINTS section, you MUST check it before each tool call and avoid violating those specific constraints. "
        "4. for time related tasks, if not specified, please use the time zone of GMT+0800 (China Standard Time). "
        "5. use code to solve problem if possible. "
        "6. Avoid unnecessary redundantly calling the same tool with identical arguments."
    )
    MEMORY_SYSTEM_PROMPT = (
        "You own the shared task memory—call update_memory to keep it accurate. "
        "Treat this memory as a compressed snapshot of everything the agent has done and still needs to do. "
        "Use the prior memory plus the newest tool outputs; you may apply straightforward logical implications, but never invent facts that are not supported by tools or logic, and structure the update exactly as follows:\n"
        "\n"
        "**OVERVIEW**\n"
        "- 1–2 sentences describing the current situation and goal.\n"
        "\n"
        "**TASK PLAN**\n"
        "- Bullet list of the current plan or next steps; reorder or edit as understanding changes.\n"
        "- Ensure plan aligns strictly with user instructions: do not add goals or make extra inferences/reasoning unless it is explicitly requested.\n"
        "- The task instructions might have a few typos or ambiguities, you may discover and correct them based on tool call results.\n"
        "\n"
        "**PROGRESS & GAPS**\n"
        "- Finished: actions completed with tool confirmation.\n"
        "- Outstanding: pending actions.\n"
        "- Blockers: explicit obstacles preventing progress.\n"
        "\n"
        "**PERSISTENT FACTS**\n"
        "- Bullet list for persistent metrics/identifiers (counts, aggregates, paths, key entities) related to the task.\n"
        "- Store concrete values (exact IDs, paths, thresholds), not general descriptions.\n"
        "- Use raw values without transformation - no custom conversions or abbreviations.\n"
        "- Facts must be correct, contradiction-free, updated only when tool results or logic require it.\n"
        "- CRITICAL: When recording operations(e.g remove/add/...) with cascading effects, explicitly list all affected items including descendants/dependents.\n"
        "\n"
        "**CONSTRAINTS**\n"
        "- Bullet list of key task requirements, guardrails, and rules that must remain in force throughout execution.\n"
        "- Keep these bullets even when no new data arrives, updating them only when tools or logic clearly require it.\n"
        "\n"
        "**VERIFIED CONSTRAINTS** (if present in prior memory)\n"
        "- If the prior memory contains a VERIFIED CONSTRAINTS section, carry it forward EXACTLY as-is. Never modify or remove items from this section.\n"
        "- This section contains specific constraints/issues (with concrete names/paths/IDs) discovered during verification that must persist and should not be violated.\n"
        "\n"
        "Rules: keep the memory concise yet precise, cite only tool-backed information or direct logical consequences. If nothing new was learned leave the memory unchanged."
    )
    VERIFICATION_SYSTEM_PROMPT = (
        "You are a verification agent responsible for validating and correcting memory reports. "
        "Check the generated memory for:\n"
        "\n"
        "1. **FACTS ACCURACY**: Verify all concrete values (counts, paths, IDs, metrics) are accurate "
        "and directly supported by tool results. Flag any claims without tool evidence.\n"
        "\n"
        "2. **LOGICAL CONSISTENCY**: Ensure no contradictions exist, language is clear, and sections align - "
        "completed steps in PROGRESS must match PERSISTENT FACTS, "
        "plans must not conflict with constraints and user instructions, facts must not contradict each other. \n"
        "When memory records operations affecting items with dependencies or hierarchies, "
        "cross-check tool outputs to verify all affected related items are explicitly listed.\n"
        "\n"
        "3. **REASONING CORRECTNESS**: Verify logical inferences are valid. "
        "When memory makes interpretive claims (e.g., 'X contains Y', dependencies, hierarchies), "
        "trace back to tool output to verify these are directly shown, not assumed. "
        "Flag interpretive claims that should be verified with additional tool calls.\n"
        "\n"
        "4. **VERIFIED CONSTRAINTS**: When discovering errors, add them to VERIFIED CONSTRAINTS section:\n"
        "   - Only add constraints explicitly violated or contradicted by tool results\n"
        "   - Make constraints SPECIFIC with actual entity names, paths, IDs, values - no abstractions\n"
        "   - Include: (1) specific instance that failed with actual names, (2) underlying rule explaining why\n"
        "   - Create this section if it doesn't exist\n"
        "\n"
        "Call verify_memory with your analysis. If issues found, provide corrected version. "
        "If no issues, return unchanged and acknowledge validity."
    )
    ACTION_SYSTEM_PROMPT = SYSTEM_PROMPT
    DEFAULT_TIMEOUT = BaseMCPAgent.DEFAULT_TIMEOUT

    def __init__(
        self,
        litellm_input_model_name: str,
        api_key: str,
        base_url: str,
        mcp_service: str,
        timeout: int = DEFAULT_TIMEOUT,
        service_config: Optional[Dict[str, Any]] = None,
        service_config_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = "default",
    ):
        super().__init__(
            litellm_input_model_name=litellm_input_model_name,
            api_key=api_key,
            base_url=base_url,
            mcp_service=mcp_service,
            timeout=timeout,
            service_config=service_config,
            service_config_provider=service_config_provider,
            reasoning_effort=reasoning_effort,
        )
        
        # Initialize OpenAI client for non-Claude models
        logger.info("self.is_claude: %s", self.is_claude)        
        model_name = litellm_input_model_name.split("/", 1)[1] if "/" in litellm_input_model_name else litellm_input_model_name
        logger.info(f"Initializing OpenAI client for model: {model_name}")
        # self._openai_client = SimpleOpenAIClient(
        #     model=model_name,
        #     api_key=api_key,
        #     base_url=base_url,
        #     reasoning_effort=reasoning_effort or "default",
        # )
        self._openai_client = None
        
        # Memory management configuration
        self.memory_update_threshold = self.service_config.get("memory_update_threshold", 8)
        logger.info(f"Memory update threshold: {self.memory_update_threshold} tool calls")
        
        logger.debug(
            "Initialized MCPMarkAgent for '%s' with model '%s' (Claude: %s, OpenAI Client: %s)",
            mcp_service,
            litellm_input_model_name,
            self.is_claude,
            self._openai_client is not None,
        )

    # ==================== Public Interface Methods ====================

    async def execute(
        self, 
        instruction: str, 
        tool_call_log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute instruction with the agent.
        
        Args:
            instruction: The instruction/prompt to execute
            tool_call_log_file: Optional path to log tool calls
            
        Returns:
            Dictionary containing execution results
        """
        start_time = time.time()
        
        try:
            # Reset partial progress for this run
            self._reset_progress()
            # Refresh service configuration
            self._refresh_service_config()
            
            # Execute with timeout control
            async def _execute_with_strategy():
                if self.use_claude_thinking:
                    # Claude with thinking -> native Anthropic API with tools
                    return await self._execute_claude_native_with_tools(
                        instruction, tool_call_log_file
                    )
                else:
                    # All other cases -> LiteLLM with tools
                    return await self._execute_litellm_with_tools(
                        instruction, tool_call_log_file
                    )
            
            # Apply timeout to the entire execution
            result = await asyncio.wait_for(
                _execute_with_strategy(),
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Update usage statistics
            self.usage_tracker.update(
                success=result["success"],
                token_usage=result.get("token_usage", {}),
                turn_count=result.get("turn_count", 0),
                execution_time=execution_time
            )
            
            result["execution_time"] = execution_time
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            if isinstance(e, asyncio.TimeoutError):
                error_msg = f"Execution timed out after {self.timeout} seconds"
                logger.error(error_msg)
            else:
                error_msg = f"Agent execution failed: {e}"
                logger.error(error_msg, exc_info=True)
            
            self.usage_tracker.update(
                success=False,
                token_usage=self._partial_token_usage or {},
                turn_count=self._partial_turn_count or 0,
                execution_time=execution_time
            )

            if self._partial_messages:
                if not self.is_claude:
                    final_msg = self._convert_to_sdk_format(self._partial_messages)
                else:
                    final_msg = self._partial_messages
            else:
                final_msg = []
                
            return {
                "success": False,
                "output": final_msg,
                "token_usage": self._partial_token_usage or {},
                "turn_count": self._partial_turn_count or 0,
                "execution_time": execution_time,
                "error": error_msg,
                "litellm_run_model_name": self.litellm_run_model_name,
            }
            

    def execute_sync(
        self,
        instruction: str,
        tool_call_log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for execute method.
        """
        return asyncio.run(self.execute(instruction, tool_call_log_file))
    

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self.usage_tracker.get_stats()
    

    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_tracker.reset()
    


    # ==================== Claude Native API Execution Path ====================

    async def _execute_claude_native_with_tools(
        self,
        instruction: str,
        tool_call_log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Claude with thinking using native Anthropic API.
        Creates MCP server, gets tools, and executes with thinking.
        """
        logger.debug("Using Claude native API with thinking")
        
        thinking_budget = self._get_claude_thinking_budget()
        
        # Create and start MCP server
        mcp_server = await self._create_mcp_server()
        
        async with mcp_server:
            # Get available tools
            tools = await mcp_server.list_tools()
            
            # Convert MCP tools to Anthropic format
            anthropic_tools = self._convert_to_anthropic_format(tools)
            
            # Execute with function calling loop
            return await self._execute_anthropic_native_tool_loop(
                instruction, anthropic_tools, mcp_server, 
                thinking_budget, tool_call_log_file
            )
    

    async def _call_claude_native_api(
        self,
        messages: List[Dict],
        thinking_budget: int,
        tools: Optional[List[Dict]] = None,
        mcp_servers: Optional[List[Dict]] = None,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call Claude's native API directly using httpx.
        
        Args:
            messages: Conversation messages
            thinking_budget: Token budget for thinking
            tools: Tool definitions for function calling
            mcp_servers: MCP server configurations
            system: System prompt
            
        Returns:
            API response as dictionary
        """
        # Get API base and headers
        import os
        api_base = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com") 
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "anthropic-beta": "context-1m-2025-08-07" # by default
        } 
        
        # Build payload
        max_tokens = max(thinking_budget + 4096, 4096)
        payload = {
            "model": self.litellm_input_model_name.replace("anthropic/", ""),
            "max_tokens": max_tokens,
            "messages": messages,
        }
        
        # Add thinking configuration
        if thinking_budget:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = {"type": "auto"}
        
        # Add MCP servers if provided
        if mcp_servers:
            headers["anthropic-beta"] = "mcp-client-2025-04-04"
            payload["mcp_servers"] = mcp_servers
        
        # Add system prompt if provided
        if system:
            payload["system"] = system
        
        # Make the API call
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{api_base}/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json(), None
            except httpx.HTTPStatusError as e:
                return None, e.response.text
            except Exception as e:
                return None, e
    

    async def _execute_anthropic_native_tool_loop(
        self,
        instruction: str,
        tools: List[Dict],
        mcp_server: Any,
        thinking_budget: int,
        tool_call_log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Claude thinking loop with function calling.
        Handles thinking blocks, tool calls, and message formatting.
        """
        messages = [{"role": "user", "content": instruction}]
        total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
        turn_count = 0
        max_turns = self.MAX_TURNS
        hit_turn_limit = False
        ended_normally = False
        
        system_text = self.SYSTEM_PROMPT
        # Record initial state
        self._update_progress(messages, total_tokens, turn_count)
        
        for _ in range(max_turns):
            turn_count += 1
            
            # Call Claude native API
            response, error_msg = await self._call_claude_native_api(
                messages=messages,
                thinking_budget=thinking_budget,
                tools=tools,
                system=system_text
            )
            if turn_count == 1:
                self.litellm_run_model_name = response['model'].split("/")[-1]
            
            if error_msg:
                break
            
            # Update token usage
            if "usage" in response:
                usage = response["usage"]
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                # Calculate output tokens as total - input for consistency
                total_tokens_count = output_tokens + input_tokens
                
                total_tokens["input_tokens"] += input_tokens
                total_tokens["output_tokens"] += output_tokens
                total_tokens["total_tokens"] += total_tokens_count
                
                ## TODO: add reasoning tokens for claude
            
            # Extract blocks from response
            blocks = response.get("content", [])
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            thinking_blocks = [b for b in blocks if b.get("type") == "thinking"]
            text_blocks = [b for b in blocks if b.get("type") == "text"]
            
            # Log text output
            for tb in text_blocks:
                if tb.get("text") and tool_call_log_file:
                    with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{tb['text']}\n")
                if tb.get("text"):
                    for line in tb["text"].splitlines():
                        logger.info(f"| {line}")
            
            # Build assistant message with all blocks
            assistant_content = []
            
            # Add thinking blocks
            for tb in thinking_blocks:
                assistant_content.append({
                    "type": "thinking",
                    "thinking": tb.get("thinking", ""),
                    "signature": tb.get("signature", ""),
                })
            
            # Add text blocks
            for tb in text_blocks:
                if tb.get("text"):
                    assistant_content.append({"type": "text", "text": tb["text"]})
            
            # Add tool_use blocks
            for tu in tool_uses:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tu.get("id"),
                    "name": tu.get("name"),
                    "input": tu.get("input", {}),
                })
            
            messages.append({"role": "assistant", "content": assistant_content})
            
            # Update partial progress after assistant response
            self._update_progress(messages, total_tokens, turn_count)

            # If no tool calls, we're done
            if not tool_uses:
                ended_normally = True
                break
            
            # Execute tools and add results
            tool_results = []
            for tu in tool_uses:
                name = tu.get("name")
                inputs = tu.get("input", {})
                
                # Log tool call
                args_str = json.dumps(inputs, separators=(",", ": "))
                display_args = args_str[:140] + "..." if len(args_str) > 140 else args_str
                logger.info(f"| \033[1m{name}\033[0m \033[2;37m{display_args}\033[0m")
                
                if tool_call_log_file:
                    with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"| {name} {args_str}\n")
                
                # Execute tool
                try:
                    result = await asyncio.wait_for(
                        mcp_server.call_tool(name, inputs),
                        timeout=60
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": [{"type": "text", "text": json.dumps(result)}],
                    })
                except Exception as e:
                    logger.error(f"Tool call failed: {e}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                    })
            
            messages.append({"role": "user", "content": tool_results})
            # Update partial progress after tool results
            self._update_progress(messages, total_tokens, turn_count)
        
        # Detect if we exited due to hitting the turn limit
        if not ended_normally:
            if turn_count >= max_turns:
                hit_turn_limit = True
                logger.warning(f"| Max turns ({max_turns}) exceeded; returning failure with partial output.")
                if tool_call_log_file:
                    try:
                        with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"| Max turns ({max_turns}) exceeded\n")
                    except Exception:
                        pass
            elif error_msg:
                logger.warning(f"| {error_msg}\n")
                if tool_call_log_file:
                    try:
                        with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"| {error_msg}\n")
                    except Exception:
                        pass
        
        # Display final token usage
        if total_tokens["total_tokens"] > 0:
            log_msg = (
                f"|\n| Token usage: Total: {total_tokens['total_tokens']:,} | "
                f"Input: {total_tokens['input_tokens']:,} | "
                f"Output: {total_tokens['output_tokens']:,}"
            )
            if total_tokens.get("reasoning_tokens", 0) > 0:
                log_msg += f" | Reasoning: {total_tokens['reasoning_tokens']:,}"
            logger.info(log_msg)
            logger.info(f"| Turns: {turn_count}")
        
        # Convert messages to SDK format
        # sdk_format_messages = self._convert_to_sdk_format(messages)
        
        if hit_turn_limit:
            return {
                "success": False,
                "output": messages,
                "token_usage": total_tokens,
                "turn_count": turn_count,
                "error": f"Max turns ({max_turns}) exceeded",
                "litellm_run_model_name": self.litellm_run_model_name,
            }
        
        if error_msg:
            return {
                "success": False,
                "output": messages,
                "token_usage": total_tokens,
                "turn_count": turn_count,
                "error": error_msg,
                "litellm_run_model_name": self.litellm_run_model_name,
            }
        
        return {
            "success": True,
            "output": messages,
            "token_usage": total_tokens,
            "turn_count": turn_count,
            "error": None,
            "litellm_run_model_name": self.litellm_run_model_name,
        }


    # ==================== LiteLLM Execution Path ====================

    async def _execute_litellm_with_tools(
        self,
        instruction: str,
        tool_call_log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute with manual MCP server management.
        Used for all non-Anthropic models and Anthropic models with STDIO services.
        """
        logger.debug("Using manual MCP execution with function calling loop")
        
        # Create and start MCP server
        mcp_server = await self._create_mcp_server()
        
        try:
            async with mcp_server:
                # Get available tools
                tools = await mcp_server.list_tools()
                
                # Convert MCP tools to OpenAI function format
                functions = self._convert_to_openai_format(tools)
                
                # Execute with function calling loop
                return await self._execute_two_phase_tool_loop(
                    instruction, functions, mcp_server, tool_call_log_file
                )
                # return await self._execute_litellm_tool_loop(
                #     instruction, functions, mcp_server, tool_call_log_file
                # )
                
        except Exception as e:
            logger.error(f"Manual MCP execution failed: {e}")
            raise
        
    
    def _create_update_memory_tool(self) -> Dict:
        """Create the shared memory update tool"""
        return {
            "name": "update_memory",
            "description": (
                "Update the shared task memory using the latest tool results. "
                "The string you return must follow the OVERVIEW / TASK PLAN / PROGRESS & GAPS / PERSISTENT FACTS / CONSTRAINTS / VERIFIED CONSTRAINTS(optional) layout "
                "and only include information directly supported by tool outputs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "report": {
                        "type": "string",
                        "description": (
                            "A concise memory update covering the required sections."
                        )
                    }
                },
                "required": ["report"]
            }
        }
    
    def _create_verify_memory_tool(self) -> Dict:
        """Create the memory verification tool"""
        return {
            "name": "verify_memory",
            "description": (
                "Verify and potentially correct the generated memory report. "
                "Check for correctness of facts, logical consistency, completeness, reasoning validity, and format. "
                "Return the verified(corrected if needed) memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "verified_report": {
                        "type": "string",
                        "description": (
                            "The verified and potentially corrected memory report following the same structure."
                        )
                    },
                    "issues_found": {
                        "type": "string",
                        "description": (
                            "Brief description of any issues found and corrections made, or 'None' if memory was correct."
                        )
                    }
                },
                "required": ["verified_report", "issues_found"]
            }
        }
    
    def _build_messages_with_context(
        self,
        instruction: str,
        latest_memory: str = "",
        last_tool_contexts: Optional[List[Dict[str, Any]]] = None,
        mode: str = "action",
        tool_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Build message list with proper roles: user, assistant (memory), tool (result)"""
        memory_text = latest_memory or "none"
        
        if mode == "report":
            messages = [
                {"role": "system", "content": self.MEMORY_SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "user", "content": "update the shared memory to reflect the latest tool results and adjust the plan accordingly."},
            ]
        elif mode == "verify":
            messages = [
                {"role": "system", "content": self.VERIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Original Task: {instruction}"},
            ]
        else:  # action
            messages = [
                {"role": "system", "content": self.ACTION_SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ]
        
        def _append_tool_call(ctx: Dict[str, Any]) -> None:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": ctx["id"],
                            "type": "function",
                            "function": {
                                "name": ctx["name"],
                                "arguments": ctx.get("arguments", "{}")
                            }
                        }
                    ]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": ctx["id"],
                    "content": ctx.get("formatted_result") or str(ctx["result"])
                })
        if mode == "action" and tool_history:
            for ctx in tool_history:
                _append_tool_call(ctx)
        elif mode == "report" and last_tool_contexts:
            for ctx in last_tool_contexts:
                _append_tool_call(ctx)
        elif mode == "verify" and last_tool_contexts:
            for ctx in last_tool_contexts:
                _append_tool_call(ctx)

        if mode != "verify":
            messages.append({
                "role": "user",
                "content": f"Reference memory (if any): {memory_text}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Generated Memory to Verify:\n{memory_text}\n"
            })

        return messages

    @staticmethod
    def _safe_json_loads(value: Any) -> Optional[Any]:
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None

    def _format_tool_result_for_model(self, result: Any, tool_name: str = "") -> str:
        """
        Convert raw MCP tool output into a JSON-friendly string for the model.
        Attempts to strip extra wrappers (meta/content) while preserving human-readable text.
        """
        payload = result

        parsed = self._safe_json_loads(payload) if isinstance(payload, str) else None
        if parsed is not None:
            payload = parsed
        
        if tool_name == "directory_tree":
            # Extract the actual directory tree data from the nested structure
            tree_data = None
            
            # Case 1: payload is already a list of entries
            if isinstance(payload, list) and self._looks_like_directory_tree(payload):
                tree_data = payload
            # Case 2: payload has content wrapper with text field
            elif isinstance(payload, dict):
                content_blocks = payload.get("content")
                if isinstance(content_blocks, list) and content_blocks:
                    for block in content_blocks:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            text = block["text"].strip()
                            # Try to parse the text as JSON
                            parsed_tree = self._safe_json_loads(text)
                            if parsed_tree and isinstance(parsed_tree, list) and self._looks_like_directory_tree(parsed_tree):
                                tree_data = parsed_tree
                                break
            
            if tree_data:
                formatted = self._format_directory_tree(tree_data)
                return formatted
            else:
                logger.warning("Could not extract directory tree data from payload, returning raw result")
                # Fall through to general handling as fallback

        if isinstance(payload, dict):
            content_blocks = payload.get("content")
            text_segments: List[str] = []
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        text = block["text"].strip()
                        if text:
                            text_segments.append(text)
            if text_segments:
                return "\n".join(text_segments)


        if isinstance(payload, (dict, list)):
            try:
                return json.dumps(payload, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(payload)

        return str(payload)

    @staticmethod
    def _looks_like_directory_tree(payload: Any) -> bool:
        if not isinstance(payload, list) or not payload:
            return False
        sample = payload[0]
        return isinstance(sample, dict) and "name" in sample and "type" in sample

    def _format_directory_tree(self, entries: List[Dict[str, Any]]) -> str:
        def recurse(nodes: List[Dict[str, Any]], prefix: str, is_root: bool) -> List[str]:
            lines: List[str] = []
            count = len(nodes)
            for idx, node in enumerate(nodes):
                name = node.get("name", "unknown")
                node_type = node.get("type", "")
                children = node.get("children") if isinstance(node, dict) else None
                is_dir = node_type == "directory"
                suffix = "/" if is_dir else ""
                is_last = idx == count - 1

                if is_root:
                    line = f"{name}{suffix}"
                    child_prefix = "    "
                else:
                    connector = "└── " if is_last else "├── "
                    line = f"{prefix}{connector}{name}{suffix}"
                    child_prefix = prefix + ("    " if is_last else "│   ")

                lines.append(line)
                if is_dir and isinstance(children, list) and children:
                    lines.extend(recurse(children, child_prefix, False))
            return lines

        return "\n".join(recurse(entries, "", True))

    def _contains_critical_tools(self, tool_contexts: List[Dict[str, Any]]) -> bool:
        """Check if any critical tools were called in the given tool contexts."""
        for ctx in tool_contexts:
            tool_name = ctx.get("name", "")
            if tool_name in self.CRITICAL_TOOLS_FOR_MEMORY:
                return True
        return False


    def _get_report_model_config(self) -> Dict[str, Optional[str]]:
        """Return the model/api/base_url tuple for report summarization."""
        model_name = self.service_config.get("report_model_name") or self.REPORT_MODEL_DEFAULT
        api_key = self.service_config.get("report_model_api_key") or self.REPORT_MODEL_API_KEY_DEFAULT
        base_url = self.service_config.get("report_model_base_url") or self.REPORT_MODEL_BASE_URL_DEFAULT
        return {
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url
        }

    async def _execute_two_phase_tool_loop(
        self,
        instruction: str,
        functions: List[Dict],
        mcp_server: Any,
        tool_call_log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute function calling loop with LiteLLM using report-based memory compression."""
        
        # Prepare tool definitions
        memory_tool_def = self._create_update_memory_tool()
        action_tools = [{"type": "function", "function": func} for func in functions] if functions else []
        memory_tool = {"type": "function", "function": memory_tool_def}
        available_tools = action_tools + [memory_tool]

        
        # State tracking for report-based prompting
        latest_memory = ""
        tool_contexts_since_last_memory: List[Dict[str, Any]] = []  # Accumulate until memory update
        tool_call_history: List[Dict[str, Any]] = []  # Full history for action LLM
        
        # Message accumulation for output (backward compatibility)
        all_messages = []
        
        total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
        turn_count = 0
        max_turns = self.MAX_TURNS  # Limit turns to prevent infinite loops
        consecutive_failures = 0
        max_consecutive_failures = 3
        hit_turn_limit = False
        ended_normally = False

        def _record_usage(response_obj):
            """Accumulate token usage from a LiteLLM response."""
            if hasattr(response_obj, 'usage') and response_obj.usage:
                input_tokens = response_obj.usage.prompt_tokens or 0
                total_tokens_count = response_obj.usage.total_tokens or 0
                output_tokens = (
                    total_tokens_count - input_tokens
                    if total_tokens_count > 0
                    else (response_obj.usage.completion_tokens or 0)
                )
                total_tokens["input_tokens"] += input_tokens
                total_tokens["output_tokens"] += output_tokens
                total_tokens["total_tokens"] += total_tokens_count
                if hasattr(response_obj.usage, 'completion_tokens_details'):
                    details = response_obj.usage.completion_tokens_details
                    if hasattr(details, 'reasoning_tokens'):
                        total_tokens["reasoning_tokens"] += details.reasoning_tokens or 0
        
        # Log available tools
        if tool_call_log_file and available_tools:
            max_name_length = max(
                len(tool.get("function", {}).get("name", ""))
                for tool in available_tools
            )
            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                f.write("===== Available Tools =====\n")
                for tool in available_tools:
                    function_info = tool.get("function", {})
                    tool_name = function_info.get("name", "N/A")
                    description = function_info.get("description", "N/A")
                    f.write(f"- ToolName: {tool_name:<{max_name_length}} Description: {description}\n")
                f.write("\n===== Execution Logs =====\n")
        
        try:
            while turn_count < max_turns:
                # Build fresh messages with proper roles (reset each turn for compression)
                messages = self._build_messages_with_context(
                    instruction=instruction,
                    latest_memory=latest_memory,
                    mode="action",
                    tool_history=tool_call_history,
                )
                
                # Store for output (only add on first turn to avoid duplication)
                if turn_count == 0:
                    all_messages.append({"role": "system", "content": self.SYSTEM_PROMPT})
                    all_messages.append({"role": "user", "content": instruction})
                
                # Build completion kwargs
                completion_kwargs = {
                    "model": self.litellm_input_model_name,
                    "messages": messages,
                    "api_key": self.api_key,
                }
                
                # Action phase uses only action tools
                if action_tools:
                    completion_kwargs["tools"] = action_tools
                    completion_kwargs["tool_choice"] = "auto"
                
                # Add reasoning_effort and base_url if specified
                if self.reasoning_effort != "default":
                    completion_kwargs["reasoning_effort"] = self.reasoning_effort
                if self.base_url:
                    completion_kwargs["base_url"] = self.base_url
                
                # DEBUG: Log what we're sending to LiteLLM
                # logger.info(f"\n===== LITELLM CALL (Turn {turn_count + 1}) =====")
                # logger.info(f"Tools: {len(completion_kwargs.get('tools', []))} tools available")
                # logger.info(f"Tool choice: {completion_kwargs.get('tool_choice', 'not set')}")
                # logger.info(f"Messages: {messages}")
                # logger.info("===== END CALL INFO =====\n")
                
                try:
                    # Call OpenAI client or LiteLLM depending on model type
                    if self._openai_client:
                        response = await asyncio.wait_for(
                            self._openai_client.acompletion(
                                messages=messages,
                                tools=completion_kwargs.get("tools"),
                                tool_choice=completion_kwargs.get("tool_choice"),
                            ),
                            timeout = self.timeout / 2
                        )
                    else:
                        response = await asyncio.wait_for(
                            litellm.acompletion(**completion_kwargs),
                            timeout = self.timeout / 2
                        )
                    consecutive_failures = 0  # Reset failure counter on success
                except asyncio.TimeoutError:
                    logger.warning(f"| ✗ LLM call timed out on turn {turn_count + 1}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise Exception(f"Too many consecutive failures ({consecutive_failures})")
                    await asyncio.sleep(8 ** consecutive_failures)  # Exponential backoff
                    continue
                except Exception as e:
                    logger.error(f"| ✗ LLM call failed on turn {turn_count + 1}: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise
                    if "ContextWindowExceededError" in str(e):
                        raise
                    elif "RateLimitError" in str(e):
                        await asyncio.sleep(12 ** consecutive_failures)
                    else:
                        await asyncio.sleep(2 ** consecutive_failures)
                    continue
                
                # Extract actual model name from response (first turn only)
                if turn_count == 0 and hasattr(response, 'model') and response.model:
                    self.litellm_run_model_name = response.model.split("/")[-1]
                
                # Update token usage including reasoning tokens
                _record_usage(response)
                
                # Get response message
                choices = response.choices
                if len(choices):
                    message = choices[0].message
                    message_dict = message.model_dump() if hasattr(message, 'model_dump') else dict(message)
                    
                    # DEBUG: Log what LLM returned (summary)
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        tool_names = [tc.function.name for tc in message.tool_calls]
                        logger.info(f"| 🔧 LLM returned {len(message.tool_calls)} tool call(s): {tool_names}")
                        if tool_call_log_file:
                            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"| DEBUG: Tool calls returned: {tool_names}\n")
                    else:
                        finish_reason = choices[0].finish_reason if choices else 'unknown'
                        has_content = bool(hasattr(message, 'content') and message.content)
                        logger.info(f"| 🔧 LLM returned NO tool calls (finish_reason: {finish_reason}, has_content: {has_content})")
                        if tool_call_log_file:
                            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"| DEBUG: No tool calls (finish_reason: {finish_reason}, has_content: {has_content})\n")
                    
                # Log assistant's text content if present
                if hasattr(message, 'content') and message.content:
                    # Display the content with line prefix
                    for line in message.content.splitlines():
                        logger.info(f"| {line}")
                    
                    # Also log to file if specified
                    if tool_call_log_file:
                        with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"{message.content}\n")
                
                # Check for tool calls (newer format)
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Add assistant message to output
                    all_messages.append(message_dict)
                    # Process tool calls 
                    action_tool_executed = False
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        
                        # Execute real MCP tool
                        try:
                            result = await asyncio.wait_for(
                                mcp_server.call_tool(func_name, func_args),
                                timeout=60
                            )
                            formatted_result = self._format_tool_result_for_model(result, func_name)
                            # Store for memory update context (accumulate across turns)
                            tool_contexts_since_last_memory.append({
                                "name": func_name,
                                "result": result,
                                "formatted_result": formatted_result,
                                "id": tool_call.id,
                                "arguments": json.dumps(func_args, separators=(",", ": "))
                            })
                            
                            # Add to messages for output
                            all_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": formatted_result
                            })
                        except asyncio.TimeoutError:
                            error_msg = f"Tool call '{func_name}' timed out after 60 seconds"
                            logger.error(error_msg)
                            formatted_result = error_msg
                            tool_contexts_since_last_memory.append({
                                "name": func_name,
                                "result": f"Error: {error_msg}",
                                "formatted_result": formatted_result,
                                "id": tool_call.id,
                                "arguments": json.dumps(func_args, separators=(",", ": "))
                            })
                            
                            # Add error to messages
                            all_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {error_msg}"
                            })
                        except Exception as e:
                            logger.error(f"Tool call failed: {e}")
                            formatted_result = f"Error: {str(e)}"
                            tool_contexts_since_last_memory.append({
                                "name": func_name,
                                "result": f"Error: {str(e)}",
                                "formatted_result": formatted_result,
                                "id": tool_call.id,
                                "arguments": json.dumps(func_args, separators=(",", ": "))
                            })
                            
                            # Add error to messages
                            all_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": formatted_result
                            })
                        action_tool_executed = True
                        tool_call_history.append(dict(tool_contexts_since_last_memory[-1]))
                        
                        # Format arguments for display (truncate if too long)
                        args_str = json.dumps(func_args, separators=(",", ": "))
                        display_arguments = args_str[:140] + "..." if len(args_str) > 140 else args_str
                        
                        # Log with ANSI color codes (bold tool name, dim gray arguments)
                        logger.info(f"| \033[1m{func_name}\033[0m \033[2;37m{display_arguments}\033[0m")
                        
                        if tool_call_log_file:
                            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"| {func_name} {args_str}\n")
                                f.write(f"  Result: {formatted_result}\n")
                
                    # Update progress after tool results
                    messages_for_progress = [message_dict]
                    turn_count += 1
                    self._update_progress(messages_for_progress, total_tokens, turn_count)
                    
                    # Check if critical tools were called
                    has_critical_tools = self._contains_critical_tools(tool_contexts_since_last_memory)
                    
                    # Decide if we should update memory based on accumulated tool calls
                    should_update_memory = (
                        action_tool_executed and (
                            turn_count <= 3  # First 3 turns always update
                            or len(tool_contexts_since_last_memory) >= self.memory_update_threshold
                            or turn_count >= max_turns - 5  # Near end always update
                            or has_critical_tools  # Critical tools always trigger memory update
                        )
                    )
                    
                    if should_update_memory:
                        # Log reason for memory update
                        if has_critical_tools:
                            critical_tool_names = [ctx.get("name") for ctx in tool_contexts_since_last_memory 
                                                  if ctx.get("name") in self.CRITICAL_TOOLS_FOR_MEMORY]
                            logger.info(f"🔑 Critical tool(s) detected: {critical_tool_names} - triggering memory update")
                        
                        # Build messages with accumulated tool contexts
                        memory_messages = self._build_messages_with_context(
                            instruction=instruction,
                            latest_memory=latest_memory,
                            last_tool_contexts=tool_contexts_since_last_memory,
                            mode="report"
                        )

                        memory_kwargs = {
                            "model": self.litellm_input_model_name,
                            "messages": memory_messages,
                            "api_key": self.api_key,
                            "tools": [memory_tool],
                            "tool_choice": {
                                "type": "function",
                                "function": {"name": "update_memory"}
                            }
                        }
                        
                        if self.reasoning_effort != "default":
                            memory_kwargs["reasoning_effort"] = self.reasoning_effort
                        if self.base_url:
                            memory_kwargs["base_url"] = self.base_url

                        # DEBUG: Log what we're sending to LiteLLM
                        # logger.info(f"\n===== LITELLM SUBMIT REPORT CALL (Turn {turn_count + 1}) =====")
                        # logger.info(f"Tools: {len(report_kwargs.get('tools', []))} tools available")
                        # logger.info(f"Tool choice: {report_kwargs.get('tool_choice', 'not set')}")
                        # logger.info(f"Messages: {report_messages}")
                        # logger.info("===== END CALL INFO =====\n")

                        try:
                            if self._openai_client:
                                logger.info(f"update_memory call with OpenAI client")

                                memory_response = await asyncio.wait_for(
                                    self._openai_client.acompletion(
                                        messages=memory_messages,
                                        tools=memory_kwargs.get("tools"),
                                        tool_choice=memory_kwargs.get("tool_choice"),
                                    ),
                                    timeout=self.timeout / 2
                                )
                            else:
                                logger.info(f"📝 update_memory call ({len(tool_contexts_since_last_memory)} tool calls accumulated)")
                                memory_response = await asyncio.wait_for(
                                    litellm.acompletion(**memory_kwargs),
                                    timeout=self.timeout / 2
                                )
                            consecutive_failures = 0
                        except asyncio.TimeoutError:
                            logger.warning("| ✗ Memory call timed out; continuing without updated memory")
                            continue
                        except Exception as e:
                            logger.error(f"| ✗ Memory call failed: {e}")
                            continue
                        
                        _record_usage(memory_response)
                        
                        memory_choices = memory_response.choices
                        if not len(memory_choices):
                            logger.error("| Memory call returned no choices")
                            continue
                        
                        memory_message = memory_choices[0].message
                        memory_message_dict = memory_message.model_dump() if hasattr(memory_message, 'model_dump') else dict(memory_message)
                        
                        if not hasattr(memory_message, 'tool_calls') or not memory_message.tool_calls:
                            logger.error("| Memory call did not return update_memory tool call")
                            continue
                        
                        all_messages.append(memory_message_dict)
                        
                        # Extract memory from the single tool call
                        memory_tool_call = memory_message.tool_calls[0]
                        func_args = json.loads(memory_tool_call.function.arguments)
                        
                        latest_memory = func_args.get("report", "")
                        logger.info(f"🧠 Memory (unverified): {latest_memory[:200]}{'...' if len(latest_memory) > 200 else ''}")
                        
                        if tool_call_log_file:
                            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"| update_memory ({len(tool_contexts_since_last_memory)} calls)\n")
                                f.write(f"  Memory: {latest_memory}\n")
                        
                        all_messages.append({
                            "role": "tool",
                            "tool_call_id": memory_tool_call.id,
                            "content": f"Memory recorded: {latest_memory}"
                        })
                        
                        # === VERIFICATION PHASE ===
                        # Now verify the generated memory
                        verification_tool_def = self._create_verify_memory_tool()
                        verification_tool = {"type": "function", "function": verification_tool_def}
                        
                        verification_messages = self._build_messages_with_context(
                            instruction=instruction,
                            latest_memory=latest_memory,
                            last_tool_contexts=tool_contexts_since_last_memory,
                            mode="verify"
                        )
                        
                        verification_kwargs = {
                            "model": self.litellm_input_model_name,
                            "messages": verification_messages,
                            "api_key": self.api_key,
                            "tools": [verification_tool],
                            "tool_choice": {
                                "type": "function",
                                "function": {"name": "verify_memory"}
                            }
                        }
                        
                        if self.reasoning_effort != "default":
                            verification_kwargs["reasoning_effort"] = self.reasoning_effort
                        if self.base_url:
                            verification_kwargs["base_url"] = self.base_url
                        
                        logger.info(f"verify_memory call")
                        
                        try:
                            if self._openai_client:
                                verification_response = await asyncio.wait_for(
                                    self._openai_client.acompletion(
                                        messages=verification_messages,
                                        tools=verification_kwargs.get("tools"),
                                        tool_choice=verification_kwargs.get("tool_choice"),
                                    ),
                                    timeout=self.timeout / 2
                                )
                            else:
                                verification_response = await asyncio.wait_for(
                                    litellm.acompletion(**verification_kwargs),
                                    timeout=self.timeout / 2
                                )
                            consecutive_failures = 0
                        except asyncio.TimeoutError:
                            logger.warning("| ✗ Verification call timed out; using unverified memory")
                            continue
                        except Exception as e:
                            logger.error(f"| ✗ Verification call failed: {e}; using unverified memory")
                            continue
                        
                        _record_usage(verification_response)
                        
                        verification_choices = verification_response.choices
                        if not len(verification_choices):
                            logger.error("| Verification call returned no choices; using unverified memory")
                            continue
                        
                        verification_message = verification_choices[0].message
                        verification_message_dict = verification_message.model_dump() if hasattr(verification_message, 'model_dump') else dict(verification_message)
                        
                        if not hasattr(verification_message, 'tool_calls') or not verification_message.tool_calls:
                            logger.error("| Verification call did not return verify_memory tool call; using unverified memory")
                            continue
                        
                        all_messages.append(verification_message_dict)
                        
                        # Extract verified memory from the single tool call
                        verify_tool_call = verification_message.tool_calls[0]
                        verify_func_args = json.loads(verify_tool_call.function.arguments)
                        
                        verified_memory = verify_func_args.get("verified_report", latest_memory)
                        issues_found = verify_func_args.get("issues_found", "Unknown")
                        
                        # Replace latest_memory with verified version
                        latest_memory = verified_memory
                        
                        if issues_found.lower() == "none":
                            logger.info(f"✅ Memory verified (no issues): {verified_memory[:200]}{'...' if len(verified_memory) > 200 else ''}")
                        else:
                            logger.info(f"🔧 Memory verified with corrections: {issues_found[:100]}{'...' if len(issues_found) > 100 else ''}")
                            logger.info(f"✅ Corrected Memory: {verified_memory[:200]}{'...' if len(verified_memory) > 200 else ''}")
                        
                        if tool_call_log_file:
                            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"| verify_memory\n")
                                f.write(f"  Issues Found: {issues_found}\n")
                                f.write(f"  Verified Memory: {verified_memory}\n")
                        
                        all_messages.append({
                            "role": "tool",
                            "tool_call_id": verify_tool_call.id,
                            "content": f"Memory verified. Issues: {issues_found}"
                        })
                        
                        # Update progress with memory and verification responses
                        self._update_progress([memory_message_dict, verification_message_dict], total_tokens, turn_count)
                        
                        # Reset accumulated contexts after successful memory update
                        # (tool_call_history is kept for action LLM's full context)
                        tool_contexts_since_last_memory = []
                    else:
                        # Skip memory update this turn
                        if action_tool_executed:
                            logger.info(f"⏭️  Memory update skipped ({len(tool_contexts_since_last_memory)}/{self.memory_update_threshold} calls accumulated, no critical tools)")
                    
                    continue
                else:
                    # Log end reason
                    if not choices:
                        logger.info("|\n|\n| Task ended with no messages generated by the model.")
                    elif choices[0].finish_reason == "stop":
                        logger.info("|\n|\n| Task ended with the finish reason from messages being 'stop'.")
                    
                    # No tool/function call, add message and we're done
                    all_messages.append(message_dict)
                    turn_count += 1
                    # Update progress before exiting
                    messages_for_progress = [message_dict]
                    self._update_progress(messages_for_progress, total_tokens, turn_count)
                    ended_normally = True
                    break
                
        except Exception as loop_error:
            # On any error, return partial conversation, token usage, and turn count
            logger.error(f"Manual MCP loop failed: {loop_error}", exc_info=True)
            sdk_format_messages = self._convert_to_sdk_format(all_messages)
            return {
                "success": False,
                "output": sdk_format_messages,
                "token_usage": total_tokens,
                "turn_count": turn_count,
                "error": str(loop_error),
                "litellm_run_model_name": self.litellm_run_model_name,
            }
        
        # Detect if we exited due to hitting the turn limit
        if (not ended_normally) and (turn_count >= max_turns):
            hit_turn_limit = True
            logger.warning(f"| Max turns ({max_turns}) exceeded); returning failure with partial output.")
            if tool_call_log_file:
                try:
                    with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"| Max turns ({max_turns}) exceeded\n")
                except Exception:
                    pass

        # Display final token usage
        if total_tokens["total_tokens"] > 0:
            log_msg = (
                f"| Token usage: Total: {total_tokens['total_tokens']:,} | "
                f"Input: {total_tokens['input_tokens']:,} | "
                f"Output: {total_tokens['output_tokens']:,}"
            )
            if total_tokens.get("reasoning_tokens", 0) > 0:
                log_msg += f" | Reasoning: {total_tokens['reasoning_tokens']:,}"
            logger.info(log_msg)
            logger.info(f"| Turns: {turn_count}")
        
        # Convert messages to SDK format for backward compatibility
        sdk_format_messages = self._convert_to_sdk_format(all_messages)
        
        return {
            "success": not hit_turn_limit,
            "output": sdk_format_messages,
            "token_usage": total_tokens,
            "turn_count": turn_count,
            "error": (f"Max turns ({max_turns}) exceeded" if hit_turn_limit else None),
            "litellm_run_model_name": self.litellm_run_model_name
        }
    
    async def _execute_litellm_tool_loop(
        self,
        instruction: str,
        functions: List[Dict],
        mcp_server: Any,
        tool_call_log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute function calling loop with LiteLLM."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": instruction}
        ]
        total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
        turn_count = 0
        max_turns = self.MAX_TURNS  # Limit turns to prevent infinite loops
        consecutive_failures = 0
        max_consecutive_failures = 3
        hit_turn_limit = False
        ended_normally = False
        
        # Convert functions to tools format for newer models
        tools = [{"type": "function", "function": func} for func in functions] if functions else None

        if tool_call_log_file and tools:
            max_name_length = max(
                len(tool.get("function", {}).get("name", ""))
                for tool in tools
            ) if tools else 15
            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                f.write("===== Available Tools =====\n")
                for tool in tools:
                    function_info = tool.get("function", {})
                    tool_name = function_info.get("name", "N/A")
                    description = function_info.get("description", "N/A")
                    f.write(f"- ToolName: {tool_name:<{max_name_length}} Description: {description}\n")
                f.write("\n===== Execution Logs =====\n")

        # Record initial state
        self._update_progress(messages, total_tokens, turn_count)
        
        try:
            while turn_count < max_turns:
                
                # Build completion kwargs
                completion_kwargs = {
                    "model": self.litellm_input_model_name,
                    "messages": messages,
                    "api_key": self.api_key,
                }
                
                # Always use tools format if available - LiteLLM will handle conversion
                if tools:
                    completion_kwargs["tools"] = tools
                    completion_kwargs["tool_choice"] = "auto"
                
                # Add reasoning_effort and base_url if specified
                if self.reasoning_effort != "default":
                    completion_kwargs["reasoning_effort"] = self.reasoning_effort
                if self.base_url:
                    completion_kwargs["base_url"] = self.base_url
                
                try:
                    # Call LiteLLM with timeout for individual call
                    response = await asyncio.wait_for(
                        litellm.acompletion(**completion_kwargs),
                        timeout = self.timeout / 2  # Use half of total timeout
                    )
                    consecutive_failures = 0  # Reset failure counter on success
                except asyncio.TimeoutError:
                    logger.warning(f"| ✗ LLM call timed out on turn {turn_count + 1}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise Exception(f"Too many consecutive failures ({consecutive_failures})")
                    await asyncio.sleep(8 ** consecutive_failures)  # Exponential backoff
                    continue
                except Exception as e:
                    logger.error(f"| ✗ LLM call failed on turn {turn_count + 1}: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise
                    if "ContextWindowExceededError" in str(e):
                        raise
                    elif "RateLimitError" in str(e):
                        await asyncio.sleep(12 ** consecutive_failures)
                    else:
                        await asyncio.sleep(2 ** consecutive_failures)
                    continue
                
                # Extract actual model name from response (first turn only)
                if turn_count == 0 and hasattr(response, 'model') and response.model:
                    self.litellm_run_model_name = response.model.split("/")[-1]
                
                # Update token usage including reasoning tokens
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = response.usage.prompt_tokens or 0
                    total_tokens_count = response.usage.total_tokens or 0
                    # Calculate output tokens as total - input for consistency
                    output_tokens = total_tokens_count - input_tokens if total_tokens_count > 0 else (response.usage.completion_tokens or 0)
                    
                    total_tokens["input_tokens"] += input_tokens
                    total_tokens["output_tokens"] += output_tokens
                    total_tokens["total_tokens"] += total_tokens_count
                    
                    # Extract reasoning tokens if available
                    if hasattr(response.usage, 'completion_tokens_details'):
                        details = response.usage.completion_tokens_details
                        if hasattr(details, 'reasoning_tokens'):
                            total_tokens["reasoning_tokens"] += details.reasoning_tokens or 0
                
                # Get response message
                choices = response.choices
                if len(choices):
                    message = choices[0].message
                    message_dict = message.model_dump() if hasattr(message, 'model_dump') else dict(message)
                    
                # Log assistant's text content if present
                if hasattr(message, 'content') and message.content:
                    # Display the content with line prefix
                    for line in message.content.splitlines():
                        logger.info(f"| {line}")
                    
                    # Also log to file if specified
                    if tool_call_log_file:
                        with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"{message.content}\n")
                
                # Check for tool calls (newer format)
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    messages.append(message_dict)
                    turn_count += 1
                    # Update progress after assistant with tool calls
                    self._update_progress(messages, total_tokens, turn_count)
                    # Process tool calls
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        
                        try:
                            result = await asyncio.wait_for(
                                mcp_server.call_tool(func_name, func_args),
                                timeout=60
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result)
                            })
                        except asyncio.TimeoutError:
                            error_msg = f"Tool call '{func_name}' timed out after 60 seconds"
                            logger.error(error_msg)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {error_msg}"
                            })
                        except Exception as e:
                            logger.error(f"Tool call failed: {e}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {str(e)}"
                            })   
                            
                        # Format arguments for display (truncate if too long)
                        args_str = json.dumps(func_args, separators=(",", ": "))
                        display_arguments = args_str[:140] + "..." if len(args_str) > 140 else args_str
                        
                        # Log with ANSI color codes (bold tool name, dim gray arguments)
                        logger.info(f"| \033[1m{func_name}\033[0m \033[2;37m{display_arguments}\033[0m")
                        
                        if tool_call_log_file:
                            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"| {func_name} {args_str}\n")
                    # Update progress after tool results appended
                    self._update_progress(messages, total_tokens, turn_count)
                    continue
                else:
                    # Log end reason
                    if not choices:
                        logger.info("|\n|\n| Task ended with no messages generated by the model.")
                    elif choices[0].finish_reason == "stop":
                        logger.info("|\n|\n| Task ended with the finish reason from messages being 'stop'.")
                    
                    # No tool/function call, add message and we're done
                    messages.append(message_dict)
                    turn_count += 1
                    # Update progress before exiting
                    self._update_progress(messages, total_tokens, turn_count)
                    ended_normally = True
                    break
                
        except Exception as loop_error:
            # On any error, return partial conversation, token usage, and turn count
            logger.error(f"Manual MCP loop failed: {loop_error}", exc_info=True)
            sdk_format_messages = self._convert_to_sdk_format(messages)
            return {
                "success": False,
                "output": sdk_format_messages,
                "token_usage": total_tokens,
                "turn_count": turn_count,
                "error": str(loop_error),
                "litellm_run_model_name": self.litellm_run_model_name,
            }
        
        # Detect if we exited due to hitting the turn limit
        if (not ended_normally) and (turn_count >= max_turns):
            hit_turn_limit = True
            logger.warning(f"| Max turns ({max_turns}) exceeded); returning failure with partial output.")
            if tool_call_log_file:
                try:
                    with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"| Max turns ({max_turns}) exceeded\n")
                except Exception:
                    pass

        # Display final token usage
        if total_tokens["total_tokens"] > 0:
            log_msg = (
                f"| Token usage: Total: {total_tokens['total_tokens']:,} | "
                f"Input: {total_tokens['input_tokens']:,} | "
                f"Output: {total_tokens['output_tokens']:,}"
            )
            if total_tokens.get("reasoning_tokens", 0) > 0:
                log_msg += f" | Reasoning: {total_tokens['reasoning_tokens']:,}"
            logger.info(log_msg)
            logger.info(f"| Turns: {turn_count}")
        
        # Convert messages to SDK format for backward compatibility
        sdk_format_messages = self._convert_to_sdk_format(messages)
        
        return {
            "success": not hit_turn_limit,
            "output": sdk_format_messages,
            "token_usage": total_tokens,
            "turn_count": turn_count,
            "error": (f"Max turns ({max_turns}) exceeded" if hit_turn_limit else None),
            "litellm_run_model_name": self.litellm_run_model_name
        }
    


    # ==================== Format Conversion Methods ====================

    


    # ==================== MCP Server Management ====================

    async def _create_mcp_server(self) -> Any:
        """Create and return an MCP server instance."""
        # Check if service is configured for HTTP/REST mode
        use_http_mode = self.service_config.get("use_http_mode", False)
        
        # For dual-mode services, check the mode configuration
        if self.mcp_service in self.DUAL_MODE_SERVICES and use_http_mode:
            return self._create_http_server()
        
        if self.mcp_service in self.STDIO_SERVICES:
            return self._create_stdio_server()
        elif self.mcp_service in self.HTTP_SERVICES:
            return self._create_http_server()
        else:
            raise ValueError(f"Unsupported MCP service: {self.mcp_service}")
    

    def _create_stdio_server(self) -> MCPStdioServer:
        """Create stdio-based MCP server."""
        if self.mcp_service == "notion":
            notion_key = self.service_config.get("notion_key")
            if not notion_key:
                raise ValueError("Notion API key required")
            
            return MCPStdioServer(
                command="npx",
                args=["-y", "@notionhq/notion-mcp-server"],
                env={
                    "OPENAPI_MCP_HEADERS": (
                        '{"Authorization": "Bearer ' + notion_key + '", '
                        '"Notion-Version": "2022-06-28"}'
                    )
                }
            )
        
        elif self.mcp_service == "filesystem":
            test_directory = self.service_config.get("test_directory")
            if not test_directory:
                raise ValueError("Test directory required for filesystem service")
            
            return MCPStdioServer(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", str(test_directory)]
            )
        
        elif self.mcp_service in ["playwright", "playwright_webarena"]:
            browser = self.service_config.get("browser", "chromium")
            headless = self.service_config.get("headless", True)
            viewport_width = self.service_config.get("viewport_width", 1280)
            viewport_height = self.service_config.get("viewport_height", 720)
            
            args = ["-y", "@playwright/mcp@latest"]
            if headless:
                args.append("--headless")
            args.extend([
                "--isolated",
                "--no-sandbox",
                "--browser", browser,
                "--viewport-size", f"{viewport_width},{viewport_height}"
            ])
            
            return MCPStdioServer(command="npx", args=args)
        
        elif self.mcp_service == "postgres":
            host = self.service_config.get("host", "localhost")
            port = self.service_config.get("port", 5432)
            username = self.service_config.get("username")
            password = self.service_config.get("password")
            database = self.service_config.get("current_database") or self.service_config.get("database")
            
            if not all([username, password, database]):
                raise ValueError("PostgreSQL requires username, password, and database")
            
            database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            
            return MCPStdioServer(
                command="pipx",
                args=["run", "postgres-mcp", "--access-mode=unrestricted"],
                env={"DATABASE_URI": database_url}
            )
        
        else:
            raise ValueError(f"Unsupported stdio service: {self.mcp_service}")
    

    def _create_http_server(self) -> MCPHttpServer:
        """Create HTTP-based MCP server."""
        if self.mcp_service == "github":
            github_token = self.service_config.get("github_token")
            if not github_token:
                raise ValueError("GitHub token required")
            
            return MCPHttpServer(
                url="https://api.githubcopilot.com/mcp/",
                headers={
                    "Authorization": f"Bearer {github_token}",
                    "User-Agent": "MCPMark/1.0"
                }
            )
        
        elif self.mcp_service == "filesystem":
            rest_url = self.service_config.get("rest_url", "http://127.0.0.1:8001")
            rest_headers = self.service_config.get("rest_headers", {})
            logger.info(f"Connecting to filesystem MCP REST server at: {rest_url}")
            return MCPRestClient(url=rest_url, headers=rest_headers)
        
        else:
            raise ValueError(f"Unsupported HTTP service: {self.mcp_service}")
    
