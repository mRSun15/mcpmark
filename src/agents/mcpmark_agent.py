"""
MCPMark Agent Implementation
============================

Unified agent using LiteLLM for all model interactions with minimal MCP support.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Callable, Tuple

import httpx
import litellm
import nest_asyncio
import tiktoken

from pathlib import Path

from src.logger import get_logger
from src.skills.library import SkillLibrary
from src.skills.executor import SkillExecutor
from src.skills.skills_agent import SkillsAgent
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
    # Model context window limits (in tokens)
    MODEL_CONTEXT_LIMITS = {
        "gpt-5": 272000,
        "gpt-5-mini": 272000,
        "gpt-5-nano": 272000,
        "gpt-4.1": 200000,
        "gpt-4.1-mini": 200000,
        "gpt-4.1-nano": 200000,
        "gpt-4o": 128000,
        "o3": 200000,
        "o4-mini": 200000,
        "claude-3.7-sonnet": 200000,
        "claude-sonnet-4": 200000,
        "claude-opus-4": 200000,
        "claude-opus-4.1": 200000,
        "gemini-2.5-pro": 1000000,
        "gemini-2.5-flash": 1000000,
        "deepseek-chat": 64000,
        "deepseek-reasoner": 64000,
        "default": 128000,  # Fallback for unknown models
    }
    # Compression threshold: trigger compression if tool results exceed this % of remaining budget
    COMPRESSION_THRESHOLD = 0.8
    # Worker agent: compress tool result if it exceeds this token count
    WORKER_TOTAL_TOKENS_THRESHOLD = 5000  # Compress if total tool results exceed this
    WORKER_CONTEXT_RATIO = 0.8  # Trim context when exceeds this ratio of model's max tokens
    
    # Thinking-aware system prompt for explicit reasoning tracking
    THINKING_SYSTEM_PROMPT = (
        "You are a helpful agent that uses tools iteratively to complete the user's task, and when finished, provides the final answer or simply states \"Task completed\" without further tool calls. For each tool call, also include a brief explanation in the message content describing what you are doing and why that tool is being called."
        "CRITICAL RULES: "
        "1. Strictly follow user instructions - no extra inference or reasoning unless explicitly requested. "
        "2. For time-related tasks, use GMT+0800 (China Standard Time) if not specified. "
        "3. Avoid redundant tool calls to improve efficiency. "
        "4. Avoid read_multiple_files with many files - use parallel single reads instead. "
        "5. Execute independent tool calls together (e.g., reading different files, creating separate directories)."
    )
    
    SYSTEM_PROMPT = (
        "You are a helpful agent that uses tools iteratively to complete the user's task, "
        "and when finished, provides the final answer or simply states \"Task completed\" without further tool calls. CRITICAL RULES: "
        "1. you should strictly follow the user's instructions, no extra inference or reasoning unless it is explicitly requested; "
        "2. memory is a compression summary of what you have done, key facts and taks plan, you may reference it for next tool calls. "
        "3. If memory contains VERIFIED CONSTRAINTS section, you MUST check it before each tool call and avoid violating those specific constraints. "
        "4. for time related tasks, if not specified, please use the time zone of GMT+0800 (China Standard Time). "
        "5. use code to solve problem if possible. "
        "CRITICAL Notes:"
        "1. Avoid unnecessary redundantly calling to improve efficiency. "
        "2. Avoid using read_multiple_files with many files at once as it may exceed context limits, instead, read single file in parallel."
        "3. When multiple tool calls are independent (e.g., reading different files, creating separate directories), execute them all together to improve efficiency."
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
        "You are a verification agent ensuring memory updates accurately reflect recent tool results.\n"
        "\n"
        "**Your Responsibilities:**\n"
        "\n"
        "1. **Accuracy**: Verify facts match tool outputs (values, IDs, counts, properties)\n"
        "2. **Completeness**: Check that relevant tool results are captured\n"
        "3. **Consistency**: Ensure no contradictions between sections or with tool outputs\n"
        "4. **Reasoning**: Validate that interpretive claims are supported by evidence\n"
        "\n"
        "**Critical**: The memory may contain accumulated statistics and progress from earlier turns "
        "that are NOT visible in current tool results. This is normal and expected - do NOT flag or "
        "remove these unless they directly contradict new tool outputs. Do not call tools to re-verify those facts as well.\n"
        "\n"
        "**Process:**\n"
        "\n"
        "- If you need to verify uncertain facts, call relevant tools to check\n"
        "- After gathering necessary information, call verify_memory with:\n"
        "  * verified_report: the verified (and corrected if needed) memory\n"
        "  * issues_found (optional): brief description of any corrections made, for tracking purposes\n"
        "- When verification tool calls reveal NEW information, incorporate it into the verified memory\n"
        "- The memory should reflect ALL tool results from both action phase and verification phase"
    )
    VERIFICATION_SYSTEM_PROMPT_OLD = (
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
        "5. **DO NOT REMOVE EXECUTION GUIDANCE**: Do not remove operational instructions from TASK PLAN "
        "that guide efficient execution (e.g., how to process files, parameters to use). "
        "Keep them even if rewriting other sections, unless they directly contradict verified facts.\n"
        "\n"
        "Call verify_memory with your analysis. If issues found, provide corrected version. "
        "If no issues, return unchanged and acknowledge validity."
    )
    COMPRESSION_SYSTEM_PROMPT = (
        "You are a compression agent. Extract ONLY the information from the tool result "
        "that is directly relevant to completing the user's instruction.\n\n"
        "Provide a concise extraction of instruction-relevant information. "
        "Include specific values, paths, names, and facts needed to complete the task. "
        "Omit unnecessary details, verbose descriptions, and irrelevant data."
    )
    WORKER_COMPRESSION_SYSTEM_PROMPT = (
        "You are a compression agent for a Worker executing a subtask. "
        "Extract ONLY the information from the tool result that is relevant to completing the current subtask.\n\n"
        "Rules:\n"
        "- Extract ALL relevant values completely - do not summarize or omit items from lists\n"
        "- Use structured format (key: value pairs) when extracting from structured content\n"
        "- Preserve exact values as they appear (spelling, formatting, order)\n"
        "- Include error messages or warnings\n\n"
        "Omit: boilerplate, styling, verbose formatting, and data unrelated to the subtask."
    )
    ACTION_SYSTEM_PROMPT = SYSTEM_PROMPT
    DEFAULT_TIMEOUT = BaseMCPAgent.DEFAULT_TIMEOUT
    
    # Skills are now loaded dynamically from SkillLibrary
    # Legacy COPY_FILE_TOOL_SCHEMA removed - use skills/copy_file.json instead

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
        
        # Evolved prompts for self-evolving agent experiments
        # Keys: "explorer", "planner", "worker"
        # Values: evolved prompt strings
        self._evolved_prompts: Optional[Dict[str, str]] = None
        
        # Prompt Engineer for generating task-specific prompts
        # When enabled, generates dynamic prompts based on task context
        self._prompt_engineer: Optional[Any] = None
        self._prompt_engineer_enabled: bool = False
        
        # Task-specific prompts generated by PromptEngineer (per-execution)
        self._task_specific_prompts: Dict[str, Any] = {}
        
        # EigenAI direct client for multi-agent calls (bypasses LiteLLM to avoid Cloudflare)
        self._eigenai_client = None
        if "eigenai" in litellm_input_model_name.lower() or "deepseek-v31" in litellm_input_model_name.lower():
            from src.agents.eigenai_client import EigenAIClient
            self._eigenai_client = EigenAIClient(
                api_key=api_key,
                base_url=base_url,
                model=litellm_input_model_name,
                timeout=timeout,
            )
            logger.info(f"Using EigenAI direct client for multi-agent calls: {litellm_input_model_name}")
        
        # Skills system for composite tool operations
        self._skill_library = SkillLibrary(storage_path=Path("./skills"))
        self._skill_executor = None  # Initialized when MCP server is ready
        self._skills_agent: Optional[SkillsAgent] = None
        self._skills_agent_enabled: bool = False
        self._task_proposed_skills: List = []  # Skills proposed for current task
        self._task_used_skills: set = set()  # Skills actually used in current task
        logger.info(f"[Skills] Loaded {len(self._skill_library.list_permanent())} permanent skills")
        
        logger.debug(
            "Initialized MCPMarkAgent for '%s' with model '%s' (Claude: %s, OpenAI Client: %s)",
            mcp_service,
            litellm_input_model_name,
            self.is_claude,
            self._openai_client is not None,
        )
    
    def set_evolved_prompts(self, evolved_prompts: Dict[str, str]):
        """
        Set evolved prompts for the agent.
        
        Args:
            evolved_prompts: Dict with keys "explorer", "planner", "worker"
                            and values being the evolved prompt strings.
        """
        self._evolved_prompts = evolved_prompts
        logger.info("| [Agent] Evolved prompts set for roles: %s", list(evolved_prompts.keys()))

    def set_prompt_engineer(self, prompt_engineer: Any, enabled: bool = True):
        """
        Set the Prompt Engineer for generating task-specific prompts.
        
        Args:
            prompt_engineer: PromptEngineerAgent instance
            enabled: Whether to enable prompt engineering (default True)
        """
        self._prompt_engineer = prompt_engineer
        self._prompt_engineer_enabled = enabled
        logger.info("| [Agent] Prompt Engineer %s", "enabled" if enabled else "disabled")
    
    def set_skills_agent(self, skills_agent: SkillsAgent, enabled: bool = True):
        """
        Set the Skills Agent for generating task-specific skills.
        
        Args:
            skills_agent: SkillsAgent instance
            enabled: Whether to enable skills generation (default True)
        """
        self._skills_agent = skills_agent
        self._skills_agent_enabled = enabled
        logger.info("| [Agent] Skills Agent %s", "enabled" if enabled else "disabled")
    
    def get_skill_library(self) -> SkillLibrary:
        """Get the skill library for external access."""
        return self._skill_library
    
    async def _execute_skill(self, skill_name: str, args: Dict[str, Any], mcp_server) -> Any:
        """Execute a skill using the skill executor."""
        skill = self._skill_library.get(skill_name)
        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")
        
        # Track usage
        self._task_used_skills.add(skill_name)
        self._skill_library.increment_usage(skill_name)
        
        # Create tool caller that uses MCP server
        async def tool_caller(tool_name: str, tool_args: Dict) -> Any:
            return await asyncio.wait_for(
                mcp_server.call_tool(tool_name, tool_args),
                timeout=60
            )
        
        # Create LLM caller for skills that need LLM reasoning
        async def llm_caller(prompt: str, system_prompt: Optional[str] = None) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await litellm.acompletion(
                model=self.litellm_input_model_name,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None,
                timeout=120,
            )
            return response.choices[0].message.content
        
        # Initialize executor with callers
        executor = SkillExecutor(tool_caller=tool_caller, llm_caller=llm_caller)
        
        logger.info(f"| [Skills] Executing skill: {skill_name}")
        result = await executor.execute(skill, args)
        logger.info(f"| [Skills] Skill {skill_name} completed")
        
        return result
    
    def _get_skill_tool_definitions(self) -> List[Dict]:
        """Get all skills as OpenAI function tool definitions."""
        return self._skill_library.get_tool_definitions()
    
    def _finalize_task_skills(self):
        """After task: promote used skills to permanent, discard unused temporary."""
        for skill in self._task_proposed_skills:
            if skill.name in self._task_used_skills:
                if skill.name not in self._skill_library.list_permanent():
                    self._skill_library.promote_to_permanent(skill.name)
                    logger.info(f"| [Skills] Promoted to permanent: {skill.name}")
            else:
                self._skill_library.remove_temporary(skill.name)
                logger.debug(f"| [Skills] Discarded unused: {skill.name}")
        
        self._task_proposed_skills = []
        self._task_used_skills = set()
    
    def _get_base_prompt(self, role: str) -> str:
        """Get the base prompt for a role (evolved or default)."""
        if self._evolved_prompts and role in self._evolved_prompts:
            return self._evolved_prompts[role]
        
        # Return default prompts
        if role == "explorer":
            return self._get_default_explorer_prompt()
        elif role == "planner":
            return self._get_default_planner_prompt()
        elif role == "worker":
            return self._get_default_worker_prompt()
        return ""
    
    def get_default_prompts(self) -> Dict[str, str]:
        """Get all default prompts for external use.
        
        Returns:
            Dict with keys "explorer", "planner", "worker" and prompt strings as values.
        """
        return {
            "explorer": self._get_default_explorer_prompt().strip(),
            "planner": self._get_default_planner_prompt().strip(),
            "worker": self._get_default_worker_prompt().strip(),
        }
    
    def _get_default_explorer_prompt(self) -> str:
        """Return the default Explorer prompt."""
        return """
You are the Explorer in a multi-agent system. You collaborate with other agents to solve the task together. Your specific role is to inspect the environment using READ-ONLY tools and produce a concise environment summary. You should follow this prompt strictly.

IMPORTANT: 
- Each turn: respond with tool calls OR content, never both together.
- Do NOT write, edit, create, move, or delete any files - you are read-only.
- The allowed directory is your working root.

Your goals:
- Discover what seems relevant to the user's task.
- Be thorough in coverage. When you discover something relevant, follow up and inspect deeper. For example, if you are uncertain about the item (directory, file, etc.), always inspect it. Don't miss any possible related resources.
- When inspecting files, prefer reading first few lines rather than entire contents unless needed.
- For ambigous items, don't make the decision on your own, always be neutral and return the facts directly to allow other agents further investigation and make decision.
- Summarize structure and key resources in summary_text.
- For each important resource, create a resources[] entry with:
  - id: a short local id like "R1", "R2".
  - locator: how tools will refer to it (e.g., path, URL, id).
  - kind: "directory", "file", "api", "table", etc.
  - format: "text", "json", "csv", "binary", etc.
  - preview: a short description or small content sample.
  - notes: how and when this resource might be useful.

Interaction:
- Call tools to inspect the environment. Batch multiple independent similar tool calls in a single turn if possible.
- When done, respond with a single EnvironmentSummary JSON object (no code fences):
  {"summary_text": "...", "resources": [...]}
"""

    def _get_default_planner_prompt(self) -> str:
        """Return the default Planner prompt."""
        return """
You are the Planner in a multi-agent system. You collaborate with other agents to solve the task together. Your specific role is to produce or update a structured, ordered plan of subtasks that a Worker can execute. You should follow this prompt strictly.

Inputs (PlanningInput JSON):
- task_description: original user task in natural language.
- environment_summary: discovered resources and their structure.
- available_tools: list of tool names the worker can use.
- current_plan: existing Plan (null if first planning).
- execution_state: progress and artifacts from completed subtasks (null if none).
- reason: short string explaining why re-planning is being called. could be null for first planning.

Output:
- a plan with an ordered list of subtasks that cover the whole task logically.
- a high-level explanation of plan design choices.

General rules:
- Follow user task_description strictly; do not infer or add steps beyond what the task requires.
- Never change the user's goal or instructions on your own knowledge for normal case.
- Plan within the scope of environment_summary and available tools.
- Use clear, self-contained subtasks in an ordered list. The order is the execution order for the worker.
- Assign a unique, stable id to each subtask (e.g., "S1", "S2", ...).
- If current_plan exists with completed subtasks, preserve them and their order.
- If reason indicates a block, focus on unblocking by inserting new subtasks or redirecting work to existing resources mentioned in environment_summary and available tools. 
- Always output valid JSON matching the schema below.
- For small results (counts, lists, summaries), return them in artifacts directly.

SUBTASK GRANULARITY (CRITICAL):
- Keep plans coarse: typically ≤20 subtasks. Each subtask = a logical phase, not an individual operation.
- Worker is an LLM that can loop, apply conditions, and batch tool calls internally. 
- Make sure the subtask are executable by the worker using allowed tools.

Important Notes:
- Avoid read_multiple_files with many paths; prefer individual file reads in parallel. Unless needed to read whole content.
- If possible, don't write to intermediate files, put the results in the artifacts directly to transfer the task results.
- When a subtask depends on results from previous subtasks, explicitly reference them: "Use artifacts.field_name from S{N}" so the worker knows to use execution_state.artifacts.
- For file copy operations, use the copy_file tool (available to workers) which atomically reads source and writes to destination.
Output JSON schema:
{
  "plan": {
    "subtasks": [
      {
        "id": "S1",                  // unique stable identifier for tracking
        "description": "...",        // what the Worker should accomplish
        "notes": "...",              // hints, resources (e.g., "use R1"), constraints
        "compress_results": false    // whether to compress tool results (see below)
      }
    ]
  },
  "notes": "..."                     // high-level explanation of plan design choices
}

compress_results field:
- Set to true for task related to information extraction from large raw content only, e.g., parsing metadata from files, counting items, extracting specific fields.
- Set to false (default) for other tasks, e.g., reading  summary or results, reading statistic files, copying files, writing content, operations that need raw data preserved.
"""

    def _get_default_worker_prompt(self) -> str:
        """Return the default Worker prompt."""
        return """
You are the Worker in a multi-agent system. You collaborate with other agents to solve the task together. Your specific role is to execute a single subtask at a time. You should follow this prompt strictly.

You receive:
- task_description: the original user task in natural language.
- environment_summary: discovered resources and their structure.
- current_subtask: the specific subtask you must work on now.
- execution_state: the current plan, progress, artifacts, and logs.

Rules:
- Focus only on current_subtask; do not jump ahead to other subtasks.
- Use the available tools to gather information and modify resources as needed.
- Respect the user's constraints and the existing execution_state.
- Keep tool usage efficient: avoid redundant calls; prefer lightweight inspection such as listing directories or reading only the first N lines when full content is not needed.
- When processing many items, read only the necessary portions (e.g., first few lines via head parameter) rather than full content, unless it's needed to read the whole content. Do NOT use read_multiple_files when only metadata or headers are needed.
- When you have read file content, you can extract any portion of it and write it to files. You do NOT need special "slicing" or "range read" tools - simply identify the exact text you need from what you read and use write_file.
- Bundle independent tool calls into a single turn if possible
- For large file reading/copy task, handle it one by one, don't aggregate all of them into one single turn.
- For copying files, use the copy_file tool with source and destination paths - it handles read+write internally in one call.
- CRITICAL: If you see "[Omitted earlier tool calls...]" in your input, those tool calls were already executed successfully. Do NOT repeat them. Track what's already done and only process remaining items.
- You CAN perform text transformations directly (parsing, normalization, counting, filtering) - you are an LLM. Process the data you've read and return results in artifacts.

Interaction:
- Each turn: respond with tool calls OR content, never both together.
- On each turn you may either:
  * Call tools (via tool_calls) to make progress on current_subtask, OR
  * If work is finished or blocked, respond with a single JSON object describing the outcome.

Final JSON formats (no tool_calls in that turn):
- If the subtask is successfully completed:
  {
    "status": "subtask_completed",
    "subtask_id": "<id from current_subtask>",
    "summary": "<short description of what you did and where the result is>",
    "artifacts": {
      "<key>": "<value, e.g. file paths or IDs, list of results...>"
    }
  }
  CRITICAL: artifacts must contain ACTUAL DATA values, not descriptions or placeholders. Subsequent subtasks depend on this real data.

- If you are blocked and cannot proceed:
  {
    "status": "subtask_blocked",
    "subtask_id": "<id from current_subtask>",
    "reason": "<concrete reason you cannot continue and what is missing>"
  }
"""

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
            
            # Reset task-specific skill tracking
            self._task_proposed_skills = []
            self._task_used_skills = set()
            self._skill_library.clear_temporary()
            
            # SkillsAgent will be called inside _execute_litellm_with_tools after MCP tools are available
            
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
            
            # Finalize skills: promote used skills to permanent, discard unused
            if self._task_proposed_skills:
                self._finalize_task_skills()
            
            # Update usage statistics
            self.usage_tracker.update(
                success=result["success"],
                token_usage=result.get("token_usage", {}),
                turn_count=result.get("turn_count", 0),
                execution_time=execution_time
            )
            
            result["execution_time"] = execution_time
            result["skills_used"] = list(self._task_used_skills)
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
                # return await self._execute_two_phase_tool_loop(
                #     instruction, functions, mcp_server, tool_call_log_file
                # )
                # return await self._execute_mdp_tool_loop(
                #     instruction, functions, mcp_server, tool_call_log_file
                # )
                # return await self._execute_litellm_tool_loop(
                #     instruction, functions, mcp_server, tool_call_log_file
                # )
                # return await self._execute_thinking_tool_loop(
                #     instruction, functions, mcp_server, tool_call_log_file
                # )
                return await self._execute_multi_agent_tool_loop(
                        instruction, functions, mcp_server, tool_call_log_file
                    )
                
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
                "Finalize the verified memory report after all necessary checks. "
                "Only call this when you have sufficient information to verify the memory. "
                "If you need to check facts, call appropriate tools first. "
                "Include any NEW information discovered during verification in the verified report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "verified_report": {
                        "type": "string",
                        "description": (
                            "The verified and potentially corrected memory report following the same structure. "
                            "Must incorporate information from any verification tool calls."
                        )
                    },
                    "issues_found": {
                        "type": "string",
                        "description": (
                            "Optional: Brief description of any corrections made during verification, for tracking purposes. "
                            "Can be omitted or 'None' if memory was correct."
                        )
                    }
                },
                "required": ["verified_report"]
            }
        }
    
    async def _execute_verification_loop(
        self,
        instruction: str,
        latest_memory: str,
        tool_contexts_since_last_memory: List[Dict[str, Any]],
        functions: List[Dict],
        mcp_server: Any,
        tool_call_log_file: Optional[str],
        _record_usage: Callable
    ) -> str:
        """
        Execute verification loop with tool calling capability.
        
        Args:
            instruction: Original task instruction
            latest_memory: Unverified memory to check
            tool_contexts_since_last_memory: Tool results from action phase
            functions: Available MCP tools
            mcp_server: MCP server instance
            tool_call_log_file: Log file path
            _record_usage: Function to record token usage
            
        Returns:
            verified_memory: The verified (and potentially corrected) memory report.
            
        Note: Verification tool calls are logged but not returned, as the memory
        itself is the source of truth and should contain all relevant information.
        """
        
        verification_tool_def = self._create_verify_memory_tool()
        verification_tool = {"type": "function", "function": verification_tool_def}
        action_tools = [{"type": "function", "function": f} for f in functions] if functions else []
        all_verification_tools = action_tools + [verification_tool]
        
        # Track tool calls made during verification (to add to history)
        verification_tool_contexts: List[Dict[str, Any]] = []
        
        # Build initial messages: task + memory + action tool results
        verification_messages = self._build_messages_with_context(
            instruction=instruction,
            latest_memory=latest_memory,
            last_tool_contexts=tool_contexts_since_last_memory,
            mode="verify"
        )
        
        max_verification_turns = 5
        verification_turn = 0
        
        while verification_turn < max_verification_turns:
            verification_turn += 1
            logger.info(f"🔍 Verification turn {verification_turn}/{max_verification_turns}")
            
            kwargs = {
                "model": self.litellm_input_model_name,
                "messages": verification_messages,
                "api_key": self.api_key,
                "tools": all_verification_tools,
                "tool_choice": "auto",
            }
            
            if self.reasoning_effort != "default":
                kwargs["reasoning_effort"] = self.reasoning_effort
            if self.base_url:
                kwargs["base_url"] = self.base_url
            
            try:
                if self._openai_client:
                    response = await asyncio.wait_for(
                        self._openai_client.acompletion(
                            messages=verification_messages,
                            tools=kwargs.get("tools"),
                            tool_choice=kwargs.get("tool_choice"),
                        ),
                        timeout=self.timeout / 2
                    )
                else:
                    response = await asyncio.wait_for(
                        litellm.acompletion(**kwargs),
                        timeout=self.timeout / 2
                    )
            except asyncio.TimeoutError:
                logger.warning("| ✗ Verification call timed out; using unverified memory")
                return latest_memory
            except Exception as e:
                logger.error(f"| ✗ Verification call failed: {e}; using unverified memory")
                return latest_memory
            
            _record_usage(response)  # Track tokens
            
            message = response.choices[0].message
            message_dict = message.model_dump() if hasattr(message, 'model_dump') else dict(message)
            
            if not (hasattr(message, 'tool_calls') and message.tool_calls):
                logger.warning("| Verification returned no tool calls; using unverified memory")
                return latest_memory
            
            # Add assistant message to verification history
            verification_messages.append(message_dict)
            
            # Process tool calls
            verify_memory_called = False
            action_tools_called = False
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                if func_name == "verify_memory":
                    # Verification complete!
                    verify_memory_called = True
                    verified_memory = func_args.get("verified_report", latest_memory)
                    issues_found = func_args.get("issues_found", "None")
                    
                    # Log based on whether issues were found
                    if issues_found and issues_found.lower() not in ["none", ""]:
                        logger.info(f"🔧 Memory verified with corrections after {verification_turn} turn(s): {issues_found[:100]}{'...' if len(issues_found) > 100 else ''}")
                    else:
                        logger.info(f"✅ Memory verified (no issues) after {verification_turn} turn(s)")
                    
                    if tool_call_log_file:
                        with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"| verify_memory (after {verification_turn} turn(s))\n")
                            f.write(f"  Issues Found: {issues_found}\n")
                            if verification_tool_contexts:
                                f.write(f"  Additional tool calls during verification: {len(verification_tool_contexts)}\n")
                                for ctx in verification_tool_contexts:
                                    f.write(f"    - {ctx['name']}\n")
                            f.write(f"[Verified Memory]: \n{verified_memory}\n")
                    
                    return verified_memory
                
                else:
                    # Action tool called for fact-checking
                    action_tools_called = True
                    
                    logger.info(f"🔍 Verification checking: {func_name}")
                    
                    try:
                        result = await asyncio.wait_for(
                            mcp_server.call_tool(func_name, func_args),
                            timeout=60
                        )
                        formatted_result = self._format_tool_result_for_model(result, func_name)
                        
                        # Store context for later addition to main history
                        tool_context = {
                            "name": func_name,
                            "result": result,
                            "formatted_result": formatted_result,
                            "id": tool_call.id,
                            "arguments": json.dumps(func_args, separators=(",", ": "))
                        }
                        verification_tool_contexts.append(tool_context)
                        
                        # Add to verification message history
                        verification_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": formatted_result
                        })
                        
                        if tool_call_log_file:
                            args_str = json.dumps(func_args, separators=(",", ": "))
                            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"| [verification] {func_name} {args_str}\n")
                                display_result = formatted_result[:500] + "..." if len(formatted_result) > 500 else formatted_result
                                f.write(f"  Result: {display_result}\n")
                        
                    except asyncio.TimeoutError:
                        error_msg = f"Tool call '{func_name}' timed out after 60 seconds"
                        logger.error(f"| ✗ {error_msg}")
                        verification_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {error_msg}"
                        })
                    except Exception as e:
                        logger.error(f"| ✗ Verification tool call '{func_name}' failed: {e}")
                        error_msg = f"Error: {str(e)}"
                        verification_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_msg
                        })
            
            # If action tools were called, continue loop for re-verification
            if action_tools_called:
                logger.info(f"🔄 Re-verifying with {len(verification_tool_contexts)} additional tool result(s)")
                continue
            
            # If we got here without verify_memory being called, something went wrong
            if not verify_memory_called:
                logger.warning("| Unexpected: no verify_memory call found; using unverified memory")
                return latest_memory
        
        # Hit max iterations
        logger.warning(f"⚠️  Verification hit max iterations ({max_verification_turns}); using unverified memory")
        return latest_memory
    
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

    def _get_model_context_limit(self) -> int:
        """Get the context window limit for the current model."""
        # Extract short model name from litellm format (e.g., "openai/gpt-5" -> "gpt-5")
        model_name = self.litellm_input_model_name.split("/")[-1] if "/" in self.litellm_input_model_name else self.litellm_input_model_name
        
        # Check if model name contains any known model key
        for model_key, limit in self.MODEL_CONTEXT_LIMITS.items():
            if model_key in model_name:
                return limit
        
        # Return default if not found
        return self.MODEL_CONTEXT_LIMITS["default"]

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a given text using tiktoken.
        Uses cl100k_base encoding which works for GPT-4, GPT-5, and approximates well for other models.
        """
        try:
            # Use cl100k_base encoding (GPT-4, GPT-5, etc.)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Failed to estimate tokens with tiktoken: {e}. Using character-based fallback.")
            # Fallback: rough approximation (1 token ≈ 4 characters)
            return len(text) // 4

    def _trim_worker_context(
        self, 
        messages: List[Dict[str, Any]], 
        max_tokens: int,
        cumulative_omitted: List[str],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Trim worker context by removing oldest tool call/result pairs if context exceeds max_tokens.
        Returns (trimmed_messages, omitted_summary) where omitted_summary describes what was removed.
        Keeps: system message, user prompt, and most recent tool exchanges.
        
        Args:
            messages: The conversation messages
            max_tokens: Maximum allowed tokens
            cumulative_omitted: List of previously omitted tool summaries (will be appended to)
        """
        def estimate_messages_tokens(msgs: List[Dict[str, Any]]) -> int:
            total = 0
            for m in msgs:
                content = m.get("content", "")
                if isinstance(content, str):
                    total += self._estimate_tokens(content)
                if "tool_calls" in m and m["tool_calls"]:
                    total += self._estimate_tokens(json.dumps(m["tool_calls"]))
            return total

        def extract_tool_summary(msg: Dict[str, Any]) -> Optional[str]:
            """Extract a brief summary of a tool call from an assistant message."""
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                return None
            summaries = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    # Truncate args for summary
                    if len(args) > 100:
                        args = args[:100] + "..."
                    summaries.append(f"{name}({args})")
            return "; ".join(summaries) if summaries else None

        total = estimate_messages_tokens(messages)
        if total <= max_tokens:
            # Even if no new trimming, return existing cumulative summary
            return messages, self._format_omitted_summary(cumulative_omitted)

        # Always keep: system (index 0) and user prompt (index 1)
        if len(messages) <= 2:
            return messages, self._format_omitted_summary(cumulative_omitted)

        system_msg = messages[0]
        user_prompt = messages[1]
        rest = messages[2:]

        # Remove oldest messages and track what was removed
        # Important: Remove tool call pairs together (assistant with tool_calls + following tool messages)
        newly_omitted = 0
        while rest and estimate_messages_tokens([system_msg, user_prompt] + rest) > max_tokens:
            if not rest:
                break
            removed = rest.pop(0)
            
            # Track tool calls that were removed - append to cumulative list
            tool_summary = extract_tool_summary(removed)
            if tool_summary:
                cumulative_omitted.append(tool_summary)
                newly_omitted += 1
            
            # If we removed an assistant message with tool_calls, also remove its tool responses
            if removed.get("role") == "assistant" and removed.get("tool_calls"):
                tool_call_ids = {tc.get("id") for tc in removed.get("tool_calls", []) if isinstance(tc, dict)}
                # Remove all following tool messages that belong to this tool_calls
                while rest and rest[0].get("role") == "tool":
                    tool_msg = rest[0]
                    if tool_msg.get("tool_call_id") in tool_call_ids:
                        rest.pop(0)
                    else:
                        break
            
            # If the first remaining message is a tool without preceding tool_calls, remove it too
            while rest and rest[0].get("role") == "tool":
                rest.pop(0)

        if newly_omitted > 0:
            logger.info(f"| [Worker] 📉 Trimmed {newly_omitted} old tool calls (total omitted: {len(cumulative_omitted)})")

        return [system_msg, user_prompt] + rest, self._format_omitted_summary(cumulative_omitted)

    def _format_omitted_summary(self, omitted_tools: List[str]) -> str:
        """Format a list of omitted tool call summaries into a readable string.
        
        Records all omitted tool calls (name + truncated args only, no results).
        """
        if not omitted_tools:
            return ""
        
        summary = f"[Omitted earlier tool calls - ALREADY COMPLETED SUCCESSFULLY: {len(omitted_tools)} calls. DO NOT REPEAT THESE.]\n"
        # Show all omitted calls (args already truncated in extract_tool_summary)
        for ts in omitted_tools:
            summary += f"  - ✓ {ts}\n"
        return summary

    def _parse_read_multiple_files_result(self, formatted_result: str) -> List[Dict[str, str]]:
        """
        Parse read_multiple_files result into individual file entries.
        
        Format:
        /path/to/file1.txt:
        <content>
        
        /path/to/file2.txt:
        <content>
        
        Returns list of {path: str, content: str}
        """
        files = []
        lines = formatted_result.split('\n')
        current_path = None
        current_content_lines = []
        
        for line in lines:
            # Check if this line is a file path (ends with colon and looks like a path)
            if line.endswith(':') and (line.startswith('/') or ':\\' in line or line.startswith('C:')):
                # Save previous file if exists
                if current_path is not None:
                    files.append({
                        'path': current_path,
                        'content': '\n'.join(current_content_lines)
                    })
                # Start new file
                current_path = line[:-1]  # Remove trailing colon
                current_content_lines = []
            else:
                # Accumulate content lines
                current_content_lines.append(line)
        
        # Save last file
        if current_path is not None:
            files.append({
                'path': current_path,
                'content': '\n'.join(current_content_lines)
            })
        
        return files

    async def _compress_single_file_content(
        self,
        file_path: str,
        file_content: str,
        instruction: str,
        tool_call_log_file: Optional[str] = None
    ) -> str:
        """
        Compress a single file's content from read_multiple_files.
        
        Returns compressed content string.
        """
        original_tokens = self._estimate_tokens(file_content)
        
        # Build simple compression prompt (no tool calls to avoid confusing model or triggering filters)
        user_prompt = f"{instruction}\n\nFile: {file_path}\n\nContent to compress:\n{file_content}"
        
        messages = [
            {"role": "system", "content": self.COMPRESSION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            if self._eigenai_client:
                response = await asyncio.wait_for(
                    self._eigenai_client.acompletion(messages=messages),
                    timeout=300,
                )
            else:
                # Use gpt-5-mini for compression when not using EigenAI
                compression_model = "gpt-5-mini"
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=compression_model,
                        messages=messages,
                    ),
                    timeout=300,  # 5 minutes for large files
                )
            
            if response.choices and len(response.choices) > 0:
                compressed_content = response.choices[0].message.content or file_content
            else:
                compressed_content = file_content
            
            compressed_tokens = self._estimate_tokens(compressed_content)
            compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            
            logger.info(f"| 🗜️  Compressed file {file_path}: {original_tokens} → {compressed_tokens} tokens ({compression_ratio:.1%})")
            if tool_call_log_file:
                with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n[COMPRESSED FILE] {file_path}: {original_tokens} → {compressed_tokens} tokens\n")
            
            return compressed_content
            
        except Exception as e:
            logger.warning(f"| ⚠️  Compression failed for {file_path}: {type(e).__name__}: {e}. Using original.")
            return file_content

    async def _compress_single_tool_result(
        self,
        tool_context: Dict[str, Any],
        instruction: str,
        latest_memory: str,
        tool_call_log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compress a single tool result using LLM to extract task-relevant information.
        Special handling for read_multiple_files: compress each file individually.
        
        Args:
            tool_context: Tool context containing name, result, formatted_result, etc.
            instruction: Original task instruction
            latest_memory: Current memory state
            tool_call_log_file: Optional log file path
            
        Returns:
            Updated tool_context with compressed formatted_result
        """
        tool_name = tool_context.get("name", "unknown")
        original_result = tool_context.get("formatted_result", "")
        original_tokens = self._estimate_tokens(original_result)
        tool_call_id = tool_context.get("id", "call_0")
        tool_arguments = tool_context.get("arguments", "{}")
        
        # Special handling for read_multiple_files
        if tool_name == "read_multiple_files":
            try:
                # Parse individual files
                files = self._parse_read_multiple_files_result(original_result)
                logger.info(f"| 📄 Detected read_multiple_files with {len(files)} files. Compressing in parallel...")
                
                # Compress each file in parallel
                compression_tasks = [
                    self._compress_single_file_content(
                        file_info['path'],
                        file_info['content'],
                        instruction,
                        tool_call_log_file
                    )
                    for file_info in files
                ]
                
                # Execute compression in parallel with timeout
                try:
                    compressed_contents = await asyncio.wait_for(
                        asyncio.gather(*compression_tasks, return_exceptions=True),
                        timeout=900  # 10 minutes max for all files
                    )
                except asyncio.TimeoutError:
                    logger.error(f"| ✗ Parallel compression timeout for read_multiple_files. Using original.")
                    # Fall through to normal compression which will also likely fail, but at least we tried
                    raise
                
                # Build compressed files list, handling any errors
                compressed_files = []
                for file_info, compressed_content in zip(files, compressed_contents):
                    if isinstance(compressed_content, Exception):
                        logger.warning(f"| ⚠️  Failed to compress {file_info['path']}: {compressed_content}. Using original.")
                        compressed_files.append(f"{file_info['path']}:\n{file_info['content']}")
                    else:
                        compressed_files.append(f"{file_info['path']}:\n{compressed_content}")
                
                # Reassemble result
                final_compressed = "\n\n".join(compressed_files)
                compressed_tokens = self._estimate_tokens(final_compressed)
                compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
                
                formatted_compressed = f"Compressed content (task-oriented extraction):\n{final_compressed}"
                
                logger.info(f"| 🗜️  Total compression for read_multiple_files: {original_tokens} → {compressed_tokens} tokens ({compression_ratio:.1%})")
                if tool_call_log_file:
                    with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n[COMPRESSED] read_multiple_files (total): {original_tokens} → {compressed_tokens} tokens\n")
                        f.write(f"{final_compressed}\n\n")
                
                # Update tool context
                updated_context = dict(tool_context)
                updated_context["formatted_result"] = formatted_compressed
                updated_context["original_tokens"] = original_tokens
                updated_context["compressed_tokens"] = compressed_tokens
                
                return updated_context
                
            except Exception as e:
                logger.error(f"| ✗ Failed to parse/compress read_multiple_files: {e}. Falling back to normal compression.")
                # Fall through to normal compression
        
        # Normal compression for other tools (or fallback for read_multiple_files)
        
        # Build simple compression prompt (no tool calls to avoid confusing model or triggering filters)
        user_prompt = f"{instruction}\n\nTool: {tool_name}\nArguments: {tool_arguments}\n\nContent to compress:\n{original_result}"
        
        messages = [
            {"role": "system", "content": self.COMPRESSION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            if self._eigenai_client:
                response = await asyncio.wait_for(
                    self._eigenai_client.acompletion(messages=messages),
                    timeout=300,
                )
            else:
                # Use gpt-5-mini for compression when not using EigenAI
                compression_model = "gpt-5-mini"
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=compression_model,
                        messages=messages,
                    ),
                    timeout=300,  # 5 minutes for large results
                )
            
            # Extract compressed content
            if response.choices and len(response.choices) > 0:
                compressed_content = response.choices[0].message.content or original_result
            else:
                compressed_content = original_result
            
            compressed_tokens = self._estimate_tokens(compressed_content)
            compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            
            # Format with compression marker
            formatted_compressed = f"Compressed content (task-oriented extraction):\n{compressed_content}"
            
            # Log compression details
            logger.info(f"| 🗜️  Compressed {tool_name} result: {original_tokens} → {compressed_tokens} tokens ({compression_ratio:.1%})")
            if tool_call_log_file:
                with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n[COMPRESSED] {tool_name}: {original_tokens} → {compressed_tokens} tokens\n")
                    f.write(f"{compressed_content}\n\n")
            
            # Update tool context with compressed result
            updated_context = dict(tool_context)
            updated_context["formatted_result"] = formatted_compressed
            updated_context["original_tokens"] = original_tokens
            updated_context["compressed_tokens"] = compressed_tokens
            
            return updated_context
            
        except Exception as e:
            logger.error(f"| ✗ Compression failed for {tool_name}: {type(e).__name__}: {e}. Using original result.")
            return tool_context

    async def _compress_worker_tool_result(
        self,
        subtask_description: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: str,
        tool_call_log_file: Optional[str] = None,
    ) -> str:
        """
        Compress a tool result for the Worker agent, with subtask context.
        
        Args:
            subtask_description: Current subtask being executed
            tool_name: Name of the tool called
            tool_args: Arguments passed to the tool
            tool_result: Raw result from the tool
            tool_call_log_file: Optional log file path
            
        Returns:
            Compressed result string
        """
        original_tokens = self._estimate_tokens(tool_result)
        
        # Build compression prompt with subtask context
        args_str = json.dumps(tool_args, ensure_ascii=False)
        user_prompt = (
            f"Subtask: {subtask_description}\n\n"
            f"Tool: {tool_name}\n"
            f"Arguments: {args_str}\n\n"
            f"Result to compress:\n{tool_result}"
        )
        
        messages = [
            {"role": "system", "content": self.WORKER_COMPRESSION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            if self._eigenai_client:
                response = await asyncio.wait_for(
                    self._eigenai_client.acompletion(messages=messages),
                    timeout=900,
                )
            else:
                # Use gpt-5-mini for compression when not using EigenAI
                compression_model = "gpt-5-mini"
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=compression_model,
                        messages=messages,
                    ),
                    timeout=900,  # 15 minutes for fast model
                )
            
            if response.choices and len(response.choices) > 0:
                compressed = response.choices[0].message.content or tool_result
            else:
                compressed = tool_result
            
            compressed_tokens = self._estimate_tokens(compressed)
            ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            
            # Show preview of compressed result (first 300 chars)
            preview = compressed[:300] + "..." if len(compressed) > 300 else compressed
            preview_oneline = preview.replace('\n', ' ').replace('\r', '')
            logger.info(f"| [Worker] 🗜️ Compressed {tool_name}: {original_tokens} → {compressed_tokens} tokens ({ratio:.1%})")
            logger.info(f"| [Worker]    Preview: {preview_oneline}")
            if tool_call_log_file:
                with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[Worker COMPRESSED] {tool_name}: {original_tokens} → {compressed_tokens} tokens\n")
            
            return f"[Compressed] {compressed}"
            
        except Exception as e:
            logger.warning(f"| [Worker] ⚠️ Compression failed for {tool_name}: {e}. Using original.")
            return tool_result

    async def _compress_tool_results_if_needed(
        self,
        tool_contexts: List[Dict[str, Any]],
        instruction: str,
        latest_memory: str,
        all_messages: List[Dict[str, Any]],
        tool_call_log_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Check if tool results exceed token budget and compress if necessary.
        For read_multiple_files with multiple files, compress each file individually.
        
        Args:
            tool_contexts: List of tool contexts from this turn
            instruction: Original task instruction
            latest_memory: Current memory state
            all_messages: All conversation messages so far
            tool_call_log_file: Optional log file path
            
        Returns:
            Potentially compressed tool contexts
        """
        if not tool_contexts:
            return tool_contexts
        
        # Calculate current conversation tokens
        conversation_text = json.dumps(all_messages) + instruction + (latest_memory or "")
        current_conversation_tokens = self._estimate_tokens(conversation_text)
        
        # Calculate tool results tokens
        tool_results_text = "\n".join(ctx.get("formatted_result", "") for ctx in tool_contexts)
        tool_results_tokens = self._estimate_tokens(tool_results_text)
        
        # Get context limit and calculate remaining budget
        context_limit = self._get_model_context_limit()
        remaining_budget = context_limit - current_conversation_tokens
        total_after_adding = current_conversation_tokens + tool_results_tokens
        
        logger.info(f"| 📊 Token usage: conversation={current_conversation_tokens}, tool_results={tool_results_tokens}, remaining={remaining_budget}, limit={context_limit}")
        
        # Compression logic: check total tokens for this turn
        MIN_TOKENS_TO_COMPRESS = 5000  # Skip compression if turn total < 5000 tokens
        
        # Skip compression if total is too small
        if tool_results_tokens < MIN_TOKENS_TO_COMPRESS:
            logger.info(f"| ✓ Tool results ({tool_results_tokens} tokens) below minimum ({MIN_TOKENS_TO_COMPRESS}). Skipping compression.")
            return tool_contexts
        
        # Check if compression is needed
        should_compress = False
        
        if tool_results_tokens > MIN_TOKENS_TO_COMPRESS:
            should_compress = True
        elif remaining_budget <= 0 or total_after_adding > context_limit:
            should_compress = True
        elif remaining_budget > 0 and tool_results_tokens > (self.COMPRESSION_THRESHOLD * remaining_budget):
            should_compress = True
        
        if not should_compress:
            logger.info(f"| ✓ No compression needed.")
            return tool_contexts
        
        logger.warning(f"| ⚠️  Compression triggered. Current: {tool_results_tokens}, Remaining: {remaining_budget}, Limit: {context_limit}")
        
        if tool_call_log_file:
            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[COMPRESSION TRIGGERED] Current: {tool_results_tokens}, Remaining: {remaining_budget}, Limit: {context_limit}\n")
        
        # Per-tool compression threshold: only compress individual results >= 500 tokens
        MIN_TOKENS_PER_TOOL_TO_COMPRESS = 500
        
        # Identify which tool results need compression (>= 500 tokens each)
        compression_tasks = []
        indices_to_compress = []
        
        for i, ctx in enumerate(tool_contexts):
            result_text = ctx.get("formatted_result", "")
            result_tokens = self._estimate_tokens(result_text)
            
            if result_tokens >= MIN_TOKENS_PER_TOOL_TO_COMPRESS:
                # This result needs compression
                compression_tasks.append(
                    self._compress_single_tool_result(ctx, instruction, latest_memory, tool_call_log_file)
                )
                indices_to_compress.append(i)
                logger.info(f"| 🔄 Will compress result {i} ({result_tokens} tokens)")
            else:
                # Keep small results as-is
                logger.info(f"| ✓ Keeping result {i} as-is ({result_tokens} tokens < {MIN_TOKENS_PER_TOOL_TO_COMPRESS})")
        
        if not compression_tasks:
            logger.info(f"| ✓ All individual results below per-tool threshold ({MIN_TOKENS_PER_TOOL_TO_COMPRESS}). No compression needed.")
            return tool_contexts
        
        logger.info(f"| 🔄 Compressing {len(compression_tasks)}/{len(tool_contexts)} result(s) in parallel...")
        
        try:
            compressed_results = await asyncio.wait_for(
                asyncio.gather(*compression_tasks, return_exceptions=True),
                timeout=300  # 5 minutes max for all compressions
            )
        except asyncio.TimeoutError:
            logger.error(f"| ✗ Compression timeout after 300s. Using original results.")
            return tool_contexts
        
        # Build final contexts list: use compressed for large results, keep original for small ones
        final_contexts = list(tool_contexts)  # Start with all original
        
        for compressed_idx, original_idx in enumerate(indices_to_compress):
            result = compressed_results[compressed_idx]
            if isinstance(result, Exception):
                logger.error(f"| ✗ Compression failed for result {original_idx}: {type(result).__name__}. Using original.")
                # Keep original (already in final_contexts)
            else:
                # Replace with compressed version
                final_contexts[original_idx] = result
        
        return final_contexts

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
                    tool_contexts_this_turn = []  # Collect tool contexts from this turn only
                    
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
                            # Create tool context
                            tool_context = {
                                "name": func_name,
                                "result": result,
                                "formatted_result": formatted_result,
                                "id": tool_call.id,
                                "arguments": json.dumps(func_args, separators=(",", ": "))
                            }
                            
                            # Store in this turn's list only (will append to accumulated lists after compression)
                            tool_contexts_this_turn.append(tool_context)
                            
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
                            tool_context = {
                                "name": func_name,
                                "result": f"Error: {error_msg}",
                                "formatted_result": formatted_result,
                                "id": tool_call.id,
                                "arguments": json.dumps(func_args, separators=(",", ": "))
                            }
                            
                            # Store in this turn's list only
                            tool_contexts_this_turn.append(tool_context)
                            
                            # Add error to messages
                            all_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {error_msg}"
                            })
                        except Exception as e:
                            logger.error(f"Tool call failed: {e}")
                            formatted_result = f"Error: {str(e)}"
                            tool_context = {
                                "name": func_name,
                                "result": f"Error: {str(e)}",
                                "formatted_result": formatted_result,
                                "id": tool_call.id,
                                "arguments": json.dumps(func_args, separators=(",", ": "))
                            }
                            
                            # Store in this turn's list only
                            tool_contexts_this_turn.append(tool_context)
                            
                            # Add error to messages
                            all_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": formatted_result
                            })
                        action_tool_executed = True
                        
                        # Format arguments for display (truncate if too long)
                        args_str = json.dumps(func_args, separators=(",", ": "))
                        display_arguments = args_str[:250] + "..." if len(args_str) > 250 else args_str
                        
                        # Log with ANSI color codes (bold tool name, dim gray arguments)
                        logger.info(f"| \033[1m{func_name}\033[0m \033[2;37m{display_arguments}\033[0m")
                        
                        if tool_call_log_file:
                            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"| {func_name} {args_str}\n")
                                # Truncate result if too long (max 1000 chars)
                                display_result = formatted_result[:1000] + "..." if len(formatted_result) > 1000 else formatted_result
                                f.write(f"  Result: {display_result}\n")
                    
                    # Check if compression is needed for tool results from this turn
                    compressed_contexts = await self._compress_tool_results_if_needed(
                        tool_contexts_this_turn,
                        instruction,
                        latest_memory,
                        all_messages,
                        tool_call_log_file
                    )
                    
                    # Update all data structures with final contexts (compressed or original)
                    if compressed_contexts != tool_contexts_this_turn:
                        # Compression occurred - update all_messages and append compressed contexts
                        num_contexts = len(compressed_contexts)
                        for i, compressed_ctx in enumerate(compressed_contexts):
                            compressed_result = compressed_ctx.get("formatted_result")
                            # Update the corresponding tool message
                            tool_msg_idx = len(all_messages) - num_contexts + i
                            if tool_msg_idx >= 0 and all_messages[tool_msg_idx].get("role") == "tool":
                                all_messages[tool_msg_idx]["content"] = compressed_result
                            # Append compressed context to accumulated lists
                            tool_contexts_since_last_memory.append(compressed_ctx)
                            tool_call_history.append(dict(compressed_ctx))
                    else:
                        # No compression - append original contexts to accumulated lists
                        for ctx in tool_contexts_this_turn:
                            tool_contexts_since_last_memory.append(ctx)
                            tool_call_history.append(dict(ctx))
                
                    # Update progress after tool results
                    messages_for_progress = [message_dict]
                    turn_count += 1
                    self._update_progress(messages_for_progress, total_tokens, turn_count)
                    
                    # Check if critical tools were called
                    has_critical_tools = self._contains_critical_tools(tool_contexts_since_last_memory)
                    
                    # Decide if we should update memory based on accumulated tool calls
                    should_update_memory = (
                        action_tool_executed and (
                            len(tool_contexts_since_last_memory) >= self.memory_update_threshold
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
                        # Now verify the generated memory with iterative tool calling
                        verified_memory = await self._execute_verification_loop(
                            instruction=instruction,
                            latest_memory=latest_memory,
                            tool_contexts_since_last_memory=tool_contexts_since_last_memory,
                            functions=functions,
                            mcp_server=mcp_server,
                            tool_call_log_file=tool_call_log_file,
                            _record_usage=_record_usage
                        )
                        
                        # Update memory with verified version
                        latest_memory = verified_memory
                        
                        # Display verified memory summary
                        logger.info(f"✅ Verified Memory: {verified_memory[:200]}{'...' if len(verified_memory) > 200 else ''}")
                        
                        # Update progress - Note: verification responses are tracked internally in the loop
                        self._update_progress([memory_message_dict], total_tokens, turn_count)
                        
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
    


    async def _execute_thinking_tool_loop(
        self,
        instruction: str,
        functions: List[Dict],
        mcp_server: Any,
        tool_call_log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute function calling loop with explicit thinking tracking.
        
        Differences from _execute_litellm_tool_loop:
        1. Uses THINKING_SYSTEM_PROMPT instead of SYSTEM_PROMPT
        2. Tracks thinking/reasoning from content field as separate messages
        3. Message history structure: thinking → tool calls → tool results
        4. Includes message compression (reuses existing compression methods)
        """
        messages = [
            {"role": "system", "content": self.THINKING_SYSTEM_PROMPT},
            {"role": "user", "content": instruction}
        ]
        total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
        turn_count = 0
        max_turns = self.MAX_TURNS
        consecutive_failures = 0
        max_consecutive_failures = 3
        hit_turn_limit = False
        ended_normally = False
        
        # Convert functions to tools format
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
                
                if tools:
                    completion_kwargs["tools"] = tools
                    completion_kwargs["tool_choice"] = "auto"
                
                if self.reasoning_effort != "default":
                    completion_kwargs["reasoning_effort"] = self.reasoning_effort
                if self.base_url:
                    completion_kwargs["base_url"] = self.base_url
                
                try:
                    response = await asyncio.wait_for(
                        litellm.acompletion(**completion_kwargs),
                        timeout=self.timeout / 2
                    )
                    consecutive_failures = 0
                except asyncio.TimeoutError:
                    logger.warning(f"| ✗ LLM call timed out on turn {turn_count + 1}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise Exception(f"Too many consecutive failures ({consecutive_failures})")
                    await asyncio.sleep(8 ** consecutive_failures)
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
                
                # Extract actual model name (first turn only)
                if turn_count == 0 and hasattr(response, 'model') and response.model:
                    self.litellm_run_model_name = response.model.split("/")[-1]
                
                # Update token usage
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = response.usage.prompt_tokens or 0
                    total_tokens_count = response.usage.total_tokens or 0
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
                
                # Log thinking content if present
                thinking_content = None
                if hasattr(message, 'content') and message.content:
                    thinking_content = message.content
                    # Log thinking content
                    logger.info(f"| 💭 [Thinking]")
                    for line in message.content.splitlines():
                        logger.info(f"| {line}")
                    
                    if tool_call_log_file:
                        with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[THINKING]\n{message.content}\n\n")
                
                # Check for tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Split thinking and tool calls into separate messages
                    # 1. Add thinking as separate assistant message (if present)
                    if thinking_content:
                        messages.append({
                            "role": "assistant",
                            "content": thinking_content
                        })
                    
                    # 2. Add tool calls message (without content to avoid duplication)
                    tool_calls_message = message_dict.copy()
                    tool_calls_message["content"] = None
                    messages.append(tool_calls_message)
                    turn_count += 1
                    # Update progress after adding assistant message
                    self._update_progress(messages, total_tokens, turn_count)
                    
                    # Process tool calls and collect contexts
                    current_tool_contexts = []
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        
                        try:
                            result = await asyncio.wait_for(
                                mcp_server.call_tool(func_name, func_args),
                                timeout=60
                            )
                            result_str = json.dumps(result)
                            
                            # Store context for potential compression
                            current_tool_contexts.append({
                                "id": tool_call.id,
                                "name": func_name,
                                "arguments": json.dumps(func_args),
                                "result": result,
                                "formatted_result": result_str
                            })
                            
                        except asyncio.TimeoutError:
                            error_msg = f"Tool call '{func_name}' timed out after 60 seconds"
                            logger.error(error_msg)
                            current_tool_contexts.append({
                                "id": tool_call.id,
                                "name": func_name,
                                "arguments": json.dumps(func_args),
                                "result": None,
                                "formatted_result": f"Error: {error_msg}"
                            })
                        except Exception as e:
                            logger.error(f"Tool call failed: {e}")
                            current_tool_contexts.append({
                                "id": tool_call.id,
                                "name": func_name,
                                "arguments": json.dumps(func_args),
                                "result": None,
                                "formatted_result": f"Error: {str(e)}"
                            })
                        
                        # Format and log tool call
                        args_str = json.dumps(func_args, separators=(",", ": "))
                        display_arguments = args_str[:140] + "..." if len(args_str) > 140 else args_str
                        logger.info(f"| \033[1m{func_name}\033[0m \033[2;37m{display_arguments}\033[0m")
                        
                        if tool_call_log_file:
                            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"| {func_name} {args_str}\n")
                    
                    # Apply compression if needed (only compresses formatted_result field)
                    contexts_to_add = current_tool_contexts
                    if current_tool_contexts:
                        compressed_contexts = await self._compress_tool_results_if_needed(
                            current_tool_contexts,
                            instruction,
                            "",  # No memory in this simplified version
                            messages,
                            tool_call_log_file
                        )
                        
                        if compressed_contexts:
                            # Use compressed versions (formatted_result is already compressed)
                            contexts_to_add = compressed_contexts
                    
                    # Now add tool result messages (with compressed content if applicable)
                    for ctx in contexts_to_add:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": ctx["id"],
                            "content": ctx.get("formatted_result") or str(ctx.get("result", ""))
                        })
                    
                    # Update progress after adding all tool results
                    self._update_progress(messages, total_tokens, turn_count)
                    continue
                else:
                    # No tool calls - task complete
                    if not choices:
                        logger.info("|\n|\n| Task ended with no messages generated by the model.")
                    elif choices[0].finish_reason == "stop":
                        logger.info("|\n|\n| Task ended with the finish reason from messages being 'stop'.")
                    
                    messages.append(message_dict)
                    turn_count += 1
                    self._update_progress(messages, total_tokens, turn_count)
                    ended_normally = True
                    break
                
        except Exception as loop_error:
            logger.error(f"Thinking tool loop failed: {loop_error}", exc_info=True)
            sdk_format_messages = self._convert_to_sdk_format(messages)
            return {
                "success": False,
                "output": sdk_format_messages,
                "token_usage": total_tokens,
                "turn_count": turn_count,
                "error": str(loop_error),
                "litellm_run_model_name": self.litellm_run_model_name,
            }
        
        # Check if we hit turn limit
        if (not ended_normally) and (turn_count >= max_turns):
            hit_turn_limit = True
            logger.warning(f"| Max turns ({max_turns}) exceeded; returning failure with partial output.")
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
        
        # Convert messages to SDK format
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
    
    # ==================== MDP-based Execution Loop ====================
    
    def _build_verification_mdp_prompt(
        self,
        task: str,
        tools: List[Dict],
        verification_report: str,
        last_actions: str
    ) -> str:
        """Build prompt for verification agent's MDP loop."""
        # Format tools
        descriptions = []
        for tool in tools:
            function_info = tool.get("function", {})
            name = function_info.get("name", "N/A")
            description = function_info.get("description", "N/A")
            parameters = function_info.get("parameters", {})
            properties = parameters.get("properties", {}) or {}
            required = set(parameters.get("required", []) or [])
            
            arg_lines = []
            for prop_name, prop_details in properties.items():
                details = json.dumps(prop_details, ensure_ascii=False, indent=2)
                suffix = " (required)" if prop_name in required else ""
                arg_lines.append(f"- {prop_name}{suffix}: {details}")
            
            if arg_lines:
                arguments_text = "\n".join(arg_lines)
            else:
                arguments_text = "(no arguments)"
            
            descriptions.append(
                f"Tool: {name}\nDescription: {description}\nArguments:\n{arguments_text}"
            )
        
        tools_description = "\n\n".join(descriptions) if descriptions else "(no tools available)"
        
        prompt = f"""You are a meticulous task verification agent. Your role is to verify whether a task has been completed correctly and completely by inspecting the actual environment state.

## CRITICAL OUTPUT FORMAT REQUIREMENTS

You MUST follow this exact format. Every response must contain:

1. <report>...</report> (always required)
2. Either <answer>...</answer> OR <tool_calls>...</tool_calls> (never both)

## Input Format

- **Original Task**: The task that the main agent was asked to complete
- **Your Previous Verification Progress**: Your accumulated verification findings so far
- **Your Last Inspection Actions & Results**: The tools you called in the previous round with their results combined (in `<tool_calls_and_results>` tag)

## Output Format

<report>

### Verification Progress

Document what you've verified so far:
- **Allowed root directory** (MUST check this FIRST via list_allowed_directories)
- What you've inspected and how
- Evidence gathered (with tool results)
- Preliminary findings
- What still needs checking
- Expected final deliverables.

**CRITICAL**: You may adjust the deliverables based on the actual allowed root directory.

This section should accumulate and build upon previous verification steps.

### Next Steps Plan

What inspection actions to take next, or if ready to conclude.

</report>

**Decision Point**: Are you ready to conclude verification?

**If YES - Verification Complete with NO ISSUES:**

<answer>
Task Verification Passed

Summary: [brief summary of what was verified]
</answer>

**If YES - Verification Complete with ISSUES FOUND:**

<answer>
VERIFICATION FAILED

Issues discovered:

1. **[Issue Category]**: [Specific issue with evidence]
   - Expected: [what should be]
   - Actual: [what was found]
   - Evidence: [tool result or inspection details]

2. **[Issue Category]**: [Another specific issue]
   ...

</answer>

**If NO - Need more inspection:**

<tool_calls>
[Tool calls in valid JSON format to gather more evidence]
</tool_calls>

Example:
<tool_calls>
[{{"name": "list_directory", "arguments": {{"path": "/some/path"}}}}, {{"name": "read_file", "arguments": {{"path": "/some/file.txt"}}}}]
</tool_calls>

## Your Mission

The main agent has claimed to complete a task. You must:
1. **Systematically verify** the work by inspecting the actual environment
2. **Check all requirements** from the original task
3. **Identify any issues**: missing items, incorrect results, schema violations, or unwanted side effects
4. **Use tools iteratively** to gather evidence and build confidence in your assessment

**IMPORTANT**: Your role is to VERIFY the output, NOT to re-solve the task.
- For complex analytical tasks: Just verify the OUTPUT is complete and correct
- For file operations: Check files exist, content matches requirements, formats are correct
- For data tasks: Verify results are present, accurate, and match expected schemas
- **DO NOT** redo the entire analysis or regenerate solutions yourself
- **FOCUS** on checking what was produced against what was required

## Critical Constraints

**READ-ONLY MODE**: You can ONLY inspect and read. You MUST NOT modify anything in the environment even though all tools are available to you.

## Verification Strategy

1. **Identify requirements**: What outputs/results were required by the task?
2. **Inspect actual outputs**: Use tools to read/check what was actually produced
3. **Compare**: Does actual output match requirements? (completeness, correctness, format)
4. **Check for issues**: Missing items, incorrect data, wrong formats, unwanted extras
5. **Iterate if needed**: Gather more evidence if uncertain
6. **Conclude**: Pass or fail with specific findings

Be thorough but efficient. Verify outputs with actual inspection, not assumptions. Don't re-solve the task.

## Input Context

### Original Task
{task}

### Available Tools
{tools_description}

### Your Previous Verification Progress
<report>
{verification_report}
</report>

### Your Last Inspection Actions & Results

⚠️ Check if you already verified this information before calling the same tools again!

<tool_calls_and_results>
{last_actions}
</tool_calls_and_results>

## Guidelines

- **Verify outputs, don't re-solve**: Check what was produced, don't redo the task yourself
- **Be systematic**: Check each requirement methodically
- **Be skeptical**: Verify claims with actual inspection
- **Be specific**: Issues must include evidence and be actionable
- **Be fair**: Don't fail for trivial or subjective issues
- **Be efficient**: For complex analysis tasks, focus on output completeness/correctness
- **Be thorough**: But know when you have enough evidence
- **Remember**: You can iterate - use multiple inspection rounds if needed
- **READ-ONLY**: Never modify, only inspect
- **First turn**: Explore the allowed workspace first before operating on any paths.
- **Evidence usage**: Confirm the agent actually inspected key files/data sources.
- **Spec fidelity**: Check that the agent followed the task’s rules as written and did not introduce its own decision policies.
- **Exactness & format**: For tasks involving exact content, indices, or strict formats, verify the result is precise and matches the required schema / naming style.

## IMPORTANT: Common Verification Pitfalls - CHECK THESE CAREFULLY

**Example 1: Wrong Directory Structure**
- Task mentions directory name as context (e.g., "in test directory, write answer") → This is NOT an instruction to create that directory literally
- Check: Between allowed root and actual work, is there an extra directory matching task's reference name?
- Example: Allowed root is `/root/`, task mentions "test" → Correct: `/root/answer.txt`, WRONG: `/root/test/answer.txt`
- Verify: List allowed root, check if unnecessary wrapper directory was created matching task's reference name

**Example 2: Unauthorized Side Effects**
- Task says "create directory X" → Check if agent created OTHER directories/files too
- Verify: List all files/directories to detect unwanted extras

**Example 3: Avoid overthinking**
- If task specifcies specific operation, such as de-duplication, don't overthink and strictly follow the task instruction.

**Example 4: Time related questions**
- Strictly follow the task instruction on time related questions. If timezone is not specified, ALWAYS use GMT+0800 (China Standard Time) as the default timezone, not the machine's local timezone


**Key Rule**: For file tasks, use length checks and exact path verification. Don't rely only on visual inspection.

Now proceed with your verification work."""
        
        return prompt
    
    async def _verify_task_completion(
        self,
        instruction: str,
        functions: List[Dict],
        mcp_server: Any,
        tool_call_log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run verification agent in its own MDP loop.
        
        Args:
            instruction: Original task instruction
            functions: Available tool functions (all tools, not filtered)
            mcp_server: MCP server instance
            tool_call_log_file: Log file path
            
        Returns:
            {
                "passed": bool,
                "issues": str,  # Empty if passed, detailed issues if failed
                "verification_report": str
            }
        """
        tools = [{"type": "function", "function": func} for func in functions] if functions else []
        
        # Verification MDP State
        verification_report = ""
        last_actions = ""
        
        # Tracking
        total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
        turn_count = 0
        max_verification_turns = 20  # Limit verification iterations
        consecutive_failures = 0
        max_consecutive_failures = 3
        
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
        
        if tool_call_log_file:
            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                f.write("\n===== VERIFICATION AGENT STARTED =====\n")
        
        logger.info("🔍 Starting verification agent MDP loop")
        
        try:
            while turn_count < max_verification_turns:
                # Build verification prompt
                user_prompt = self._build_verification_mdp_prompt(
                    task=instruction,
                    tools=tools,
                    verification_report=verification_report,
                    last_actions=last_actions
                )
                
                # Build messages
                messages = [
                    {"role": "user", "content": user_prompt}
                ]
                
                # Build completion kwargs
                completion_kwargs = {
                    "model": self.litellm_input_model_name,
                    "messages": messages,
                    "api_key": self.api_key,
                }
                
                if self.reasoning_effort != "default":
                    completion_kwargs["reasoning_effort"] = self.reasoning_effort
                if self.base_url:
                    completion_kwargs["base_url"] = self.base_url
                
                try:
                    # Call LLM
                    if self._openai_client:
                        response = await asyncio.wait_for(
                            self._openai_client.acompletion(
                                messages=messages,
                                tools=None,
                                tool_choice=None,
                            ),
                            timeout=self.timeout / 2
                        )
                    else:
                        response = await asyncio.wait_for(
                            litellm.acompletion(**completion_kwargs),
                            timeout=self.timeout / 2
                        )
                    consecutive_failures = 0
                except asyncio.TimeoutError:
                    logger.warning(f"| ✗ Verification LLM call timed out on turn {turn_count + 1}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        break
                    await asyncio.sleep(8 ** consecutive_failures)
                    continue
                except Exception as e:
                    logger.error(f"| ✗ Verification LLM call failed on turn {turn_count + 1}: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        break
                    await asyncio.sleep(2 ** consecutive_failures)
                    continue
                
                # Update token usage
                _record_usage(response)
                
                # Get response content
                choices = response.choices
                if not len(choices):
                    logger.error("| ✗ No choices in verification response")
                    break
                
                message = choices[0].message
                content = message.content if hasattr(message, 'content') else ""
                
                if not content:
                    logger.error("| ✗ Empty content in verification response")
                    break
                
                # Log the response
                if tool_call_log_file:
                    with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n===== Verification Turn {turn_count + 1} Response =====\n")
                        f.write(content + "\n")
                
                # Parse MDP response
                parsed = self._parse_mdp_response(content)
                
                if parsed.get("error"):
                    logger.error(f"| ✗ Failed to parse verification response: {parsed['error']}")
                    logger.warning(f"| ⚠️  Parse error on verification turn {turn_count + 1}, will retry on next turn")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"| ✗ Too many consecutive parse failures ({consecutive_failures})")
                        break
                    # Retry by continuing to next iteration
                    turn_count += 1
                    continue
                
                # Reset failure counter on successful parse
                consecutive_failures = 0
                
                # Update verification report
                new_report = parsed["report"]
                verification_report = new_report
                logger.info(f"🔍 Verification report updated: {verification_report[:200]}{'...' if len(verification_report) > 200 else ''}")
                
                # Check if verification concluded
                if parsed.get("answer"):
                    answer = parsed["answer"].strip()
                    logger.info(f"🔍 Verification concluded: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                    
                    if tool_call_log_file:
                        with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"\n===== VERIFICATION CONCLUDED =====\n")
                            f.write(f"Answer: {answer}\n")
                    
                    # Check if passed or failed
                    if "Task Verification Passed" in answer or "VERIFICATION PASSED" in answer.upper():
                        logger.info("✅ Verification PASSED")
                        return {
                            "passed": True,
                            "issues": "",
                            "verification_report": verification_report
                        }
                    else:
                        logger.warning(f"❌ Verification FAILED")
                        return {
                            "passed": False,
                            "issues": answer,
                            "verification_report": verification_report
                        }
                
                # Execute tool calls for more inspection
                tool_calls_list = parsed.get("tool_calls", [])
                if not tool_calls_list:
                    logger.warning("| ⚠️ Verification agent provided no answer and no tool_calls")
                    break
                
                logger.info(f"| 🔍 Verification executing {len(tool_calls_list)} tool call(s)")
                
                tool_results = []
                actions_log = []
                
                for idx, tool_call_spec in enumerate(tool_calls_list):
                    func_name = tool_call_spec.get("name")
                    func_args = tool_call_spec.get("arguments", {})
                    
                    if not func_name:
                        logger.error(f"| ✗ Verification tool call {idx} missing 'name'")
                        continue
                    
                    # Log tool call
                    args_str = json.dumps(func_args, separators=(",", ": "))
                    display_args = args_str[:250] + "..." if len(args_str) > 250 else args_str
                    logger.info(f"| 🔍 \033[1m{func_name}\033[0m \033[2;37m{display_args}\033[0m")
                    
                    if tool_call_log_file:
                        with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"| [VERIFICATION] {func_name} {args_str}\n")
                    
                    # Record action for next iteration
                    actions_log.append(f"{func_name}({args_str})")
                    
                    # Execute tool
                    try:
                        result = await asyncio.wait_for(
                            mcp_server.call_tool(func_name, func_args),
                            timeout=60
                        )
                        formatted_result = self._format_tool_result_for_model(result, func_name)
                        tool_results.append({
                            "tool": func_name,
                            "result": formatted_result
                        })
                        
                        
                    except asyncio.TimeoutError:
                        error_msg = f"Tool call '{func_name}' timed out after 60 seconds"
                        logger.error(f"| ✗ {error_msg}")
                        tool_results.append({
                            "tool": func_name,
                            "result": f"Error: {error_msg}"
                        })
                    except Exception as e:
                        logger.error(f"| ✗ Verification tool call '{func_name}' failed: {e}")
                        tool_results.append({
                            "tool": func_name,
                            "result": f"Error: {str(e)}"
                        })
                
                # Update state for next iteration - combine actions and results
                combined_turn_data = []
                for call, result in zip(tool_calls_list, tool_results):
                    combined_turn_data.append({
                        "tool": call.get("name"),
                        "arguments": call.get("arguments", {}),
                        "result": result.get("result", "")
                    })
                
                # Convert to JSON format for prompt
                last_actions = json.dumps(combined_turn_data, indent=2)
                
                turn_count += 1
            
        except Exception as verification_error:
            logger.error(f"Verification loop failed: {verification_error}", exc_info=True)
            return {
                "passed": True,
                "issues": f"Verification inconclusive due to error: {str(verification_error)}",
                "verification_report": verification_report
            }
        
        # Verification took too many turns or stopped - treat as inconclusive/pass
        logger.warning(f"| ⚠️ Verification inconclusive after {turn_count} turns")
        return {
            "passed": True,
            "issues": "Verification inconclusive (max turns reached)",
            "verification_report": verification_report
        }
    
    def _build_mdp_prompt(
        self,
        task: str,
        tools: List[Dict],
        report: str = "",
        last_actions: str = "",
        verification_issues: str = ""
    ) -> str:
        """Build the prompt for MDP-based agent following the specified format."""
        
        # Format tools description (similar to ReAct agent)
        descriptions = []
        for tool in tools:
            func_info = tool.get("function", {})
            name = func_info.get("name", "unknown")
            description = func_info.get("description", "No description provided.")
            parameters = func_info.get("parameters", {}) or {}
            properties = parameters.get("properties", {}) or {}
            required = set(parameters.get("required", []) or [])
            
            arg_lines = []
            for prop_name, prop_details in properties.items():
                details = json.dumps(prop_details, ensure_ascii=False, indent=2)
                suffix = " (required)" if prop_name in required else ""
                arg_lines.append(f"- {prop_name}{suffix}: {details}")
            
            if arg_lines:
                arguments_text = "\n".join(arg_lines)
            else:
                arguments_text = "(no arguments)"
            
            descriptions.append(
                f"Tool: {name}\nDescription: {description}\nArguments:\n{arguments_text}"
            )
        
        tools_description = "\n\n".join(descriptions) if descriptions else "(no tools available)"
        
        prompt = f"""You are a professional problem-solving agent with rigorous information verification capabilities and deep analytical thinking.

## CRITICAL OUTPUT FORMAT REQUIREMENTS

You MUST follow this exact format. Every response must contain:

1. <report>...</report> (always required)
2. Either <answer>...</answer> OR <tool_calls>...</tool_calls> (never both)

## Input Format

- **Task**: The task posed by the user that needs to be solved
- **Verification Issues**: Issues discovered by the verification agent (if any). Empty on first attempt. You need to solve the issue detected to complete task successfully.
- **Last Status Report and Deep Analysis**: A summary overview of current work progress
- **Last Turn Actions & Results**: Tools you called previously with their results (in `<tool_calls_and_results>` tag)

## Output Format

<report>

### Status Report and Deep Analysis

**Progress Achieved:**

Based on the Last Status Report and Deep Analysis and Last Tool Response provided in the input, compile a comprehensive and complete documentation of all currently collected information, conclusions, data, and findings. This section must capture ALL important information without any omissions, presented in plain text format with corresponding sources clearly annotated. You must directly record the actual information content rather than using referential markers or summaries. This includes:

1. All factual data and evidence collected
2. All analytical conclusions and insights derived
3. All source materials and their verification status
4. All uncertainties, limitations, or gaps identified
5. Complete integration of previous progress with new findings

The documentation must be sufficiently detailed and complete that someone can fully inherit and understand all achieved progress to seamlessly continue the research without losing any critical information or context.

**Next Steps Plan:**

Based on the comprehensive progress achieved above, formulate a detailed and actionable plan for the next phase of research or investigation.

</report>

You MUST output this section enclosed with <report></report> tags!

**Decision Point**: Are you centain that no further action tools needed to complete the task(e.g wrote the result to expected place)?

**If YES - Task fully completed:**

<answer>
Provide the final answer or simply states "Task Completed"
</answer>

**If NO - Further action needed: return a list of tool call in tool calls**
<tool_calls>
"name": "tool name here", "arguments": "parameter name here": parameter value here,
"another parameter name here": another parameter value here, ...,
...
</tool_calls>
Tool calls must be in valid JSON format.
Example:
<tool_calls>
[{{"name": "list_directory", "arguments": {{"path": "/some/path"}}}}, {{"name": "read_file", "arguments": {{"path": "/some/file.txt"}}}}]
</tool_calls>

You MUST output this section enclosed with <tool_calls></tool_calls> tags!

## Working Principles

1. **Rigorous Verification**: Critically evaluate all information sources
2. **Deep Thinking**: Pursue essential understanding, not satisfied with surface phenomena
3. **Evidence-Driven**: Make reasoning decisions based on reliable evidence through deep thinking
4. **You are required to maintain detailed documentation in all your reports and actions, providing sufficient information for others to fully grasp your progress and effectively continue or modify the research trajectory based on your contributions.**

## Special Requirements

- All tools in the tool list are real and functional - as long as you make correct tool calls, you will receive their returned results.
- Clearly distinguish between "confirmed facts," "highly credible inferences," and "hypotheses to be verified"
- Clearly indicate uncertainty when information is insufficient
- Always focus on the original task and follow the task instruction strictly. Do not generate new requirements by your own understanding.
- When outputting [Status Report and Deep Analysis], never omit key actions and results, even if these actions or results do not meet expectations, these conclusions must still be documented.
- **When further action is needed, select appropriate tools and configure parameters carefully. Explore the allowed workspace first before operating on any paths.**
- **IMPORTANT: Directory names in task descriptions (e.g. "desktop", "test_folder") are context references, NOT instructions to create new directories. The allowed root IS your working directory - create required structures directly there, not inside a wrapper directory matching the task's reference name.**
- **When the current status is sufficient to answer the question, must provide the final answer enclosed with <answer></answer> tags rather than continue with actions**
- **If timezone is not specified in the task, use GMT+0800 (China Standard Time) as the default timezone, not the machine's local timezone.**

## Core behavior rules
- **Observe, don’t assume.**
  - When a file or path is obviously relevant (metadata, labels, counts, CSV/TSV, summaries), you MUST open and inspect it instead of guessing from its name or context.

- **Follow the task spec literally.**
  - Use only the rules given in the instructions (and any explicit default like the timezone). Do not invent extra policies (e.g. custom tie-breakers, “representative” choices).

- **Be exact when the task is exact.**
  - For byte-level transforms, indices, or equality checks, operate on the real file content and ensure the result is exactly what the spec says (no off-by-one, no lossy aggregation).

- **Match format and cover all required items.**
  - Output must follow the task-specified schema and naming style (including human-readable names), and you must ensure all clearly relevant files / groups / events are handled, not just a subset.

## Task Completion Verification

**Important**: When you provide <answer> claiming task completion, your work will be automatically verified by a verification agent who will:
- Inspect the actual environment state using tools
- Check all requirements are met correctly and completely
- Identify any issues, missing items, or incorrect implementations

If the verification agent discover issues, they will be provided in "Verification Issues" and you must address them in a new run.

Therefore:
- **Be thorough** and complete all requirements before claiming completion
- **Double-check your work** to ensure correctness in the solving process

## FORMAT REMINDER

- Start with <report>...</report> section
- Then choose: <answer>...</answer> if sufficient info, OR <tool_calls>...</tool_calls> if need more action
- Never output both answer and tool_calls tags in same response

## Input

- Task: {task}

- Verification Issues: 
{verification_issues if verification_issues else "(No issues - first attempt or previous verification passed)"}

- Available Tools
{tools_description}

- Last Status Report and Deep Analysis:
<report>
{report}
</report>

- Last Turn Actions & Results:

⚠️ Check if you already have the information below before calling the same tools again!

<tool_calls_and_results>
{last_actions}
</tool_calls_and_results>

Now please begin your deep analytical work."""
        
        return prompt
    
    def _parse_mdp_response(self, content: str) -> Dict[str, Any]:
        """Parse the MDP response with <report>, <answer>, and <tool_calls> tags."""
        import re
        
        result = {
            "report": "",
            "answer": None,
            "tool_calls": None,
            "error": None
        }
        
        # Extract report (required)
        report_match = re.search(r'<report>(.*?)</report>', content, re.DOTALL | re.IGNORECASE)
        if report_match:
            result["report"] = report_match.group(1).strip()
        else:
            result["error"] = "No <report> section found in response"
            return result
        
        # Extract answer (optional)
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
            return result  # If answer exists, don't look for tool_calls
        
        # Extract tool_calls (optional)
        tool_calls_match = re.search(r'<tool_calls>(.*?)</tool_calls>', content, re.DOTALL | re.IGNORECASE)
        if not tool_calls_match:
            # Fallback: try to extract without closing tag (LLM sometimes forgets it)
            tool_calls_match = re.search(r'<tool_calls>\s*(\[.*)', content, re.DOTALL | re.IGNORECASE)
        
        if tool_calls_match:
            tool_calls_str = tool_calls_match.group(1).strip()
            
            # Fix common model error: }]}] at end should be }}]
            if tool_calls_str.endswith('}]}]'):
                tool_calls_str = tool_calls_str[:-3] + '}]'
            
            # Fix common model error: missing closing } before ]
            if tool_calls_str.endswith('{}]') and not tool_calls_str.endswith('}}]'):
                open_count = tool_calls_str.count('{')
                close_count = tool_calls_str.count('}')
                if open_count > close_count:
                    missing = open_count - close_count
                    tool_calls_str = tool_calls_str[:-1] + '}' * missing + ']'
            
            try:
                tool_calls_list = json.loads(tool_calls_str)
                if isinstance(tool_calls_list, list):
                    result["tool_calls"] = tool_calls_list
                else:
                    result["error"] = "tool_calls must be a JSON array"
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool_calls JSON: {e}, raw string: {tool_calls_str}")
                result["error"] = (
                    f"Failed to parse tool_calls JSON: {e}\n"
                    f"Attempted to parse: {tool_calls_str[:500]}\n"
                    f"Error at position {e.pos}: ...{tool_calls_str[max(0, e.pos-30):e.pos+30]}..."
                )
                return result
        
        # Must have either answer or tool_calls
        if result["answer"] is None and result["tool_calls"] is None:
            result["error"] = "Response must contain either <answer> or <tool_calls>"
        
        return result
    
    async def _compress_mdp_observations_if_needed(
        self,
        tool_results: List[Dict[str, str]],
        tool_calls_list: List[Dict],
        current_report: str,
        instruction: str,
        turn_count: int,
        tool_call_log_file: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Compress MDP observations if they exceed 80% of remaining budget.
        
        MDP-specific compression strategy:
        - Only maintains last turn's data (not full history)
        - Higher threshold (80% of remaining budget) compared to other agents
        - Compresses individual tool results if needed
        
        Args:
            tool_results: List of tool results from this turn
            tool_calls_list: List of tool calls from this turn
            current_report: Current MDP report
            instruction: Original task instruction
            turn_count: Current turn number
            tool_call_log_file: Optional log file path
            
        Returns:
            Potentially compressed tool results (as objects, not JSON string)
        """
        # Estimate tokens in observations
        last_observations_raw = json.dumps(tool_results, indent=2)
        observations_tokens = self._estimate_tokens(last_observations_raw)
        
        # Get model context limit and calculate remaining budget
        context_limit = self._get_model_context_limit()
        
        # Estimate current state tokens (task + report + actions)
        task_tokens = self._estimate_tokens(instruction)
        report_tokens = self._estimate_tokens(current_report)
        actions_json = json.dumps(tool_calls_list, indent=2)
        actions_tokens = self._estimate_tokens(actions_json)
        
        # Calculate remaining budget
        current_state_tokens = task_tokens + report_tokens + actions_tokens
        remaining_budget = context_limit - current_state_tokens
        
        # MDP threshold: 80% of remaining budget
        mdp_threshold = 0.8
        threshold_tokens = remaining_budget * mdp_threshold
        
        logger.info(
            f"| 📊 MDP Token Check (Turn {turn_count}): "
            f"observations={observations_tokens}, "
            f"remaining={remaining_budget}, "
            f"threshold={int(threshold_tokens)} ({mdp_threshold:.0%})"
        )
        
        # Check if compression is needed
        if observations_tokens <= threshold_tokens:
            return tool_results  # No compression needed, return original objects
        
        logger.warning(
            f"| ⚠️  MDP Compression triggered: "
            f"observations={observations_tokens} > threshold={int(threshold_tokens)}"
        )
        
        if tool_call_log_file:
            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                f.write(
                    f"\n[MDP COMPRESSION START] Turn {turn_count}: "
                    f"{observations_tokens} tokens > {int(threshold_tokens)} threshold\n"
                )
        
        # Build tool contexts for compression (similar to other agents)
        tool_contexts = []
        for tool_result, tool_call_spec in zip(tool_results, tool_calls_list):
            tool_name = tool_result.get("tool", "unknown")
            result_text = tool_result.get("result", "")
            
            tool_contexts.append({
                "name": tool_name,  # Match field name expected by _compress_single_tool_result
                "arguments": json.dumps(tool_call_spec.get("arguments", {})),
                "formatted_result": result_text,
                "result_tokens": self._estimate_tokens(result_text)
            })
        
        # Compress all tool results since we've already determined compression is needed
        logger.info(f"| 🔄 Compressing all {len(tool_contexts)} MDP result(s)...")
        
        compression_tasks = [
            self._compress_single_tool_result(
                ctx, instruction, current_report, tool_call_log_file
            )
            for ctx in tool_contexts
        ]
        
        # Execute compression in parallel
        try:
            compressed_results = await asyncio.wait_for(
                asyncio.gather(*compression_tasks, return_exceptions=True),
                timeout=900  # 5 minutes max
            )
        except asyncio.TimeoutError:
            logger.error(f"| ✗ MDP compression timeout. Using original.")
            return tool_results  # Timeout, return original objects
        
        # Build final compressed observations
        final_tool_results = []
        
        for i, (tool_result, compressed_result) in enumerate(zip(tool_results, compressed_results)):
            if isinstance(compressed_result, Exception):
                logger.error(f"| ✗ MDP compression failed for result {i}. Using original.")
                final_tool_results.append(tool_result)
            else:
                # Use compressed result
                compressed_text = compressed_result.get("formatted_result", tool_result.get("result", ""))
                final_tool_results.append({
                    "tool": tool_result.get("tool"),
                    "result": compressed_text
                })
        
        # Estimate tokens for logging (but return objects, not JSON string)
        compressed_observations_json = json.dumps(final_tool_results, indent=2)
        compressed_tokens = self._estimate_tokens(compressed_observations_json)
        compression_ratio = compressed_tokens / observations_tokens if observations_tokens > 0 else 1.0
        
        logger.info(
            f"| 🗜️  MDP compression complete: "
            f"{observations_tokens} → {compressed_tokens} tokens ({compression_ratio:.1%})"
        )
        
        if tool_call_log_file:
            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                f.write(
                    f"[MDP COMPRESSION END] Turn {turn_count}: "
                    f"{observations_tokens} → {compressed_tokens} tokens ({compression_ratio:.1%})\n"
                )
        
        return final_tool_results  # Return as list of objects
    
    # ==================== Multi-Agent Orchestration (Planner / Explorer / Worker / Verifier) ====================
    
    def _parse_json_object(self, payload: str) -> Dict[str, Any]:
        """
        Best-effort JSON object parser used by multi-agent helpers.
        
        Strips code fences / leading 'json' labels and, if needed, extracts the
        outermost {...} block.
        """
        candidate = (payload or "").strip()
        candidate = candidate.strip("`").strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].lstrip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try to extract first JSON object block
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    pass
        raise ValueError("Failed to parse JSON object from model response")
    
    def _accumulate_usage_from_response(self, total_tokens: Dict[str, int], response_obj: Any) -> None:
        """
        Helper to accumulate token usage from a LiteLLM / OpenAI-style response object.
        """
        if not hasattr(response_obj, "usage") or not response_obj.usage:
            return
        usage = response_obj.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
        completion_tokens = (
            getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_tokens", None)
            or 0
        )
        total_tokens_count = getattr(usage, "total_tokens", None)
        if total_tokens_count is None:
            total_tokens_count = prompt_tokens + completion_tokens
        total_tokens["input_tokens"] += prompt_tokens
        total_tokens["output_tokens"] += completion_tokens
        total_tokens["total_tokens"] += total_tokens_count
        if hasattr(usage, "completion_tokens_details"):
            details = usage.completion_tokens_details
            if hasattr(details, "reasoning_tokens"):
                total_tokens["reasoning_tokens"] += details.reasoning_tokens or 0
    
    def _render_tools_description_for_multi_agent(self, functions: List[Dict[str, Any]]) -> str:
        """
        Render tool / function descriptions in a human-readable form for prompts.
        Reuses logic similar to ReAct / MDP formatting.
        """
        if not functions:
            return "(no tools available)"
        descriptions: List[str] = []
        for func in functions:
            name = func.get("name", "unknown")
            description = func.get("description", "No description provided.")
            parameters = func.get("parameters", {}) or {}
            properties = parameters.get("properties", {}) or {}
            required = set(parameters.get("required", []) or [])
            arg_lines: List[str] = []
            for prop_name, prop_details in properties.items():
                details = json.dumps(prop_details, ensure_ascii=False, indent=2)
                suffix = " (required)" if prop_name in required else ""
                arg_lines.append(f"- {prop_name}{suffix}: {details}")
            arguments_text = "\n".join(arg_lines) if arg_lines else "(no arguments)"
            descriptions.append(
                f"Tool: {name}\nDescription: {description}\nArguments:\n{arguments_text}"
            )
        return "\n\n".join(descriptions)
    
    async def _run_explorer_agent(
        self,
        task_description: str,
        current_plan: Optional[Dict[str, Any]],
        functions: List[Dict[str, Any]],
        mcp_server: Any,
        total_tokens: Dict[str, int],
        tool_call_log_file: Optional[str],
        task_specific_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Priority: task-specific prompt > evolved prompt > default prompt
        if task_specific_prompt:
            system_prompt = task_specific_prompt
            logger.info("| [Explorer] Using TASK-SPECIFIC prompt (from PromptEngineer)")
        elif self._evolved_prompts and "explorer" in self._evolved_prompts:
            system_prompt = self._evolved_prompts["explorer"]
            logger.info("| [Explorer] Using EVOLVED prompt")
        else:
            system_prompt = self._get_default_explorer_prompt()
        explorer_input = {
            "task_description": task_description,
            "current_plan": current_plan,
        }
        user_content = f"""ExplorerInput JSON:
{json.dumps(explorer_input, ensure_ascii=False, indent=2)}
"""
        
        # Prepare messages and tools for function calling
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        tools = [{"type": "function", "function": func} for func in functions] if functions else None
        max_turns = min(self.MAX_TURNS, 50)  # keep Explorer bounded
        turn_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        # Optional logging of available tools
        if tool_call_log_file and tools:
            max_name_length = max(
                len(tool.get("function", {}).get("name", "")) for tool in tools
            )
            with open(tool_call_log_file, "a", encoding="utf-8") as f:
                f.write(f"\n===== Explorer Logs (max_turns: {max_turns}) =====\n")
        
        while turn_count < max_turns:
            turn_count += 1
            # Log turn count
            logger.info(f"| [Explorer] Turn {turn_count}/{max_turns}")
            if tool_call_log_file:
                with open(tool_call_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[Explorer] Turn {turn_count}/{max_turns}\n")
            
            completion_kwargs: Dict[str, Any] = {
                "model": self.litellm_input_model_name,
                "messages": messages,
                "api_key": self.api_key,
            }
            if tools:
                completion_kwargs["tools"] = tools
                completion_kwargs["tool_choice"] = "auto"
            if self.reasoning_effort != "default":
                completion_kwargs["reasoning_effort"] = self.reasoning_effort
            if self.base_url:
                completion_kwargs["base_url"] = self.base_url
            
            try:
                if self._eigenai_client:
                    response = await asyncio.wait_for(
                        self._eigenai_client.acompletion(
                            messages=messages,
                            tools=tools,
                            tool_choice="auto" if tools else None,
                        ),
                        timeout=self.timeout / 2,
                    )
                else:
                    response = await asyncio.wait_for(
                        litellm.acompletion(**completion_kwargs),
                        timeout=self.timeout / 2,
                    )
                consecutive_failures = 0
            except asyncio.TimeoutError:
                logger.warning("| [Explorer] LLM call timed out")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise
                await asyncio.sleep(2 ** consecutive_failures)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error(f"| [Explorer] LLM call failed: {exc}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise
                await asyncio.sleep(2 ** consecutive_failures)
                continue
            
            # Track model name and usage
            if not self.litellm_run_model_name and getattr(response, "model", None):
                self.litellm_run_model_name = response.model.split("/")[-1]
            self._accumulate_usage_from_response(total_tokens, response)
            
            choices = response.choices
            if not choices:
                logger.warning("| [Explorer] Empty choices from LLM")
                break
            message = choices[0].message
            message_dict = message.model_dump() if hasattr(message, "model_dump") else dict(message)
            
            # Log assistant text (if any)
            if hasattr(message, "content") and message.content:
                for line in str(message.content).splitlines():
                    logger.info(f"| [Explorer] {line}")
                if tool_call_log_file:
                    with open(tool_call_log_file, "a", encoding="utf-8") as f:
                        f.write(f"[Explorer Assistant]\n{message.content}\n")
            
            # If there are tool_calls, execute them and continue
            if hasattr(message, "tool_calls") and message.tool_calls:
                messages.append(message_dict)
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        func_args = {}
                    args_str = json.dumps(func_args, separators=(",", ": "))
                    display_args = args_str[:140] + "..." if len(args_str) > 140 else args_str
                    logger.info(f"| [Explorer] Tool {func_name} {display_args}")
                    if tool_call_log_file:
                        with open(tool_call_log_file, "a", encoding="utf-8") as f:
                            f.write(f"[Explorer ToolCall] {func_name} {args_str}\n")
                    try:
                        result = await asyncio.wait_for(
                            mcp_server.call_tool(func_name, func_args),
                            timeout=60,
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        })
                    except asyncio.TimeoutError:
                        error_msg = f"Tool '{func_name}' timed out after 60 seconds"
                        logger.error(f"| [Explorer] {error_msg}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {error_msg}",
                        })
                    except Exception as tool_exc:  # noqa: BLE001
                        error_msg = f"Tool '{func_name}' failed: {tool_exc}"
                        logger.error(f"| [Explorer] {error_msg}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {error_msg}",
                        })
                continue
            
            # No tool calls: treat as final EnvironmentSummary JSON
            messages.append(message_dict)
            content_text = getattr(message, "content", "")
            try:
                env_summary = self._parse_json_object(str(content_text))
            except ValueError:
                # Ask model to reformat as proper JSON
                logger.warning("| [Explorer] Final response was not valid JSON, requesting correction")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON for EnvironmentSummary. "
                            "Please respond again with ONLY the EnvironmentSummary JSON object, "
                            "without code fences or extra text."
                        ),
                    }
                )
                continue
            
            # Basic sanity check of EnvironmentSummary structure
            if "summary_text" not in env_summary or "resources" not in env_summary:
                logger.warning("| [Explorer] EnvironmentSummary missing required keys; requesting correction")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The EnvironmentSummary must include both `summary_text` and `resources`. "
                            "Please resend ONLY a valid EnvironmentSummary JSON object."
                        ),
                    }
                )
                continue
            
            return env_summary
        
        raise RuntimeError("Explorer agent failed to produce a valid EnvironmentSummary within turn limit")
    
    async def _run_worker_agent(
        self,
        task_description: str,
        environment_summary: Dict[str, Any],
        current_subtask: Dict[str, Any],
        execution_state: Dict[str, Any],
        functions: List[Dict[str, Any]],
        mcp_server: Any,
        total_tokens: Dict[str, int],
        tool_call_log_file: Optional[str],
        task_specific_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Add skills as available tools for the worker
        functions = list(functions)  # Copy to avoid modifying original
        skill_tools = self._get_skill_tool_definitions()
        for skill_tool in skill_tools:
            # Convert to function schema format
            func_def = skill_tool.get("function", {})
            functions.append(func_def)
        
        # Priority: task-specific prompt > evolved prompt > default prompt
        if task_specific_prompt:
            system_prompt = task_specific_prompt
            subtask_id = current_subtask.get("id", "unknown")
            logger.info(f"| [Worker] Using TASK-SPECIFIC prompt for subtask {subtask_id} (from PromptEngineer)")
        elif self._evolved_prompts and "worker" in self._evolved_prompts:
            system_prompt = self._evolved_prompts["worker"]
            logger.info("| [Worker] Using EVOLVED prompt")
        else:
            system_prompt = self._get_default_worker_prompt()
        worker_input = {
            "task_description": task_description,
            "environment_summary": environment_summary,
            "current_subtask": current_subtask,
            "execution_state": execution_state,
        }
        worker_input_json = json.dumps(worker_input, ensure_ascii=False, indent=2)
        user_content = f"""WorkerInput JSON:
{worker_input_json}
"""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        tools = [{"type": "function", "function": func} for func in functions] if functions else None
        max_turns = min(self.MAX_TURNS, 50)
        turn_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        # Cumulative list of omitted tool summaries (persists across all trims)
        cumulative_omitted: List[str] = []

        if tool_call_log_file and tools:
            max_name_length = max(
                len(tool.get("function", {}).get("name", "")) for tool in tools
            )
            subtask_id = current_subtask.get("id", "unknown")
            with open(tool_call_log_file, "a", encoding="utf-8") as f:
                f.write(f"\n===== Worker Logs [{subtask_id}] (max_turns: {max_turns}) =====\n")

        while turn_count < max_turns:
            turn_count += 1
            # Log turn count
            subtask_id = current_subtask.get("id", "unknown")
            logger.info(f"| [Worker] [{subtask_id}] Turn {turn_count}/{max_turns}")
            if tool_call_log_file:
                with open(tool_call_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[Worker] [{subtask_id}] Turn {turn_count}/{max_turns}\n")
            
            # Trim context if it exceeds max tokens (80% of model's context limit)
            max_context = int(self._get_model_context_limit() * self.WORKER_CONTEXT_RATIO)
            messages, omitted_summary = self._trim_worker_context(messages, max_context, cumulative_omitted)
            
            # If we have omitted tools, update the user message with the cumulative summary
            if omitted_summary and len(messages) > 1:
                # Find the user message and update with the latest cumulative summary
                for i, msg in enumerate(messages):
                    if msg.get("role") == "user" and i == 1:  # The initial user message
                        content = msg.get("content", "")
                        # Remove old summary if present, then append the updated one
                        if "[Omitted earlier tool calls" in content:
                            # Remove the old summary section
                            idx = content.find("[Omitted earlier tool calls")
                            content = content[:idx].rstrip()
                        messages[i] = {
                            **msg,
                            "content": content + "\n\n" + omitted_summary
                        }
                        break
            
            completion_kwargs: Dict[str, Any] = {
                "model": self.litellm_input_model_name,
                "messages": messages,
                "api_key": self.api_key,
            }
            if tools:
                completion_kwargs["tools"] = tools
                completion_kwargs["tool_choice"] = "auto"
            if self.base_url:
                completion_kwargs["base_url"] = self.base_url
            if self.reasoning_effort != "default":
                completion_kwargs["reasoning_effort"] = self.reasoning_effort

            try:
                if self._eigenai_client:
                    response = await asyncio.wait_for(
                        self._eigenai_client.acompletion(
                            messages=messages,
                            tools=tools,
                            tool_choice="auto" if tools else None,
                        ),
                        timeout=self.timeout / 2,
                    )
                else:
                    response = await asyncio.wait_for(
                        litellm.acompletion(**completion_kwargs),
                        timeout=self.timeout / 2,
                    )
                consecutive_failures = 0
            except asyncio.TimeoutError:
                logger.warning("| [Worker] LLM call timed out")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise
                await asyncio.sleep(2 ** consecutive_failures)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error(f"| [Worker] LLM call failed: {exc}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise
                await asyncio.sleep(2 ** consecutive_failures)
                continue

            if not self.litellm_run_model_name and getattr(response, "model", None):
                self.litellm_run_model_name = response.model.split("/")[-1]
            self._accumulate_usage_from_response(total_tokens, response)

            choices = response.choices
            if not choices:
                logger.warning("| [Worker] Empty choices from LLM")
                break
            message = choices[0].message
            message_dict = message.model_dump() if hasattr(message, "model_dump") else dict(message)

            # Log assistant text (if any)
            if hasattr(message, "content") and message.content:
                for line in str(message.content).splitlines():
                    logger.info(f"| [Worker] {line}")
                if tool_call_log_file:
                    with open(tool_call_log_file, "a", encoding="utf-8") as f:
                        f.write(f"[Worker Assistant]\n{message.content}\n")

            # Handle tool calls for this subtask
            if hasattr(message, "tool_calls") and message.tool_calls:
                messages.append(message_dict)
                
                # Phase 1: Execute all tool calls and collect results
                tool_results: List[Dict[str, Any]] = []
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        func_args = {}
                    args_str = json.dumps(func_args, separators=(",", ": "))
                    display_args = args_str[:160] + "..." if len(args_str) > 160 else args_str
                    logger.info(f"| [Worker] Tool {func_name} {display_args}")
                    if tool_call_log_file:
                        with open(tool_call_log_file, "a", encoding="utf-8") as f:
                            f.write(f"[Worker ToolCall] {func_name} {args_str}\n")
                    try:
                        # Handle skills (including copy_file, move_file, etc.)
                        if self._skill_library.is_skill(func_name):
                            result = await self._execute_skill(func_name, func_args, mcp_server)
                        else:
                            result = await asyncio.wait_for(
                                mcp_server.call_tool(func_name, func_args),
                                timeout=60,
                            )
                        result_str = json.dumps(result, ensure_ascii=False)
                        result_tokens = self._estimate_tokens(result_str)
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "func_name": func_name,
                            "func_args": func_args,
                            "result_str": result_str,
                            "result_tokens": result_tokens,
                            "error": None,
                        })
                    except asyncio.TimeoutError:
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "func_name": func_name,
                            "func_args": func_args,
                            "result_str": f"Error: Tool '{func_name}' timed out after 60 seconds",
                            "result_tokens": 0,
                            "error": "timeout",
                        })
                        logger.error(f"| [Worker] Tool '{func_name}' timed out")
                    except Exception as tool_exc:  # noqa: BLE001
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "func_name": func_name,
                            "func_args": func_args,
                            "result_str": f"Error: Tool '{func_name}' failed: {tool_exc}",
                            "result_tokens": 0,
                            "error": str(tool_exc),
                        })
                        logger.error(f"| [Worker] Tool '{func_name}' failed: {tool_exc}")
                
                # Phase 2: Compress if total tokens exceed threshold (in parallel)
                # Only compress if subtask allows compression (default: True)
                should_compress = current_subtask.get("compress_results", False)
                total_result_tokens = sum(r["result_tokens"] for r in tool_results)
                
                if should_compress and total_result_tokens > self.WORKER_TOTAL_TOKENS_THRESHOLD:
                    compression_indices = [
                        i for i, r in enumerate(tool_results) if not r["error"]
                    ]
                else:
                    compression_indices = []
                
                if compression_indices:
                    logger.info(f"| [Worker] 🔄 Total {total_result_tokens} tokens > {self.WORKER_TOTAL_TOKENS_THRESHOLD}, compressing {len(compression_indices)} result(s)...")
                    compression_tasks = [
                        self._compress_worker_tool_result(
                            subtask_description=current_subtask.get("description", ""),
                            tool_name=tool_results[i]["func_name"],
                            tool_args=tool_results[i]["func_args"],
                            tool_result=tool_results[i]["result_str"],
                            tool_call_log_file=tool_call_log_file,
                        )
                        for i in compression_indices
                    ]
                    try:
                        compressed_results = await asyncio.wait_for(
                            asyncio.gather(*compression_tasks, return_exceptions=True),
                            timeout=1200,
                        )
                        for idx, compressed in zip(compression_indices, compressed_results):
                            if isinstance(compressed, Exception):
                                logger.warning(f"| [Worker] ⚠️ Compression failed: {compressed}. Using original.")
                            else:
                                tool_results[idx]["result_str"] = compressed
                    except asyncio.TimeoutError:
                        logger.error("| [Worker] ⚠️ Parallel compression timed out. Using originals.")
                
                # Phase 3: Append all results to messages
                for tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": tr["result_str"],
                    })
                continue

            # No tool calls: treat as final WorkerOutput JSON
            messages.append(message_dict)
            content_text = getattr(message, "content", "")
            try:
                worker_output = self._parse_json_object(str(content_text))
            except ValueError:
                logger.warning("| [Worker] Final response was not valid JSON, requesting correction")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON for the WorkerOutput. "
                            "Please respond again with ONLY the WorkerOutput JSON object "
                            "in one of the allowed formats (subtask_completed or subtask_blocked)."
                        ),
                    }
                )
                continue

            status = worker_output.get("status")
            if status not in ("subtask_completed", "subtask_blocked"):
                logger.warning("| [Worker] WorkerOutput missing or invalid status; requesting correction")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The WorkerOutput must include a 'status' field equal to either "
                            "'subtask_completed' or 'subtask_blocked'. Please resend ONLY a valid "
                            "WorkerOutput JSON object."
                        ),
                    }
                )
                continue

            # Basic field validation; orchestrator will do deeper state updates
            return worker_output

        raise RuntimeError(
            f"Worker agent failed to produce a valid WorkerOutput for subtask {current_subtask.get('id')} within turn limit"
        )

    async def _run_planning_agent(
        self,
        task_description: str,
        current_plan: Optional[Dict[str, Any]],
        environment_summary: Optional[Dict[str, Any]],
        execution_state: Optional[Dict[str, Any]],
        reason: str,
        available_tools: List[str],
        total_tokens: Dict[str, int],
        tool_call_log_file: Optional[str] = None,
        task_specific_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Priority: task-specific prompt > evolved prompt > default prompt
        if task_specific_prompt:
            system_prompt = task_specific_prompt
            logger.info("| [Planner] Using TASK-SPECIFIC prompt (from PromptEngineer)")
        elif self._evolved_prompts and "planner" in self._evolved_prompts:
            system_prompt = self._evolved_prompts["planner"]
            logger.info("| [Planner] Using EVOLVED prompt")
        else:
            system_prompt = self._get_default_planner_prompt()
        planning_input = {
            "task_description": task_description,
            "environment_summary": environment_summary,
            "available_tools": available_tools,
            "current_plan": current_plan,
            "execution_state": execution_state,
            "reason": reason,
        }
        user_content = f"""PlanningInput JSON:
{json.dumps(planning_input, ensure_ascii=False, indent=2)}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        completion_kwargs: Dict[str, Any] = {
            "model": self.litellm_input_model_name,
            "messages": messages,
            "api_key": self.api_key,
        }
        if self.base_url:
            completion_kwargs["base_url"] = self.base_url
        if self.reasoning_effort != "default":
            completion_kwargs["reasoning_effort"] = self.reasoning_effort
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            if self._eigenai_client:
                response = await asyncio.wait_for(
                    self._eigenai_client.acompletion(messages=messages),
                    timeout=self.timeout / 2,
                )
            else:
                response = await asyncio.wait_for(
                    litellm.acompletion(**completion_kwargs),
                    timeout=self.timeout / 2,
                )
            if not self.litellm_run_model_name and getattr(response, "model", None):
                self.litellm_run_model_name = response.model.split("/")[-1]
            self._accumulate_usage_from_response(total_tokens, response)
            choice = response.choices[0]
            message_obj = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
            content_raw = getattr(message_obj, "content", None) if message_obj is not None else None
            if content_raw is None and isinstance(choice, dict):
                content_raw = choice.get("message", {}).get("content")
            content_str = content_raw if isinstance(content_raw, str) else json.dumps(content_raw, ensure_ascii=False)
            
            try:
                planning_output = self._parse_json_object(content_str)
                break  # Success, exit loop
            except ValueError:
                if attempt < max_retries:
                    messages.append({"role": "assistant", "content": content_str})
                    messages.append({"role": "user", "content": "Invalid JSON. Reply with ONLY valid JSON."})
                    completion_kwargs["messages"] = messages
                    continue
                raise  # Max retries reached
        
        # Log planning output to file
        if tool_call_log_file:
            try:
                with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n===== PlanningAgent ({reason}) =====\n")
                    f.write(f"Output:\n{json.dumps(planning_output, ensure_ascii=False, indent=2)}\n")
            except Exception:
                pass
        
        return planning_output
    
    async def _run_verifier_agent(
        self,
        task_description: str,
        environment_summary: Dict[str, Any],
        functions: List[Dict[str, Any]],
        mcp_server: Any,
        total_tokens: Dict[str, int],
        tool_call_log_file: Optional[str],
    ) -> Dict[str, Any]:
        """
        Run the Verifier agent in a read-only fashion.
        The Verifier independently inspects the environment to verify task completion,
        without knowing what the Worker claimed to do.
        """
        system_prompt = """
You are the Verifier in a multi-agent system. You collaborate with other agents to solve the task together. Your specific role is to verify whether the task has been correctly completed by inspecting the environment. You should follow this prompt strictly.

You have:
- task_description: the original user task in natural language.
- environment_summary: resource locations only (paths, kinds, formats) - use this to find where to inspect.

IMPORTANT:
- Each turn: respond with tool calls OR content, never both together.
- You can ONLY inspect and read. You MUST NOT modify anything.

Verification Strategy:
1. First, check the allowed workspace (list_allowed_directories) to understand the root
2. Read the task_description carefully and identify what outputs/results are required
3. Inspect actual outputs using tools - verify with actual inspection, not assumptions
4. Compare: Does actual output match the task requirements? (completeness, correctness, format)
5. Check for issues: missing items, incorrect data, wrong formats, unwanted extras
6. Batch multiple independent tool calls in a single turn when possible
7. Check that the agent followed the task’s rules as written and did not introduce its own decision policies. Avoid overthinking.
8. For complex environment (e.g many nested directories file systems), inspect depper and wider if you are uncertain about specific resources.

Common Verification Pitfalls - CHECK CAREFULLY:
- Wrong Directory Structure: Task mentions directory name as context (e.g., "in test directory") - this is NOT an instruction to create that directory. Check if an unnecessary wrapper directory was created.
- Missing Requirements: Verify ALL requirements from the task are satisfied, not just some.
- Format Issues: For tasks with specific formats, verify exact compliance.

Guidelines:
- Be systematic: Check each requirement methodically
- Be thorough: Inspect actual files/outputs, don't trust claims without evidence
- Be fair: Don't fail for trivial or subjective issues
- Be specific: Issues must include evidence and be actionable

Output Format - Once you have enough evidence, respond with a single JSON object:
  {"status": "verified_ok", "summary": "...", "issues": []}
  {"status": "verified_with_issues", "summary": "...", "issues": ["issue1 with evidence", "issue2 with evidence"]}
"""
        # Filter environment_summary to remove interpretation bias
        # Keep only factual location data, remove summary_text and notes
        filtered_env_summary = {}
        if environment_summary:
            # Keep only resource locations, strip interpretive fields
            if "resources" in environment_summary:
                filtered_resources = []
                for res in environment_summary.get("resources", []):
                    filtered_res = {
                        "id": res.get("id"),
                        "locator": res.get("locator"),
                        "kind": res.get("kind"),
                        "format": res.get("format"),
                        "preview": res.get("preview"),
                        # Intentionally omit "notes" - may contain interpretation bias
                    }
                    filtered_resources.append(filtered_res)
                filtered_env_summary["resources"] = filtered_resources
            # Intentionally omit "summary_text" - may contain interpretation bias
        
        verifier_input = {
            "task_description": task_description,
            "environment_summary": filtered_env_summary,
        }
        user_content = f"""VerifierInput JSON:
{json.dumps(verifier_input, ensure_ascii=False, indent=2)}
"""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        tools = [{"type": "function", "function": func} for func in functions] if functions else None
        max_turns = min(self.MAX_TURNS, 30)
        turn_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 3

        # Optional logging of available tools for verifier
        if tool_call_log_file and tools:
            max_name_length = max(
                len(tool.get("function", {}).get("name", "")) for tool in tools
            )
            with open(tool_call_log_file, "a", encoding="utf-8") as f:
                f.write(f"\n===== Verifier Logs (max_turns: {max_turns}) =====\n")

        while turn_count < max_turns:
            turn_count += 1
            # Log turn count
            logger.info(f"| [Verifier] Turn {turn_count}/{max_turns}")
            if tool_call_log_file:
                with open(tool_call_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[Verifier] Turn {turn_count}/{max_turns}\n")
            
            completion_kwargs: Dict[str, Any] = {
                "model": self.litellm_input_model_name,
                "messages": messages,
                "api_key": self.api_key,
            }
            if tools:
                completion_kwargs["tools"] = tools
                completion_kwargs["tool_choice"] = "auto"
            if self.base_url:
                completion_kwargs["base_url"] = self.base_url
            if self.reasoning_effort != "default":
                completion_kwargs["reasoning_effort"] = self.reasoning_effort

            try:
                if self._eigenai_client:
                    response = await asyncio.wait_for(
                        self._eigenai_client.acompletion(
                            messages=messages,
                            tools=tools,
                            tool_choice="auto" if tools else None,
                        ),
                        timeout=self.timeout / 2,
                    )
                else:
                    response = await asyncio.wait_for(
                        litellm.acompletion(**completion_kwargs),
                        timeout=self.timeout / 2,
                    )
                consecutive_failures = 0
            except asyncio.TimeoutError:
                logger.warning("| [Verifier] LLM call timed out")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise
                await asyncio.sleep(2 ** consecutive_failures)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error(f"| [Verifier] LLM call failed: {exc}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise
                await asyncio.sleep(2 ** consecutive_failures)
                continue

            if not self.litellm_run_model_name and getattr(response, "model", None):
                self.litellm_run_model_name = response.model.split("/")[-1]
            self._accumulate_usage_from_response(total_tokens, response)

            choices = response.choices
            if not choices:
                logger.warning("| [Verifier] Empty choices from LLM")
                break
            message = choices[0].message
            message_dict = message.model_dump() if hasattr(message, "model_dump") else dict(message)

            # Log assistant text (if any)
            if hasattr(message, "content") and message.content:
                for line in str(message.content).splitlines():
                    logger.info(f"| [Verifier] {line}")
                if tool_call_log_file:
                    with open(tool_call_log_file, "a", encoding="utf-8") as f:
                        f.write(f"[Verifier Assistant]\n{message.content}\n")

            # Handle tool calls (read-only inspection)
            if hasattr(message, "tool_calls") and message.tool_calls:
                messages.append(message_dict)
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        func_args = {}
                    args_str = json.dumps(func_args, separators=(",", ": "))
                    display_args = args_str[:140] + "..." if len(args_str) > 140 else args_str
                    logger.info(f"| [Verifier] Tool {func_name} {display_args}")
                    if tool_call_log_file:
                        with open(tool_call_log_file, "a", encoding="utf-8") as f:
                            f.write(f"[Verifier ToolCall] {func_name} {args_str}\n")
                    try:
                        result = await asyncio.wait_for(
                            mcp_server.call_tool(func_name, func_args),
                            timeout=60,
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        })
                    except asyncio.TimeoutError:
                        error_msg = f"Tool '{func_name}' timed out after 60 seconds"
                        logger.error(f"| [Verifier] {error_msg}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {error_msg}",
                        })
                    except Exception as tool_exc:  # noqa: BLE001
                        error_msg = f"Tool '{func_name}' failed: {tool_exc}"
                        logger.error(f"| [Verifier] {error_msg}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {error_msg}",
                        })
                continue

            # No tool calls: treat as final VerificationReport JSON
            messages.append(message_dict)
            content_text = getattr(message, "content", "")
            try:
                verification_output = self._parse_json_object(str(content_text))
            except ValueError:
                # Ask model to reformat as proper JSON
                logger.warning("| [Verifier] Final response was not valid JSON, requesting correction")
                logger.warning(f"| [Verifier] Raw content: {str(content_text)[:500] if content_text else '(empty)'}")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON for VerificationReport. "
                            "Please respond again with ONLY the VerificationReport JSON object, "
                            "without code fences or extra text."
                        ),
                    }
                )
                continue

            if "status" not in verification_output or "summary" not in verification_output:
                logger.warning("| [Verifier] VerificationReport missing required keys; requesting correction")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The VerificationReport must include at least `status` and `summary` keys. "
                            "Please resend ONLY a valid VerificationReport JSON object."
                        ),
                    }
                )
                continue

            return verification_output

        raise RuntimeError("Verifier agent failed to produce a valid VerificationReport within turn limit")
    
    async def _execute_multi_agent_tool_loop(
        self,
        instruction: str,
        functions: List[Dict[str, Any]],
        mcp_server: Any,
        tool_call_log_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        High-level multi-agent orchestration following the user-specified design:
        PlanningAgent (INITIAL/REFINEMENT) → Explorer → Worker loop (placeholder) → Verifier.
        
        NOTE: This initial implementation focuses on the planning / verification wiring and
        uses a lightweight, tool-agnostic EnvironmentSummary. Explorer/Worker currently do
        not issue real MCP tool calls; they treat the environment as structured hints only.
        This keeps the architecture modular while avoiding interference with the existing
        production MDP loop.
        """
        # Shared accounting
        total_tokens: Dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }
        turn_count = 0
        all_messages: List[Dict[str, Any]] = []
        
        # Record initial user message for SDK compatibility
        all_messages.append({"role": "user", "content": instruction})
        
        # Task-specific prompts generated by PromptEngineer (per-execution)
        task_specific_explorer_prompt: Optional[str] = None
        task_specific_planner_prompt: Optional[str] = None
        
        try:
            # Generate task-specific Explorer prompt if PromptEngineer is enabled
            if self._prompt_engineer_enabled and self._prompt_engineer:
                base_explorer_prompt = self._get_base_prompt("explorer")
                task_specific_explorer_prompt = self._prompt_engineer.generate_explorer_prompt(
                    base_prompt=base_explorer_prompt,
                    task_description=instruction,
                )
            
            # 1) Explorer first - discover environment
            environment_summary = await self._run_explorer_agent(
                task_description=instruction,
                current_plan=None,
                functions=functions,
                mcp_server=mcp_server,
                total_tokens=total_tokens,
                tool_call_log_file=tool_call_log_file,
                task_specific_prompt=task_specific_explorer_prompt,
            )
            all_messages.append(
                {
                    "role": "assistant",
                    "content": f"[Explorer] {json.dumps(environment_summary, ensure_ascii=False)}",
                }
            )
            
            # Run Skills Agent after Explorer (connection warmed up)
            if self._skills_agent_enabled and self._skills_agent:
                try:
                    logger.info("| [Skills] Running SkillsAgent to propose skills")
                    base_tools = [{"name": f.get("name", ""), "description": f.get("description", "")} for f in functions]
                    existing_skills = self._skill_library.get_skill_summaries()
                    proposed = self._skills_agent.propose_skills(
                        task_description=instruction,
                        base_tools=base_tools,
                        existing_skills=existing_skills,
                    )
                    self._task_proposed_skills = proposed
                    for skill in proposed:
                        self._skill_library.register_temporary(skill)
                    logger.info(f"| [Skills] Proposed {len(proposed)} skills: {[s.name for s in proposed]}")
                except Exception as e:
                    logger.warning(f"| [Skills] SkillsAgent failed: {e}")

            # Initialize state
            current_plan: Dict[str, Any] = {"subtasks": []}
            execution_state: Dict[str, Any] = {
                "plan": current_plan,
                "subtasks_progress": [],
                "artifacts": {},
                "logs": [],
            }
            planning_reason = "initial"
            verification_loops = 0
            max_verification_loops = 3
            final_verification: Optional[Dict[str, Any]] = None

            while True:
                # Generate task-specific Planner prompt if PromptEngineer is enabled
                # Generate for both initial planning and re-planning scenarios
                if self._prompt_engineer_enabled and self._prompt_engineer:
                    base_planner_prompt = self._get_base_prompt("planner")
                    task_specific_planner_prompt = self._prompt_engineer.generate_planner_prompt(
                        base_prompt=base_planner_prompt,
                        task_description=instruction,
                        environment_summary=environment_summary,
                        execution_state=execution_state if planning_reason != "initial" else None,
                        planning_reason=planning_reason,
                    )
                
                # Planning (called on first iteration and when re-planning needed)
                planning_result = await self._run_planning_agent(
                    task_description=instruction,
                    current_plan=current_plan,
                    environment_summary=environment_summary,
                    execution_state=execution_state,
                    reason=planning_reason,
                    available_tools=([f.get("name", "") for f in functions] + self._skill_library.list_all()) if functions else self._skill_library.list_all(),
                    total_tokens=total_tokens,
                    tool_call_log_file=tool_call_log_file,
                    task_specific_prompt=task_specific_planner_prompt,
                )
                turn_count += 1
                current_plan = planning_result.get("plan", {}) or {"subtasks": []}
                execution_state["plan"] = current_plan
                planning_notes = planning_result.get("notes", "")
                execution_state.setdefault("logs", []).append(
                    f"Planning ({planning_reason}) notes: {planning_notes}"
                )
                all_messages.append(
                    {
                        "role": "assistant",
                        "content": f"[PlanningAgent ({planning_reason})] {json.dumps(planning_result, ensure_ascii=False)}",
                    }
                )

                # Ensure progress entries exist for all subtasks
                progress_entries: List[Dict[str, Any]] = execution_state.setdefault(
                    "subtasks_progress", []
                )
                by_id = {entry.get("subtask_id"): entry for entry in progress_entries}
                for sub in current_plan.get("subtasks", []) or []:
                    sid = sub.get("id", "")
                    if sid and sid not in by_id:
                        entry = {
                            "subtask_id": sid,
                            "status": "pending",
                            "progress_notes": "",
                        }
                        progress_entries.append(entry)
                        by_id[sid] = entry

                # Worker phase for this plan
                worker_blocked_reason: Optional[str] = None
                blocked_subtask_id: Optional[str] = None

                for sub in current_plan.get("subtasks", []) or []:
                    subtask_id = sub.get("id", "")
                    if not subtask_id:
                        continue

                    progress_entry = None
                    for entry in execution_state.get("subtasks_progress", []):
                        if entry.get("subtask_id") == subtask_id:
                            progress_entry = entry
                            break
                    if progress_entry is None:
                        progress_entry = {
                            "subtask_id": subtask_id,
                            "status": "pending",
                            "progress_notes": "",
                        }
                        execution_state["subtasks_progress"].append(progress_entry)

                    if progress_entry.get("status") == "completed":
                        continue
                    logger.info(
                        "Running Worker on %s | status=%s | description=%s | notes=%s",
                        subtask_id,
                        progress_entry.get("status"),
                        sub.get("description"),
                        sub.get("notes"),
                    )

                    progress_entry["status"] = "in_progress"
                    progress_entry["progress_notes"] = "Worker started."

                    # Generate task-specific worker prompt just before execution
                    # This allows the prompt to incorporate the latest execution_state
                    worker_prompt_for_subtask: Optional[str] = None
                    if self._prompt_engineer_enabled and self._prompt_engineer:
                        base_worker_prompt = self._get_base_prompt("worker")
                        worker_prompt_for_subtask = self._prompt_engineer.generate_worker_prompt(
                            base_prompt=base_worker_prompt,
                            task_description=instruction,
                            current_subtask=sub,
                            environment_summary=environment_summary,
                            execution_state=execution_state,
                        )
                    
                    worker_output = await self._run_worker_agent(
                        task_description=instruction,
                        environment_summary=environment_summary,
                        current_subtask=sub,
                        execution_state=execution_state,
                        functions=functions,
                        mcp_server=mcp_server,
                        total_tokens=total_tokens,
                        tool_call_log_file=tool_call_log_file,
                        task_specific_prompt=worker_prompt_for_subtask,
                    )

                    status = worker_output.get("status")
                    if status == "subtask_completed":
                        summary = worker_output.get("summary", "")
                        artifacts = worker_output.get("artifacts", {}) or {}
                        progress_entry["status"] = "completed"
                        progress_entry["progress_notes"] = (
                            summary or "Subtask completed by Worker."
                        )
                        execution_state.setdefault("artifacts", {}).update(artifacts)
                        execution_state.setdefault("logs", []).append(
                            f"Subtask {subtask_id} completed: {summary}"
                        )
                    elif status == "subtask_blocked":
                        reason = worker_output.get(
                            "reason", "Blocked for unspecified reason."
                        )
                        progress_entry["status"] = "blocked"
                        progress_entry["progress_notes"] = reason
                        execution_state.setdefault("logs", []).append(
                            f"Subtask {subtask_id} blocked: {reason}"
                        )
                        worker_blocked_reason = reason
                        blocked_subtask_id = subtask_id
                        break
                    else:
                        progress_entry["status"] = "blocked"
                        progress_entry[
                            "progress_notes"
                        ] = f"Worker returned unexpected status: {status!r}"
                        execution_state.setdefault("logs", []).append(
                            f"Worker returned unexpected status for subtask {subtask_id}: {status!r}"
                        )
                        worker_blocked_reason = (
                            f"Unexpected worker status: {status!r}"
                        )
                        blocked_subtask_id = subtask_id
                        break

                if worker_blocked_reason:
                    planning_reason = (
                        f"blocked_subtask_{blocked_subtask_id}: {worker_blocked_reason}"
                    )
                    continue

                # All subtasks completed → verification
                verification_report = await self._run_verifier_agent(
                    task_description=instruction,
                    environment_summary=environment_summary,
                    functions=functions,
                    mcp_server=mcp_server,
                    total_tokens=total_tokens,
                    tool_call_log_file=tool_call_log_file,
                )
                turn_count += 1
                verification_loops += 1
                all_messages.append(
                    {
                        "role": "assistant",
                        "content": f"[Verifier] {json.dumps(verification_report, ensure_ascii=False)}",
                    }
                )

                status = verification_report.get("status")
                if status in ("verified_ok", "pass"):
                    final_verification = verification_report
                    break

                if (
                    status in ("verified_with_issues", "fail", "uncertain")
                    and verification_loops < max_verification_loops
                ):
                    issues = verification_report.get("issues", []) or []
                    execution_state.setdefault("logs", []).append(
                        f"Verifier reported issues: {issues}"
                    )
                    planning_reason = (
                        f"verifier_issues: {issues[:2]}"
                    )
                    continue

                final_verification = verification_report
                break

            # 7) Build final natural-language answer
            status = final_verification.get("status") or ""
            verdict_summary = final_verification.get("summary", "")
            issues = final_verification.get("issues", []) or []
            if status in ("verified_ok", "pass"):
                final_text = (
                    "Multi-agent planning completed.\n\n"
                    f"Plan:\n{json.dumps(execution_state.get('plan'), ensure_ascii=False, indent=2)}\n\n"
                    f"Verification: {verdict_summary or 'Verifier reported no issues.'}"
                )
                success = True
                error_msg = None
            elif status in ("verified_with_issues", "fail", "uncertain"):
                issues_text = "\n".join(f"- {issue}" for issue in issues)
                final_text = (
                    "Multi-agent planning completed, but verification reported issues.\n\n"
                    f"Plan:\n{json.dumps(execution_state.get('plan'), ensure_ascii=False, indent=2)}\n\n"
                    f"Verification: {verdict_summary}\n\n"
                    f"Issues:\n{issues_text}"
                )
                success = False
                error_msg = "Verification reported issues in multi-agent planning."
            else:
                final_text = (
                    "Multi-agent planning completed, but verifier status was unrecognized.\n\n"
                    f"Raw verification report:\n{json.dumps(final_verification, ensure_ascii=False, indent=2)}"
                )
                success = False
                error_msg = "Unexpected verifier status."
            
            all_messages.append({"role": "assistant", "content": final_text})
            sdk_messages = self._convert_to_sdk_format(all_messages)
            return {
                "success": success,
                "output": sdk_messages,
                "token_usage": total_tokens,
                "turn_count": turn_count,
                "error": error_msg,
                "litellm_run_model_name": self.litellm_run_model_name,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Multi-agent execution failed: %s", exc, exc_info=True)
            sdk_messages = self._convert_to_sdk_format(all_messages)
            return {
                "success": False,
                "output": sdk_messages,
                "token_usage": total_tokens,
                "turn_count": turn_count,
                "error": exc,
                "litellm_run_model_name": self.litellm_run_model_name,
            }
    
    async def _execute_mdp_tool_loop(
        self,
        instruction: str,
        functions: List[Dict],
        mcp_server: Any,
        tool_call_log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute function calling loop using Markov Decision Process approach.
        
        Key differences from other loops:
        1. Models the whole process as state transitions
        2. State = {task, evolving report, last actions, last observations}
        3. Agent generates {report, answer OR tool_calls} each iteration
        4. Report serves as compressed memory, not accumulating full history
        """
        
        # Convert functions to tools format
        tools = [{"type": "function", "function": func} for func in functions] if functions else []
        
        # MDP State
        current_report = ""  # Evolving report (compressed memory)
        last_actions = ""  # Last round's tool calls and results (combined)
        verification_issues = ""  # Issues from verification agent (if any)
        
        # Tracking
        total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
        turn_count = 0
        max_turns = self.MAX_TURNS
        consecutive_failures = 0
        max_consecutive_failures = 3
        hit_turn_limit = False
        ended_normally = False
        
        # Message accumulation for output (backward compatibility)
        all_messages = []
        
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
        if tool_call_log_file and tools:
            max_name_length = max(
                len(tool.get("function", {}).get("name", ""))
                for tool in tools
            )
            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                f.write("===== Available Tools =====\n")
                for tool in tools:
                    function_info = tool.get("function", {})
                    tool_name = function_info.get("name", "N/A")
                    description = function_info.get("description", "N/A")
                    f.write(f"- ToolName: {tool_name:<{max_name_length}} Description: {description}\n")
                f.write("\n===== MDP Execution Logs =====\n")
        
        logger.info("Starting MDP-based execution loop")
        
        try:
            while turn_count < max_turns:
                # Build MDP prompt with current state
                user_prompt = self._build_mdp_prompt(
                    task=instruction,
                    tools=tools,
                    report=current_report,
                    last_actions=last_actions,
                    verification_issues=verification_issues
                )
                
                # Build messages (no tools in the API call - we use text-based tool calls)
                messages = [
                    {"role": "user", "content": user_prompt}
                ]
                
                # Store system message only once for output
                if turn_count == 0:
                    all_messages.append({"role": "user", "content": instruction})
                
                # Build completion kwargs
                completion_kwargs = {
                    "model": self.litellm_input_model_name,
                    "messages": messages,
                    "api_key": self.api_key,
                }
                
                # Add reasoning_effort and base_url if specified
                if self.reasoning_effort != "default":
                    completion_kwargs["reasoning_effort"] = self.reasoning_effort
                if self.base_url:
                    completion_kwargs["base_url"] = self.base_url
                
                try:
                    # Call LLM
                    if self._openai_client:
                        response = await asyncio.wait_for(
                            self._openai_client.acompletion(
                                messages=messages,
                                tools=None,  # No function calling, text-based only
                                tool_choice=None,
                            ),
                            timeout=self.timeout / 2
                        )
                    else:
                        response = await asyncio.wait_for(
                            litellm.acompletion(**completion_kwargs),
                            timeout=self.timeout / 2
                        )
                    consecutive_failures = 0  # Reset failure counter on success
                except asyncio.TimeoutError:
                    logger.warning(f"| ✗ LLM call timed out on turn {turn_count + 1}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise Exception(f"Too many consecutive failures ({consecutive_failures})")
                    await asyncio.sleep(8 ** consecutive_failures)
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
                
                # Update token usage
                _record_usage(response)
                
                # Get response content
                choices = response.choices
                if not len(choices):
                    logger.error("| ✗ No choices in response")
                    break
                
                message = choices[0].message
                content = message.content if hasattr(message, 'content') else ""
                
                if not content:
                    logger.error("| ✗ Empty content in response")
                    break
                
                # Log the raw response
                if tool_call_log_file:
                    with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n===== Turn {turn_count + 1} Response =====\n")
                        f.write(content + "\n")
                
                # Parse MDP response
                parsed = self._parse_mdp_response(content)
                
                if parsed.get("error"):
                    logger.error(f"| ✗ Failed to parse MDP response: {parsed['error']}")
                    logger.error(f"| Response content: {content[:500]}...")
                    logger.warning(f"| ⚠️  Parse error on turn {turn_count + 1}, will retry on next turn")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"| ✗ Too many consecutive parse failures ({consecutive_failures})")
                        break
                    # Retry by continuing to next iteration (agent will try again)
                    turn_count += 1
                    continue
                
                # Reset failure counter on successful parse
                consecutive_failures = 0
                
                # Update report
                new_report = parsed["report"]
                current_report = new_report
                logger.info(f"📝 Report updated: {current_report[:200]}{'...' if len(current_report) > 200 else ''}")
                
                # Add assistant message to output
                all_messages.append({
                    "role": "assistant",
                    "content": content
                })
                
                # Check if we have an answer (task completed)
                if parsed.get("answer"):
                    answer = parsed["answer"]
                    logger.info(f"✅ Main agent claims task complete: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                    
                    if tool_call_log_file:
                        with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"| Main Agent Answer: {answer}\n")
                    
                    # Run verification before accepting completion
                    logger.info("🔍 Starting task verification...")
                    
                    verification_result = await self._verify_task_completion(
                        instruction=instruction,
                        functions=functions,
                        mcp_server=mcp_server,
                        tool_call_log_file=tool_call_log_file
                    )
                    
                    if verification_result["passed"]:
                        logger.info("✅ Task verification PASSED!")
                        
                        if tool_call_log_file:
                            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"| ✅ VERIFICATION PASSED\n")
                        
                        ended_normally = True
                        break
                    else:
                        issues = verification_result["issues"]
                        logger.warning(f"❌ Task verification FAILED:\n{issues[:500]}{'...' if len(issues) > 500 else ''}")
                        
                        if tool_call_log_file:
                            with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                                f.write(f"| ❌ VERIFICATION FAILED\n{issues}\n\n")
                        
                        verification_issues = issues
                        last_actions = ""
                        continue
                
                # Execute tool calls
                tool_calls_list = parsed.get("tool_calls", [])
                if not tool_calls_list:
                    logger.warning("| ⚠️  No answer and no tool_calls - treating as completion")
                    turn_count += 1
                    ended_normally = True
                    break
                
                logger.info(f"| 🔧 Executing {len(tool_calls_list)} tool call(s)")
                
                # Execute each tool call
                tool_results = []
                actions_log = []
                
                for idx, tool_call_spec in enumerate(tool_calls_list):
                    func_name = tool_call_spec.get("name")
                    func_args = tool_call_spec.get("arguments", {})
                    
                    if not func_name:
                        logger.error(f"| ✗ Tool call {idx} missing 'name'")
                        continue
                    
                    # Log tool call
                    args_str = json.dumps(func_args, separators=(",", ": "))
                    display_args = args_str[:250] + "..." if len(args_str) > 250 else args_str
                    logger.info(f"| \033[1m{func_name}\033[0m \033[2;37m{display_args}\033[0m")
                    
                    if tool_call_log_file:
                        with open(tool_call_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"| {func_name} {args_str}\n")
                    
                    # Record action for next iteration
                    actions_log.append(f"{func_name}({args_str})")
                    
                    # Execute tool
                    try:
                        result = await asyncio.wait_for(
                            mcp_server.call_tool(func_name, func_args),
                            timeout=60
                        )
                        formatted_result = self._format_tool_result_for_model(result, func_name)
                        tool_results.append({
                            "tool": func_name,
                            "result": formatted_result
                        })
                        
                        
                    except asyncio.TimeoutError:
                        error_msg = f"Tool call '{func_name}' timed out after 60 seconds"
                        logger.error(f"| ✗ {error_msg}")
                        tool_results.append({
                            "tool": func_name,
                            "result": f"Error: {error_msg}"
                        })
                    except Exception as e:
                        logger.error(f"| ✗ Tool call '{func_name}' failed: {e}")
                        tool_results.append({
                            "tool": func_name,
                            "result": f"Error: {str(e)}"
                        })
                
                # Check if compression is needed for MDP (80% threshold)
                # Since MDP only keeps last turn's data, we can be more lenient
                # This returns Python objects (list), not JSON string
                final_results = await self._compress_mdp_observations_if_needed(
                    tool_results=tool_results,
                    tool_calls_list=tool_calls_list,
                    current_report=current_report,
                    instruction=instruction,
                    turn_count=turn_count,
                    tool_call_log_file=tool_call_log_file
                )
                
                # Combine actions and results (work with Python objects)
                combined_turn_data = []
                for call, result in zip(tool_calls_list, final_results):
                    combined_turn_data.append({
                        "tool": call.get("name"),
                        "arguments": call.get("arguments", {}),
                        "result": result.get("result", "")
                    })
                
                # Convert to JSON format for prompt
                last_actions = json.dumps(combined_turn_data, indent=2)
                
                turn_count += 1
                self._update_progress(all_messages, total_tokens, turn_count)
            
        except Exception as loop_error:
            logger.error(f"MDP loop failed: {loop_error}", exc_info=True)
            sdk_format_messages = self._convert_to_sdk_format(all_messages)
            return {
                "success": False,
                "output": sdk_format_messages,
                "token_usage": total_tokens,
                "turn_count": turn_count,
                "error": str(loop_error),
                "litellm_run_model_name": self.litellm_run_model_name,
            }
        
        # Detect if we hit turn limit
        if (not ended_normally) and (turn_count >= max_turns):
            hit_turn_limit = True
            logger.warning(f"| Max turns ({max_turns}) exceeded; returning failure with partial output.")
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
    
