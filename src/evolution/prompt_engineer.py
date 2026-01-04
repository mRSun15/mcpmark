"""
Prompt Engineer Agent
======================

Generates task-specific prompts for Explorer, Planner, and Worker agents.
Unlike prompt evolution (which creates general improved prompts from feedback),
this generates prompts tailored to the specific task at hand.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

import litellm

from src.logger import get_logger

logger = get_logger(__name__)


class PromptEngineerAgent:
    """
    Generates task-specific prompts for each agent role.
    
    Can optionally use feedback from previous task executions to improve prompts.
    
    Modes:
    - Without feedback: base_prompt + task context → task-specific prompt
    - With feedback: base_prompt + task context + feedback → improved task-specific prompt
    """
    
    # Unified system prompts (feedback is optional)
    EXPLORER_SYSTEM_PROMPT = """You are a Prompt Engineer. Synthesize a task-specific prompt for the Explorer agent.

Input:
- base_prompt: The general Explorer prompt
- task_description: The specific task
- feedback (optional): Lessons from previous runs

Output (JSON only):
{"task_specific_prompt": "..."}

WHAT TO PRESERVE (mandatory - copy these exactly):
1. Role identity: "You are the Explorer" and its high-level objective (inspect environment, produce concise summary)
2. Output schema: EnvironmentSummary JSON with summary_text and resources[] array
   - Each resource: {id, locator, kind, format, preview, notes}
3. Interaction pattern: "Call tools to inspect... When done, respond with a single EnvironmentSummary JSON"
4. Core constraint: READ-ONLY (no writes, no modifications)

WHAT TO SYNTHESIZE (rewrite for this task):
- Guidelines and goals: Make them task-specific instead of generic
- Priorities: What to look for, what matters for this task
- Remove generic advice that doesn't apply
- If feedback provided: incorporate 1-2 key insights naturally

QUALITY PRINCIPLES:
- Focus on WHAT to achieve, not step-by-step HOW
- Don't duplicate data that will be in environment_summary
- Reference resources by property (e.g., "files needing updates") not by ID

BREVITY: Keep prompts concise (200-400 words). Use bullets, not paragraphs. No filler phrases.

Output ONLY valid JSON."""

    PLANNER_SYSTEM_PROMPT = """You are a Prompt Engineer. Synthesize a task-specific prompt for the Planner agent.

Input:
- base_prompt: The general Planner prompt
- task_description: The specific task
- environment_summary: Discovered resources
- execution_state: Current progress (null if initial)
- planning_reason: "initial" or reason for re-planning
- feedback (optional): Lessons from previous runs

Output (JSON only):
{"task_specific_prompt": "..."}

WHAT TO PRESERVE (mandatory - copy these exactly):
1. Role identity: "You are the PlanningAgent" and its objective (produce ordered subtasks)
2. Output schema: {"plan": {"subtasks": [{"id", "description", "notes", "compress_results"}]}, "notes": "..."}
3. Core rules: 
   - Follow task description strictly
   - Use stable subtask IDs (S1, S2, S3...)
   - Reference artifacts between subtasks
4. Planner outputs ONLY plan JSON (no tool calls)
5. SUBTASK GRANULARITY: Keep plans coarse (≤20 subtasks). Each subtask = logical phase, not individual operation. 

WHAT TO SYNTHESIZE (rewrite for this task):
- Planning approach: What strategy makes sense for this task type
- Subtask guidance: What logical phases does this task have?
- If re-planning: focus on what blocked and how to unblock
- If feedback provided: incorporate relevant planning insights

QUALITY PRINCIPLES:
- Focus on WHAT to achieve, not step-by-step HOW
- Don't embed specific resource IDs; reference by property
- Don't duplicate environment_summary data

BREVITY: Keep prompts concise (300-500 words). Use bullets, not paragraphs. No filler phrases.

Output ONLY valid JSON."""

    WORKER_SYSTEM_PROMPT = """You are a Prompt Engineer. Synthesize a subtask-specific prompt for the Worker agent.

Input:
- base_prompt: The general Worker prompt
- task_description: The original task
- current_subtask: The subtask to execute (id, description, notes)
- environment_summary: Discovered resources
- execution_state: Progress and artifacts from prior subtasks
- feedback (optional): Lessons from previous runs

Output (JSON only):
{"task_specific_prompt": "..."}

WHAT TO PRESERVE (mandatory - copy these exactly):
1. Role identity: "You are the Worker" and objective (execute single subtask)
2. Output schemas:
   - subtask_completed: {"status": "subtask_completed", "subtask_id", "summary", "artifacts": {...}}
   - subtask_blocked: {"status": "subtask_blocked", "subtask_id", "reason"}
3. Interaction pattern: "Call tools OR respond with JSON (not both in same turn)"
4. Core rules:
   - Focus on current subtask only
   - Bundle related tool calls together
   - Use artifacts from prior subtasks

WHAT TO SYNTHESIZE (rewrite for this subtask):
- Subtask-specific guidance: What this subtask needs to accomplish
- Relevant context: Which prior artifacts matter, what to use
- Remove generic advice that doesn't apply to this subtask
- If feedback provided: incorporate relevant execution insights

QUALITY PRINCIPLES:
- Focus on WHAT to accomplish, not detailed HOW
- Don't embed resource IDs; reference by property
- Don't duplicate data from environment_summary or execution_state

BREVITY: Keep prompts concise (200-400 words). Use bullets, not paragraphs. No filler phrases.

Output ONLY valid JSON."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        reasoning_effort: str = "default",
        timeout: int = 120,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        
        # Check if EigenAI model (use direct HTTP instead of LiteLLM)
        self.is_eigenai = "eigenai" in model.lower() or "deepseek-v31" in model.lower()
        self._eigenai_client = None
        if self.is_eigenai:
            from src.agents.eigenai_client import EigenAIClient
            self._eigenai_client = EigenAIClient(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
            )
        
        # Optional feedback from previous executions
        self._feedback: Optional[Dict[str, Any]] = None
        
        # Track generated prompts for logging/debugging
        self._generated_prompts: List[Dict[str, Any]] = []
    
    def set_feedback(self, feedback: Dict[str, Any]):
        """
        Set aggregated feedback from previous task executions.
        
        When feedback is set, the prompt engineer will incorporate it
        into all generated prompts to improve agent performance.
        
        Args:
            feedback: Aggregated feedback dict with keys like:
                - explorer_feedback: str
                - planner_feedback: str  
                - worker_feedback: str
                - patterns_observed: List[str]
        """
        self._feedback = feedback
        logger.info("| [PromptEngineer] Feedback set - will be used for prompt generation")
    
    def clear_feedback(self):
        """Clear any stored feedback."""
        self._feedback = None
        logger.info("| [PromptEngineer] Feedback cleared")
    
    def get_generated_prompts(self) -> List[Dict[str, Any]]:
        """Get all prompts generated since last clear."""
        return self._generated_prompts.copy()
    
    def clear_generated_prompts(self):
        """Clear the list of generated prompts."""
        self._generated_prompts = []

    def _call_llm(self, system_prompt: str, user_content: str) -> Dict[str, Any]:
        """Make a synchronous LLM call."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        
        if self.base_url:
            kwargs["base_url"] = self.base_url
            
        # Add reasoning effort for supported models
        if "o3" in self.model or "o4" in self.model:
            kwargs["reasoning_effort"] = self.reasoning_effort
            
        start_time = time.time()
        if self.is_eigenai and self._eigenai_client:
            response = self._eigenai_client.completion(messages=messages)
        else:
            response = litellm.completion(**kwargs)
        elapsed = time.time() - start_time
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON from response
        try:
            # Try direct parse
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try extracting from code block
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # Try finding JSON object
                brace_match = re.search(r'\{[\s\S]*\}', content)
                if brace_match:
                    result = json.loads(brace_match.group(0))
                else:
                    raise ValueError(f"Could not parse JSON from response: {content[:500]}")
        
        logger.debug(f"PromptEngineer LLM call took {elapsed:.2f}s")
        return result

    def generate_explorer_prompt(
        self,
        base_prompt: str,
        task_description: str,
    ) -> str:
        """
        Generate a task-specific prompt for the Explorer agent.
        
        Args:
            base_prompt: The general Explorer prompt
            task_description: The specific task to accomplish
            
        Returns:
            Task-specific Explorer prompt
        """
        input_data: Dict[str, Any] = {
            "base_prompt": base_prompt,
            "task_description": task_description,
        }
        
        # Add feedback if available
        if self._feedback:
            input_data["feedback"] = self._feedback.get("explorer_feedback", "")
            logger.info("| [PromptEngineer] Generating Explorer prompt WITH feedback")
        else:
            logger.info("| [PromptEngineer] Generating Explorer prompt")
        
        user_content = json.dumps(input_data, indent=2)
        result = self._call_llm(self.EXPLORER_SYSTEM_PROMPT, user_content)
        task_specific_prompt = result.get("task_specific_prompt", base_prompt)
        
        # Track generated prompt
        self._generated_prompts.append({
            "role": "explorer",
            "has_feedback": self._feedback is not None,
            "prompt": task_specific_prompt,
        })
        
        logger.info("| [PromptEngineer] Explorer prompt generated")
        return task_specific_prompt

    def generate_planner_prompt(
        self,
        base_prompt: str,
        task_description: str,
        environment_summary: Dict[str, Any],
        execution_state: Optional[Dict[str, Any]] = None,
        planning_reason: str = "initial",
    ) -> str:
        """
        Generate a task-specific prompt for the Planner agent.
        
        Args:
            base_prompt: The general Planner prompt
            task_description: The specific task to accomplish
            environment_summary: What Explorer discovered
            execution_state: Current progress and artifacts (None if initial planning)
            planning_reason: "initial" or reason string for re-planning
            
        Returns:
            Task-specific Planner prompt
        """
        input_data: Dict[str, Any] = {
            "base_prompt": base_prompt,
            "task_description": task_description,
            "environment_summary": environment_summary,
            "execution_state": execution_state,
            "planning_reason": planning_reason,
        }
        
        # Add feedback if available
        if self._feedback:
            input_data["feedback"] = self._feedback.get("planner_feedback", "")
            logger.info(f"| [PromptEngineer] Generating Planner prompt WITH feedback (reason: {planning_reason})")
        else:
            logger.info(f"| [PromptEngineer] Generating Planner prompt (reason: {planning_reason})")
        
        user_content = json.dumps(input_data, indent=2)
        result = self._call_llm(self.PLANNER_SYSTEM_PROMPT, user_content)
        task_specific_prompt = result.get("task_specific_prompt", base_prompt)
        
        # Track generated prompt
        self._generated_prompts.append({
            "role": "planner",
            "planning_reason": planning_reason,
            "has_feedback": self._feedback is not None,
            "prompt": task_specific_prompt,
        })
        
        logger.info("| [PromptEngineer] Planner prompt generated")
        return task_specific_prompt

    def generate_worker_prompt(
        self,
        base_prompt: str,
        task_description: str,
        current_subtask: Dict[str, Any],
        environment_summary: Dict[str, Any],
        execution_state: Dict[str, Any],
    ) -> str:
        """
        Generate a task-specific prompt for a SINGLE Worker subtask.
        
        Called just before each subtask execution to incorporate latest state.
        
        Args:
            base_prompt: The general Worker prompt
            task_description: The original task description
            current_subtask: The specific subtask to execute
            environment_summary: Discovered resources
            execution_state: Current progress and artifacts from previous subtasks
            
        Returns:
            Task-specific Worker prompt for this subtask
        """
        subtask_id = current_subtask.get("id", "unknown")
        
        input_data: Dict[str, Any] = {
            "base_prompt": base_prompt,
            "task_description": task_description,
            "current_subtask": {
                "id": current_subtask.get("id"),
                "description": current_subtask.get("description"),
                "notes": current_subtask.get("notes"),
            },
            "environment_summary": environment_summary,
            "execution_state": {
                "completed_subtasks": [
                    p for p in execution_state.get("subtasks_progress", [])
                    if p.get("status") == "completed"
                ],
                "artifacts": execution_state.get("artifacts", {}),
            },
        }
        
        # Add feedback if available
        if self._feedback:
            input_data["feedback"] = self._feedback.get("worker_feedback", "")
            logger.info(f"| [PromptEngineer] Generating Worker prompt WITH feedback for subtask {subtask_id}")
        else:
            logger.info(f"| [PromptEngineer] Generating Worker prompt for subtask {subtask_id}")
        
        user_content = json.dumps(input_data, indent=2)
        result = self._call_llm(self.WORKER_SYSTEM_PROMPT, user_content)
        task_specific_prompt = result.get("task_specific_prompt", base_prompt)
        
        # Track generated prompt
        self._generated_prompts.append({
            "role": "worker",
            "subtask_id": subtask_id,
            "has_feedback": self._feedback is not None,
            "prompt": task_specific_prompt,
        })
        
        logger.info(f"| [PromptEngineer] Worker prompt generated for subtask {subtask_id}")
        return task_specific_prompt

