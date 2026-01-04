"""
Feedback Agent
==============

Two-stage feedback generation:
1. TaskFeedbackAgent: Generates individual feedback for each task (analysis + instructions)
2. FeedbackAggregator: Aggregates all task feedbacks into final role-specific guidance

Uses verification result, execution log, AND base prompts for context.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import litellm

from src.logger import get_logger

logger = get_logger(__name__)


class TaskFeedbackAgent:
    """
    Generates feedback for a SINGLE task.
    
    Output has two parts:
    1. analysis: Why the task succeeded/failed (for understanding, not used in final feedback)
    2. instructions: Actionable guidance for each role (used in aggregation)
    """
    
    SYSTEM_PROMPT = """You are a Task Feedback Agent that analyzes a single task execution.

The multi-agent system has 3 roles:
1. Explorer: Discovers environment structure and resources using read-only tools
2. Planner: Creates ordered subtask plans based on environment and task description
3. Worker: Executes subtasks using tools to modify the environment

You will receive:
- task_result: Contains:
  - passed: Whether task passed/failed
  - error_message: Execution error (if task crashed during execution)
  - verification_error: Brief verification error message (if any)
  - verification_output: DETAILED verification log showing exactly what passed/failed checks
  - execution_log: Full execution trace showing how each agent performed
- base_prompts: The current prompts used by Explorer, Planner, and Worker

IMPORTANT: 
- Check error_message first - if present, the task crashed during execution
- Check verification_output for specific failure reasons (e.g., "Missing required dependencies")
- Use these and execution_log to identify the root cause accurately.

Your task: Analyze this SINGLE task execution and provide TWO parts:

PART 1 - Analysis (for understanding only, will NOT be used in final feedback):
- If FAILED: Identify the root cause. Which agent(s) made mistakes? What went wrong?
- If PASSED: What strategies worked well? What should be reinforced?

PART 2 - Instructions (WILL be used for prompt evolution):
- Provide actionable, GENERAL guidance for each role based on this task
- Focus on what could be improved or reinforced in the prompts
- Keep it general - no specific file paths, task names, or domain references

Output (JSON only, no markdown):
{
  "analysis": {
    "outcome": "passed" | "failed",
    "root_cause": "Analysis of why the task succeeded or failed...",
    "agent_performance": {
      "explorer": "How explorer performed...",
      "planner": "How planner performed...",
      "worker": "How worker performed..."
    }
  },
  "instructions": {
    "explorer_feedback": "Actionable guidance for Explorer prompt...",
    "planner_feedback": "Actionable guidance for Planner prompt...",
    "worker_feedback": "Actionable guidance for Worker prompt..."
  }
}

Rules:
1. Be SPECIFIC in analysis but GENERAL in instructions
2. In instructions, focus on behavioral patterns, not task-specific details
3. If a role performed well, note what to reinforce; if poorly, note what to fix
4. Keep each instruction concise (1-3 key points)
5. Output ONLY valid JSON, no markdown code fences"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        reasoning_effort: str = "default",
        timeout: int = 300,
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

    def generate_task_feedback(
        self,
        task_result: Dict[str, Any],
        base_prompts: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Generate feedback for a single task.
        
        Returns:
            {
                "analysis": {...},      # For understanding (not used in aggregation)
                "instructions": {...}   # Used for aggregation
            }
        """
        # Truncate execution log if too long
        execution_log = task_result.get("execution_log", "")
        max_log_length = 50000
        if len(execution_log) > max_log_length:
            half = max_log_length // 2
            execution_log = (
                execution_log[:half] + 
                "\n\n... [LOG TRUNCATED] ...\n\n" + 
                execution_log[-half:]
            )
        
        input_data = {
            "task_result": {
                "passed": task_result["passed"],
                "error_message": task_result.get("error_message"),  # Execution error (if any)
                "verification_error": task_result.get("verification_error"),
                "verification_output": task_result.get("verification_output"),  # Detailed verification log
                "execution_log": execution_log
            },
            "base_prompts": {
                "explorer": base_prompts.get("explorer", "")[:2000],  # Truncate for context
                "planner": base_prompts.get("planner", "")[:2000],
                "worker": base_prompts.get("worker", "")[:2000],
            }
        }
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(input_data, ensure_ascii=False, indent=2)}
        ]
        
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.base_url:
            completion_kwargs["base_url"] = self.base_url
        if self.reasoning_effort != "default":
            completion_kwargs["reasoning_effort"] = self.reasoning_effort
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if self.is_eigenai and self._eigenai_client:
                    response = self._eigenai_client.completion(messages=messages)
                else:
                    response = litellm.completion(**completion_kwargs)
                content = response.choices[0].message.content
                feedback = self._parse_response(content)
                return feedback
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    logger.warning(f"| [TaskFeedback] JSON parse error, retrying: {e}")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user", 
                        "content": "Invalid JSON. Reply with ONLY valid JSON, no markdown code fences."
                    })
                    completion_kwargs["messages"] = messages
                    continue
                raise
            except Exception as e:
                logger.error(f"| [TaskFeedback] Error: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        raise RuntimeError("TaskFeedbackAgent failed to produce valid feedback")

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM response."""
        content = content.strip()
        if content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        
        feedback = json.loads(content)
        
        # Validate structure
        if "analysis" not in feedback:
            feedback["analysis"] = {"outcome": "unknown", "root_cause": "No analysis provided"}
        if "instructions" not in feedback:
            raise ValueError("Missing 'instructions' in response")
        
        return feedback


class FeedbackAggregator:
    """
    Aggregates individual task feedbacks into final role-specific guidance.
    Only uses the 'instructions' part from each task feedback.
    """
    
    SYSTEM_PROMPT = """You are a Feedback Aggregator for a multi-agent system.

You will receive feedback instructions from multiple task executions. Each task's feedback
contains guidance for three roles: Explorer, Planner, and Worker.

Your job: Synthesize all the individual task feedbacks into FINAL, comprehensive guidance
for each role. Look for patterns across tasks and prioritize recurring issues.

Input:
- task_feedbacks: List of instruction sets from individual tasks
  [
    {
      "task_index": 1,
      "passed": true/false,
      "explorer_feedback": "...",
      "planner_feedback": "...",
      "worker_feedback": "..."
    },
    ...
  ]

Output (JSON only, no markdown):
{
  "explorer_feedback": "Final synthesized guidance for Explorer...",
  "planner_feedback": "Final synthesized guidance for Planner...",
  "worker_feedback": "Final synthesized guidance for Worker...",
  "patterns_observed": ["pattern1", "pattern2", ...],
  "tasks_analyzed": N
}

Rules:
1. SYNTHESIZE - identify common themes and recurring issues across tasks
2. PRIORITIZE - focus on feedback that appears in multiple failed tasks
3. Keep feedback GENERAL - no specific task names, file paths, or domain references
4. Keep each role's feedback concise but actionable (3-5 key points max)
5. patterns_observed should capture high-level behavioral patterns
6. Output ONLY valid JSON, no markdown code fences"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        reasoning_effort: str = "default",
        timeout: int = 300,
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

    def aggregate_feedbacks(
        self,
        task_feedbacks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate all task feedbacks into final guidance.
        
        Args:
            task_feedbacks: List of task feedback results (with 'instructions' field)
        
        Returns:
            Final aggregated feedback
        """
        # Extract only the instructions part for aggregation
        instructions_list = []
        for i, tf in enumerate(task_feedbacks):
            instructions = tf.get("instructions", {})
            instructions_list.append({
                "task_index": i + 1,
                "passed": tf.get("analysis", {}).get("outcome") == "passed",
                "explorer_feedback": instructions.get("explorer_feedback", ""),
                "planner_feedback": instructions.get("planner_feedback", ""),
                "worker_feedback": instructions.get("worker_feedback", ""),
            })
        
        input_data = {"task_feedbacks": instructions_list}
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(input_data, ensure_ascii=False, indent=2)}
        ]
        
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.base_url:
            completion_kwargs["base_url"] = self.base_url
        if self.reasoning_effort != "default":
            completion_kwargs["reasoning_effort"] = self.reasoning_effort
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if self.is_eigenai and self._eigenai_client:
                    response = self._eigenai_client.completion(messages=messages)
                else:
                    response = litellm.completion(**completion_kwargs)
                content = response.choices[0].message.content
                aggregated = self._parse_response(content, len(task_feedbacks))
                
                logger.info(f"| [Aggregator] Synthesized feedback from {len(task_feedbacks)} tasks")
                return aggregated
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    logger.warning(f"| [Aggregator] JSON parse error, retrying: {e}")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user", 
                        "content": "Invalid JSON. Reply with ONLY valid JSON, no markdown code fences."
                    })
                    completion_kwargs["messages"] = messages
                    continue
                raise
            except Exception as e:
                logger.error(f"| [Aggregator] Error: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        raise RuntimeError("FeedbackAggregator failed to produce valid feedback")

    def _parse_response(self, content: str, task_count: int) -> Dict[str, Any]:
        """Parse the LLM response."""
        content = content.strip()
        if content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        
        feedback = json.loads(content)
        
        # Validate required fields
        required_fields = ["explorer_feedback", "planner_feedback", "worker_feedback"]
        for field in required_fields:
            if field not in feedback:
                raise ValueError(f"Missing required field: {field}")
        
        if "patterns_observed" not in feedback:
            feedback["patterns_observed"] = []
        feedback["tasks_analyzed"] = task_count
        
        return feedback


class FailureAnalyzer:
    """
    Analyzes all failed task feedbacks and clusters them by similar failure reasons.
    Produces a final summary with grouped failures and recommendations.
    """
    
    SYSTEM_PROMPT = """You are a Failure Pattern Analyzer for a multi-agent system.

You will receive feedback from multiple FAILED tasks. Each feedback contains:
- task_name: The name of the failed task
- analysis: Why the task failed (root cause, agent performance)
- instructions: Suggested improvements for each agent role

Your job: Analyze ALL failed tasks and cluster them by SIMILAR failure reasons.

Output (JSON only, no markdown):
{
  "total_failed": N,
  "clusters": [
    {
      "cluster_name": "Short descriptive name for this failure pattern",
      "description": "Detailed description of what went wrong in these tasks",
      "affected_tasks": ["task_name_1", "task_name_2", ...],
      "root_causes": ["Common root cause 1", "Common root cause 2"],
      "agent_issues": {
        "explorer": "Common Explorer issues in this cluster (or null if none)",
        "planner": "Common Planner issues in this cluster (or null if none)",
        "worker": "Common Worker issues in this cluster (or null if none)"
      },
      "recommendations": ["Actionable fix 1", "Actionable fix 2", ...]
    },
    ...
  ],
  "cross_cutting_issues": [
    "Issues that appear across multiple clusters"
  ],
  "priority_fixes": [
    "Top 3-5 highest-impact fixes that would address the most failures"
  ],
  "summary": "2-3 sentence executive summary of the main failure patterns"
}

Rules:
1. Group tasks with genuinely SIMILAR failure patterns - don't force unrelated tasks together
2. A task can only belong to ONE cluster
3. Be specific about root causes but keep recommendations actionable
4. Priority fixes should be ordered by impact (most failures fixed first)
5. Output ONLY valid JSON, no markdown code fences"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        reasoning_effort: str = "default",
        timeout: int = 300,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        
        self.is_eigenai = "eigenai" in model.lower() or "deepseek-v31" in model.lower()
        self._eigenai_client = None
        if self.is_eigenai:
            from src.agents.eigenai_client import EigenAIClient
            self._eigenai_client = EigenAIClient(
                api_key=api_key, base_url=base_url, model=model, timeout=timeout,
            )

    def analyze_failures(self, failed_feedbacks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze all failed task feedbacks and cluster by similar reasons.
        
        Args:
            failed_feedbacks: List of feedbacks for failed tasks, each with:
                - task_name: str
                - analysis: dict (outcome, root_cause, agent_performance)
                - instructions: dict (explorer/planner/worker feedback)
        
        Returns:
            Clustered analysis with recommendations
        """
        if not failed_feedbacks:
            return {
                "total_failed": 0,
                "clusters": [],
                "cross_cutting_issues": [],
                "priority_fixes": [],
                "summary": "No failed tasks to analyze."
            }
        
        # Prepare input data
        input_data = {
            "failed_tasks": [
                {
                    "task_name": fb.get("task_name", f"task_{i}"),
                    "analysis": fb.get("analysis", {}),
                    "instructions": fb.get("instructions", {}),
                }
                for i, fb in enumerate(failed_feedbacks)
            ]
        }
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(input_data, ensure_ascii=False, indent=2)}
        ]
        
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.base_url:
            completion_kwargs["base_url"] = self.base_url
        if self.reasoning_effort != "default":
            completion_kwargs["reasoning_effort"] = self.reasoning_effort
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if self.is_eigenai and self._eigenai_client:
                    response = self._eigenai_client.completion(messages=messages)
                else:
                    response = litellm.completion(**completion_kwargs)
                content = response.choices[0].message.content
                result = self._parse_response(content, len(failed_feedbacks))
                
                logger.info(f"| [FailureAnalyzer] Clustered {len(failed_feedbacks)} failures into {len(result['clusters'])} groups")
                return result
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    logger.warning(f"| [FailureAnalyzer] JSON parse error, retrying: {e}")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "Invalid JSON. Reply with ONLY valid JSON, no markdown code fences."
                    })
                    completion_kwargs["messages"] = messages
                    continue
                raise
            except Exception as e:
                logger.error(f"| [FailureAnalyzer] Error: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        raise RuntimeError("FailureAnalyzer failed to produce valid analysis")

    def _parse_response(self, content: str, total_failed: int) -> Dict[str, Any]:
        """Parse the LLM response."""
        content = content.strip()
        if content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        
        result = json.loads(content)
        
        # Ensure required fields
        result["total_failed"] = total_failed
        if "clusters" not in result:
            result["clusters"] = []
        if "cross_cutting_issues" not in result:
            result["cross_cutting_issues"] = []
        if "priority_fixes" not in result:
            result["priority_fixes"] = []
        if "summary" not in result:
            result["summary"] = f"Analyzed {total_failed} failed tasks."
        
        return result


class FeedbackAgent:
    """
    Orchestrates the two-stage feedback generation:
    1. Generate individual feedback for each task (with analysis + instructions)
    2. Aggregate all feedbacks into final guidance (using only instructions)
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        reasoning_effort: str = "default",
        timeout: int = 300,
    ):
        self.task_feedback_agent = TaskFeedbackAgent(
            model=model,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
        self.aggregator = FeedbackAggregator(
            model=model,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )

    def generate_all_task_feedbacks(
        self,
        task_results: List[Dict[str, Any]],
        base_prompts: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Generate feedback for each task individually.
        
        Returns:
            List of task feedbacks (each with 'analysis' and 'instructions')
        """
        all_feedbacks = []
        for i, result in enumerate(task_results):
            logger.info(f"| [FeedbackAgent] Generating feedback for task {i + 1}/{len(task_results)}")
            feedback = self.task_feedback_agent.generate_task_feedback(result, base_prompts)
            all_feedbacks.append(feedback)
        return all_feedbacks

    def aggregate_feedbacks(
        self,
        task_feedbacks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate individual task feedbacks into final guidance.
        """
        return self.aggregator.aggregate_feedbacks(task_feedbacks)

    def generate_final_feedback_sync(
        self,
        task_results: List[Dict[str, Any]],
        base_prompts: Dict[str, str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Full pipeline: Generate individual feedbacks, then aggregate.
        
        Args:
            task_results: List of task results
            base_prompts: Base prompts for context
        
        Returns:
            (task_feedbacks, aggregated_feedback)
            - task_feedbacks: List of individual task feedbacks (with analysis + instructions)
            - aggregated_feedback: Final aggregated feedback for prompt evolution
        """
        # Stage 1: Generate individual feedbacks
        task_feedbacks = self.generate_all_task_feedbacks(task_results, base_prompts)
        
        # Stage 2: Aggregate into final feedback
        aggregated = self.aggregate_feedbacks(task_feedbacks)
        
        return task_feedbacks, aggregated
