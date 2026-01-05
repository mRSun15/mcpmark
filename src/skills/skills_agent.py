"""
Skills Agent
============

Pre-task skill generation using LLM to propose useful composite operations.
"""

import json
import time
from typing import Any, Dict, List, Optional

import litellm

from src.logger import get_logger
from .schema import Skill

logger = get_logger(__name__)


class SkillsAgent:
    """Generates task-specific skills before execution."""
    
    SYSTEM_PROMPT = """You are a Skills Engineer for a multi-agent system. Before a task begins, you analyze what composite operations might be useful and create reusable skills.

A SKILL is a composite operation that:
- Combines multiple tool calls into one
- Reduces context window usage (intermediate data doesn't enter LLM context)
- Encapsulates common patterns for reuse

INPUT:
- task_description: What needs to be accomplished
- base_tools: Available primitive tools
- existing_skills: Skills already in the library

YOUR JOB:
1. Analyze what operations the task likely needs
2. Identify opportunities to create composite skills that:
   - Copy/move files without loading content into context
   - Batch operations on multiple files
   - Find-and-replace across files
   - Collect information from multiple sources
3. Generate skill definitions

SKILL DEFINITION FORMAT:
{
  "name": "skill_name_snake_case",
  "description": "Clear description of what the skill does",
  "parameters": [
    {"name": "param_name", "type": "string|number|boolean|array", "description": "...", "required": true}
  ],
  "steps": [
    // Available step types:
    
    // 1. Tool call - execute a tool
    {"tool": "tool_name", "args": {"arg": "$variable"}, "output": "$result_var"},
    
    // 2. Set variable
    {"set": "$var_name", "value": "value or $other_var"},
    
    // 3. Append to list
    {"append": "$list_var", "value": "$item"},
    
    // 4. Foreach loop
    {"foreach": "$items", "as": "$item", "do": [...steps...], "collect": "$results"},
    
    // 5. Conditional
    {"if": "$condition == value", "then": [...steps...], "else": [...steps...]},
    
    // 6. LLM call (for analysis/summarization within skill)
    {"llm": {"prompt": "Analyze: $content", "output": "$analysis"}},
    
    // 7. Return value
    {"return": "$result"}
  ]
}

RULES:
- Only propose skills LIKELY to be useful for THIS task
- Don't duplicate existing_skills (check by name AND functionality)
- Keep skills general enough to be reusable in other tasks
- Skill names should be descriptive: copy_file, batch_rename, find_replace, etc.
- Parameters use $name syntax in steps

OUTPUT (JSON only, no markdown):
{
  "reasoning": "Brief analysis of task needs and why these skills help",
  "proposed_skills": [
    { ...skill definition... }
  ]
}

If no new skills are needed, return: {"reasoning": "...", "proposed_skills": []}"""

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
    
    def propose_skills(
        self,
        task_description: str,
        base_tools: List[Dict],
        existing_skills: List[Dict],
    ) -> List[Skill]:
        """
        Propose skills for a specific task.
        
        Args:
            task_description: The task instruction
            base_tools: List of base tool definitions (OpenAI format)
            existing_skills: List of existing skill summaries
        
        Returns:
            List of proposed Skill objects
        """
        logger.info("[SkillsAgent] Proposing skills for task")
        
        # Prepare input (full data - SkillsAgent needs complete info to compose tools)
        input_data = {
            "task_description": task_description,
            "base_tools": [
                {
                    "name": t.get("function", {}).get("name", t.get("name", "")),
                    "description": t.get("function", {}).get("description", t.get("description", "")),
                }
                for t in base_tools
            ],
            "existing_skills": existing_skills,
        }
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(input_data, indent=2, ensure_ascii=False)}
        ]
        
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.base_url:
            completion_kwargs["base_url"] = self.base_url
        # Only add reasoning_effort for supported models (o3/o4)
        if "o3" in self.model or "o4" in self.model:
            completion_kwargs["reasoning_effort"] = self.reasoning_effort
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Use sync completion like PromptEngineer
                if self.is_eigenai and self._eigenai_client:
                    response = self._eigenai_client.completion(messages=messages)
                else:
                    response = litellm.completion(**completion_kwargs)
                content = response.choices[0].message.content
                
                result = self._parse_response(content)
                
                skills = []
                for skill_dict in result.get("proposed_skills", []):
                    try:
                        skill = Skill.from_dict(skill_dict)
                        skills.append(skill)
                        logger.info(f"[SkillsAgent] Proposed skill: {skill.name}")
                    except Exception as e:
                        logger.warning(f"[SkillsAgent] Failed to parse skill: {e}")
                
                logger.info(f"[SkillsAgent] Proposed {len(skills)} skills")
                return skills
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    logger.warning(f"[SkillsAgent] JSON parse error, retrying: {e}")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "Invalid JSON. Reply with ONLY valid JSON, no markdown code fences."
                    })
                    completion_kwargs["messages"] = messages
                    continue
                logger.error(f"[SkillsAgent] Failed to parse response: {e}")
                return []
            except Exception as e:
                logger.error(f"[SkillsAgent] Error: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return []
        
        return []
    
    def propose_skills_sync(
        self,
        task_description: str,
        base_tools: List[Dict],
        existing_skills: List[Dict],
    ) -> List[Skill]:
        """Synchronous version of propose_skills."""
        import asyncio
        return asyncio.run(self.propose_skills(task_description, base_tools, existing_skills))
    
    def _parse_response(self, content: str) -> Dict:
        """Parse the LLM response."""
        content = content.strip()
        
        # Remove markdown code fences if present
        if content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        
        return json.loads(content)

