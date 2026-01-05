"""
Skill Executor
==============

Executes composite skills by interpreting step definitions.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Awaitable

from src.logger import get_logger
from .schema import (
    Skill, Step, ToolStep, LLMStep, ForeachStep, IfStep,
    SetStep, AppendStep, ReturnStep
)

logger = get_logger(__name__)


class SkillExecutor:
    """Executes skills by interpreting their step definitions."""
    
    def __init__(
        self,
        tool_caller: Callable[[str, Dict], Awaitable[Any]],
        llm_caller: Optional[Callable[[str, Optional[str]], Awaitable[str]]] = None,
    ):
        """
        Initialize the executor.
        
        Args:
            tool_caller: Async function to call tools: (tool_name, args) -> result
            llm_caller: Async function for LLM calls: (prompt, system_prompt) -> response
        """
        self.tool_caller = tool_caller
        self.llm_caller = llm_caller
    
    async def execute(self, skill: Skill, args: Dict[str, Any]) -> Any:
        """
        Execute a skill with given arguments.
        
        Args:
            skill: The skill to execute
            args: Input arguments matching skill parameters
        
        Returns:
            The result of the skill execution
        """
        logger.info(f"[SkillExecutor] Executing skill: {skill.name}")
        
        # Initialize variables with input args (prefixed with $)
        variables: Dict[str, Any] = {f"${k}": v for k, v in args.items()}
        
        # Execute each step
        for i, step in enumerate(skill.steps):
            try:
                result = await self._execute_step(step, variables)
                if result is not None and isinstance(step, ReturnStep):
                    logger.info(f"[SkillExecutor] Skill {skill.name} completed with return")
                    return result
            except Exception as e:
                logger.error(f"[SkillExecutor] Step {i} failed: {e}")
                raise
        
        # Return $result if set, otherwise None
        result = variables.get("$result")
        logger.info(f"[SkillExecutor] Skill {skill.name} completed")
        return result
    
    async def _execute_step(self, step: Step, variables: Dict[str, Any]) -> Any:
        """Execute a single step."""
        
        if isinstance(step, ToolStep):
            return await self._execute_tool_step(step, variables)
        
        elif isinstance(step, LLMStep):
            return await self._execute_llm_step(step, variables)
        
        elif isinstance(step, ForeachStep):
            return await self._execute_foreach_step(step, variables)
        
        elif isinstance(step, IfStep):
            return await self._execute_if_step(step, variables)
        
        elif isinstance(step, SetStep):
            return self._execute_set_step(step, variables)
        
        elif isinstance(step, AppendStep):
            return self._execute_append_step(step, variables)
        
        elif isinstance(step, ReturnStep):
            return self._execute_return_step(step, variables)
        
        else:
            raise ValueError(f"Unknown step type: {type(step)}")
    
    async def _execute_tool_step(self, step: ToolStep, variables: Dict[str, Any]) -> Any:
        """Execute a tool call step."""
        # Resolve argument values
        resolved_args = self._resolve_dict(step.args, variables)
        
        logger.debug(f"[SkillExecutor] Calling tool: {step.tool} with args: {resolved_args}")
        
        # Call the tool
        result = await self.tool_caller(step.tool, resolved_args)
        
        # Store result if output variable specified
        if step.output:
            variables[step.output] = result
        
        return result
    
    async def _execute_llm_step(self, step: LLMStep, variables: Dict[str, Any]) -> str:
        """Execute an LLM call step."""
        if not self.llm_caller:
            raise RuntimeError("LLM caller not configured for skill executor")
        
        # Resolve prompt
        prompt = self._resolve_string(step.prompt, variables)
        system_prompt = None
        if step.system_prompt:
            system_prompt = self._resolve_string(step.system_prompt, variables)
        
        logger.debug(f"[SkillExecutor] LLM call with prompt length: {len(prompt)}")
        
        # Call LLM
        result = await self.llm_caller(prompt, system_prompt)
        
        # Store result
        variables[step.output] = result
        
        return result
    
    async def _execute_foreach_step(self, step: ForeachStep, variables: Dict[str, Any]) -> List:
        """Execute a foreach loop."""
        # Get the items to iterate
        items = self._resolve_value(step.items, variables)
        if not isinstance(items, (list, tuple)):
            items = [items]
        
        results = []
        
        for i, item in enumerate(items):
            # Set loop variable
            variables[step.as_var] = item
            variables["$index"] = i
            
            # Execute inner steps
            step_result = None
            for inner_step in step.do:
                step_result = await self._execute_step(inner_step, variables)
            
            # Collect results if specified
            if step.collect and step_result is not None:
                results.append(step_result)
        
        # Store collected results
        if step.collect:
            variables[step.collect] = results
        
        return results
    
    async def _execute_if_step(self, step: IfStep, variables: Dict[str, Any]) -> Any:
        """Execute a conditional step."""
        condition = self._evaluate_condition(step.condition, variables)
        
        steps_to_run = step.then if condition else (step.else_steps or [])
        
        result = None
        for inner_step in steps_to_run:
            result = await self._execute_step(inner_step, variables)
        
        return result
    
    def _execute_set_step(self, step: SetStep, variables: Dict[str, Any]) -> None:
        """Set a variable value."""
        value = self._resolve_value(step.value, variables)
        variables[step.var] = value
    
    def _execute_append_step(self, step: AppendStep, variables: Dict[str, Any]) -> None:
        """Append to a list variable."""
        list_var = step.list_var
        if list_var not in variables:
            variables[list_var] = []
        
        value = self._resolve_value(step.value, variables)
        variables[list_var].append(value)
    
    def _execute_return_step(self, step: ReturnStep, variables: Dict[str, Any]) -> Any:
        """Return a value from the skill."""
        return self._resolve_value(step.value, variables)
    
    def _resolve_value(self, value: Any, variables: Dict[str, Any]) -> Any:
        """Resolve a value, replacing $var references."""
        if isinstance(value, str):
            # Check if entire string is a variable reference
            if value.startswith("$") and value in variables:
                return variables[value]
            # Otherwise do string interpolation
            return self._resolve_string(value, variables)
        elif isinstance(value, dict):
            return self._resolve_dict(value, variables)
        elif isinstance(value, list):
            return [self._resolve_value(v, variables) for v in value]
        else:
            return value
    
    def _resolve_string(self, s: str, variables: Dict[str, Any]) -> str:
        """Resolve variable references in a string."""
        def replacer(match):
            var_name = match.group(0)
            if var_name in variables:
                val = variables[var_name]
                return str(val) if not isinstance(val, str) else val
            return var_name
        
        # Match $word patterns
        return re.sub(r'\$\w+', replacer, s)
    
    def _resolve_dict(self, d: Dict, variables: Dict[str, Any]) -> Dict:
        """Resolve variable references in a dictionary."""
        return {k: self._resolve_value(v, variables) for k, v in d.items()}
    
    def _evaluate_condition(self, condition: str, variables: Dict[str, Any]) -> bool:
        """Evaluate a condition expression."""
        # Resolve variables in condition
        resolved = self._resolve_string(condition, variables)
        
        # Simple condition evaluations
        if resolved.lower() in ("true", "1", "yes"):
            return True
        if resolved.lower() in ("false", "0", "no", "none", "null", ""):
            return False
        
        # Check for comparison operators
        for op in ["==", "!=", ">=", "<=", ">", "<"]:
            if op in resolved:
                parts = resolved.split(op, 1)
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    try:
                        left_val = self._parse_value(left)
                        right_val = self._parse_value(right)
                        if op == "==":
                            return left_val == right_val
                        elif op == "!=":
                            return left_val != right_val
                        elif op == ">=":
                            return left_val >= right_val
                        elif op == "<=":
                            return left_val <= right_val
                        elif op == ">":
                            return left_val > right_val
                        elif op == "<":
                            return left_val < right_val
                    except Exception:
                        pass
        
        # Check for "in" operator
        if " in " in resolved:
            parts = resolved.split(" in ", 1)
            if len(parts) == 2:
                item = parts[0].strip().strip("'\"")
                container = parts[1].strip()
                if container in variables:
                    return item in variables[container]
        
        # Default: truthy check
        return bool(resolved)
    
    def _parse_value(self, s: str) -> Any:
        """Parse a string value to appropriate type."""
        s = s.strip().strip("'\"")
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        if s.lower() == "true":
            return True
        if s.lower() == "false":
            return False
        if s.lower() in ("none", "null"):
            return None
        return s

