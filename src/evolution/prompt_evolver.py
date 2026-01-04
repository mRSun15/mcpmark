"""
Prompt Evolution Agent
======================

Evolves base prompts using synthesized feedback.
CRITICAL: Preserves the exact input/output schema of prompts.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

import litellm

from src.logger import get_logger

logger = get_logger(__name__)


class PromptEvolutionAgent:
    """
    Evolves base prompts using synthesized feedback.
    
    CRITICAL: The evolved prompt MUST preserve the exact input/output schema
    as the base prompt. Only the instructional content can be modified.
    """
    
    SYSTEM_PROMPT = """You are a Prompt Evolution Agent. Your job is to improve agent prompts based on
synthesized feedback from previous task executions.

CRITICAL CONSTRAINT: You MUST preserve the EXACT input/output schema of the base prompt.
- Do NOT add, remove, or rename any fields in the input/output JSON schemas
- Do NOT change the structure or format requirements
- ONLY modify the instructional text, rules, and guidance,...

Input:
- role: "explorer" | "planner" | "worker"
- base_prompt: The original prompt for this role (contains schema definitions)
- role_feedback: Synthesized feedback specific to this role
- patterns_observed: List of behavioral patterns from feedback

Output (JSON only, no markdown):
{
  "evolved_prompt": "The improved prompt text (MUST keep same schema)...",
  "changes_summary": "Brief summary of textual changes made"
}

Rules for evolution:
1. PRESERVE ALL SCHEMA DEFINITIONS EXACTLY AS-IS
   - Keep "Input JSON schema:" sections unchanged
   - Keep "Output JSON schema:" sections unchanged  
   - Keep "Output:" sections unchanged
   - Keep field names, types, and structure identical
   
2. ONLY MODIFY instructional text:
   - Add new rules or guidelines based on feedback
   - Clarify existing instructions
   - Add warnings about observed failure patterns
   - Strengthen guidance for problematic areas
   
3. Keep changes MINIMAL and TARGETED
   - Don't rewrite the entire prompt
   - Add 2-4 new bullet points at most
   - Place new guidance in appropriate sections (e.g., add to "Rules:" or "Important Notes:")

4. Output ONLY valid JSON, no markdown code fences"""

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

    def evolve_prompt(
        self,
        role: str,
        base_prompt: str,
        role_feedback: str,
        patterns_observed: List[str]
    ) -> Dict[str, Any]:
        """
        Evolve a single role's prompt while preserving schema.
        
        Args:
            role: "explorer", "planner", or "worker"
            base_prompt: Original prompt with schema definitions
            role_feedback: Synthesized feedback for this role
            patterns_observed: Behavioral patterns from feedback
        
        Returns:
            {
                "evolved_prompt": str,  # Same schema, evolved guidance
                "changes_summary": str
            }
        """
        input_data = {
            "role": role,
            "base_prompt": base_prompt,
            "role_feedback": role_feedback,
            "patterns_observed": patterns_observed
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
                response = litellm.completion(**completion_kwargs)
                
                content = response.choices[0].message.content
                evolved = self._parse_evolution_response(content)
                
                # Validate schema preservation
                self._validate_schema_preserved(base_prompt, evolved["evolved_prompt"], role)
                
                logger.info(
                    f"| [PromptEvolver] Evolved {role} prompt: {evolved['changes_summary'][:100]}..."
                )
                return evolved
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    logger.warning(f"| [PromptEvolver] JSON parse error, retrying: {e}")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "Invalid JSON. Reply with ONLY valid JSON, no markdown code fences."
                    })
                    completion_kwargs["messages"] = messages
                    continue
                raise
            except Exception as e:
                logger.error(f"| [PromptEvolver] Error for {role}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        raise RuntimeError(f"PromptEvolutionAgent failed to evolve {role} prompt")

    def _parse_evolution_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM response into evolution dict."""
        # Remove markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        
        evolved = json.loads(content)
        
        # Validate required fields
        if "evolved_prompt" not in evolved:
            raise ValueError("Missing 'evolved_prompt' in response")
        if "changes_summary" not in evolved:
            evolved["changes_summary"] = "No summary provided"
        
        return evolved

    def _validate_schema_preserved(
        self, 
        base_prompt: str, 
        evolved_prompt: str,
        role: str
    ):
        """
        Verify that input/output schema definitions are unchanged.
        Raises error if schema was modified.
        """
        # Extract schema-like sections using regex patterns
        schema_patterns = [
            # JSON schema blocks
            r'(Input JSON schema:|Output JSON schema:|Output:)\s*\{[^}]+\}',
            # Field definitions in curly braces that look like schemas
            r'"[a-z_]+"\s*:\s*"[^"]*"',  # Field definitions
        ]
        
        def extract_schemas(text: str) -> set:
            schemas = set()
            for pattern in schema_patterns:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    # Normalize whitespace
                    normalized = re.sub(r'\s+', ' ', str(match).strip())
                    schemas.add(normalized)
            return schemas
        
        base_schemas = extract_schemas(base_prompt)
        evolved_schemas = extract_schemas(evolved_prompt)
        
        # Check if any base schemas are missing in evolved
        missing = base_schemas - evolved_schemas
        if missing and len(missing) > len(base_schemas) * 0.3:
            # Only warn if significant portion is missing (allow minor variations)
            logger.warning(
                f"| [PromptEvolver] Schema validation warning for {role}: "
                f"Some schema elements may have changed"
            )
        
        # Log the change for debugging
        base_len = len(base_prompt)
        evolved_len = len(evolved_prompt)
        change_pct = abs(evolved_len - base_len) / base_len * 100
        logger.debug(
            f"| [PromptEvolver] {role} prompt size: {base_len} -> {evolved_len} "
            f"({change_pct:.1f}% change)"
        )

    def evolve_all_prompts_sync(
        self,
        base_prompts: Dict[str, str],
        feedback: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evolve prompts for all roles while preserving their schemas.
        
        Args:
            base_prompts: {"explorer": "...", "planner": "...", "worker": "..."}
            feedback: Synthesized feedback from FeedbackAgent
        
        Returns:
            {
                "explorer": {"evolved_prompt": "...", "changes_summary": "..."},
                "planner": {"evolved_prompt": "...", "changes_summary": "..."},
                "worker": {"evolved_prompt": "...", "changes_summary": "..."}
            }
        """
        results = {}
        patterns = feedback.get("patterns_observed", [])
        
        for role in ["explorer", "planner", "worker"]:
            if role not in base_prompts:
                logger.warning(f"| [PromptEvolver] No base prompt for {role}, skipping")
                continue
            
            role_feedback = feedback.get(f"{role}_feedback", "")
            if not role_feedback:
                logger.warning(f"| [PromptEvolver] No feedback for {role}, using base prompt")
                results[role] = {
                    "evolved_prompt": base_prompts[role],
                    "changes_summary": "No changes - no feedback provided"
                }
                continue
            
            results[role] = self.evolve_prompt(
                role=role,
                base_prompt=base_prompts[role],
                role_feedback=role_feedback,
                patterns_observed=patterns
            )
        
        return results

