"""
Skill Schema
============

Data structures for defining composite skills.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
import json


@dataclass
class SkillParameter:
    """A parameter for a skill."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SkillParameter":
        return cls(**data)


@dataclass
class ToolStep:
    """Execute a tool call."""
    tool: str
    args: Dict[str, Any]  # Values can be "$var" references
    output: Optional[str] = None  # Variable to store result

    def to_dict(self) -> Dict:
        d = {"tool": self.tool, "args": self.args}
        if self.output:
            d["output"] = self.output
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "ToolStep":
        return cls(
            tool=data["tool"],
            args=data.get("args", {}),
            output=data.get("output"),
        )


@dataclass
class LLMStep:
    """Execute an LLM call within the skill."""
    prompt: str  # Can include "$var" references
    output: str  # Variable to store result
    system_prompt: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {"llm": {"prompt": self.prompt, "output": self.output}}
        if self.system_prompt:
            d["llm"]["system_prompt"] = self.system_prompt
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "LLMStep":
        llm_data = data.get("llm", data)
        return cls(
            prompt=llm_data["prompt"],
            output=llm_data["output"],
            system_prompt=llm_data.get("system_prompt"),
        )


@dataclass
class ForeachStep:
    """Iterate over a list."""
    items: str  # "$var" reference to list
    as_var: str  # Variable name for each item
    do: List["Step"]  # Steps to execute for each item
    collect: Optional[str] = None  # Variable to collect results

    def to_dict(self) -> Dict:
        d = {
            "foreach": self.items,
            "as": self.as_var,
            "do": [step_to_dict(s) for s in self.do],
        }
        if self.collect:
            d["collect"] = self.collect
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "ForeachStep":
        return cls(
            items=data["foreach"],
            as_var=data["as"],
            do=[step_from_dict(s) for s in data.get("do", [])],
            collect=data.get("collect"),
        )


@dataclass
class IfStep:
    """Conditional execution."""
    condition: str  # Expression to evaluate
    then: List["Step"]  # Steps if true
    else_steps: Optional[List["Step"]] = None  # Steps if false

    def to_dict(self) -> Dict:
        d = {
            "if": self.condition,
            "then": [step_to_dict(s) for s in self.then],
        }
        if self.else_steps:
            d["else"] = [step_to_dict(s) for s in self.else_steps]
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "IfStep":
        return cls(
            condition=data["if"],
            then=[step_from_dict(s) for s in data.get("then", [])],
            else_steps=[step_from_dict(s) for s in data["else"]] if "else" in data else None,
        )


@dataclass
class SetStep:
    """Set a variable value."""
    var: str  # Variable name (with $)
    value: Any  # Value or expression

    def to_dict(self) -> Dict:
        return {"set": self.var, "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict) -> "SetStep":
        return cls(var=data["set"], value=data["value"])


@dataclass
class AppendStep:
    """Append to a list variable."""
    list_var: str  # Variable name (with $)
    value: Any  # Value to append

    def to_dict(self) -> Dict:
        return {"append": self.list_var, "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict) -> "AppendStep":
        return cls(list_var=data["append"], value=data["value"])


@dataclass
class ReturnStep:
    """Return a value from the skill."""
    value: Any  # Value or "$var" reference

    def to_dict(self) -> Dict:
        return {"return": self.value}

    @classmethod
    def from_dict(cls, data: Dict) -> "ReturnStep":
        return cls(value=data["return"])


# Union type for all step types
Step = Union[ToolStep, LLMStep, ForeachStep, IfStep, SetStep, AppendStep, ReturnStep]


def step_to_dict(step: Step) -> Dict:
    """Convert a step to dictionary."""
    return step.to_dict()


def step_from_dict(data: Dict) -> Step:
    """Parse a step from dictionary."""
    if "tool" in data:
        return ToolStep.from_dict(data)
    elif "llm" in data:
        return LLMStep.from_dict(data)
    elif "foreach" in data:
        return ForeachStep.from_dict(data)
    elif "if" in data:
        return IfStep.from_dict(data)
    elif "set" in data:
        return SetStep.from_dict(data)
    elif "append" in data:
        return AppendStep.from_dict(data)
    elif "return" in data:
        return ReturnStep.from_dict(data)
    else:
        raise ValueError(f"Unknown step type: {data}")


@dataclass
class Skill:
    """A composite skill made of multiple steps."""
    name: str
    description: str
    parameters: List[SkillParameter]
    steps: List[Step]
    version: int = 1
    created_from_task: Optional[str] = None
    usage_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "steps": [step_to_dict(s) for s in self.steps],
            "version": self.version,
            "created_from_task": self.created_from_task,
            "usage_count": self.usage_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Skill":
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=[SkillParameter.from_dict(p) for p in data.get("parameters", [])],
            steps=[step_from_dict(s) for s in data.get("steps", [])],
            version=data.get("version", 1),
            created_from_task=data.get("created_from_task"),
            usage_count=data.get("usage_count", 0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Skill":
        return cls.from_dict(json.loads(json_str))

    def to_tool_definition(self) -> Dict:
        """Convert to OpenAI function tool format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

