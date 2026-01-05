"""
Skill Library
=============

Persistent storage and management of skills.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.logger import get_logger
from .schema import Skill

logger = get_logger(__name__)


class SkillLibrary:
    """Manages persistent and temporary skills."""
    
    def __init__(self, storage_path: Path = Path("./skills")):
        """
        Initialize the skill library.
        
        Args:
            storage_path: Directory to store skill JSON files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.permanent_skills: Dict[str, Skill] = {}  # Persisted to disk
        self.temporary_skills: Dict[str, Skill] = {}  # Task-specific, not persisted
        
        self._load_all_permanent()
    
    def _load_all_permanent(self):
        """Load all permanent skills from disk."""
        for skill_file in self.storage_path.glob("*.json"):
            try:
                skill = self._load_skill_file(skill_file)
                self.permanent_skills[skill.name] = skill
                logger.debug(f"[SkillLibrary] Loaded skill: {skill.name}")
            except Exception as e:
                logger.warning(f"[SkillLibrary] Failed to load {skill_file}: {e}")
        
        logger.info(f"[SkillLibrary] Loaded {len(self.permanent_skills)} permanent skills")
    
    def _load_skill_file(self, path: Path) -> Skill:
        """Load a skill from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Skill.from_dict(data)
    
    def _save_skill_file(self, skill: Skill):
        """Save a skill to a JSON file."""
        path = self.storage_path / f"{skill.name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(skill.to_dict(), f, indent=2, ensure_ascii=False)
        logger.debug(f"[SkillLibrary] Saved skill: {skill.name}")
    
    def get(self, name: str) -> Optional[Skill]:
        """
        Get a skill by name.
        
        Checks temporary skills first, then permanent.
        """
        return self.temporary_skills.get(name) or self.permanent_skills.get(name)
    
    def is_skill(self, name: str) -> bool:
        """Check if a name corresponds to a registered skill."""
        return name in self.temporary_skills or name in self.permanent_skills
    
    def register_permanent(self, skill: Skill):
        """Register a skill as permanent (persisted to disk)."""
        self.permanent_skills[skill.name] = skill
        self._save_skill_file(skill)
        logger.info(f"[SkillLibrary] Registered permanent skill: {skill.name}")
    
    def register_temporary(self, skill: Skill):
        """Register a skill as temporary (task-specific)."""
        self.temporary_skills[skill.name] = skill
        logger.info(f"[SkillLibrary] Registered temporary skill: {skill.name}")
    
    def promote_to_permanent(self, skill_name: str):
        """Promote a temporary skill to permanent."""
        skill = self.temporary_skills.get(skill_name)
        if skill:
            del self.temporary_skills[skill_name]
            self.register_permanent(skill)
            logger.info(f"[SkillLibrary] Promoted skill to permanent: {skill_name}")
    
    def remove_temporary(self, skill_name: str):
        """Remove a temporary skill."""
        if skill_name in self.temporary_skills:
            del self.temporary_skills[skill_name]
            logger.debug(f"[SkillLibrary] Removed temporary skill: {skill_name}")
    
    def clear_temporary(self):
        """Clear all temporary skills."""
        count = len(self.temporary_skills)
        self.temporary_skills.clear()
        logger.debug(f"[SkillLibrary] Cleared {count} temporary skills")
    
    def increment_usage(self, skill_name: str):
        """Increment the usage count for a skill."""
        skill = self.get(skill_name)
        if skill:
            skill.usage_count += 1
            if skill_name in self.permanent_skills:
                self._save_skill_file(skill)
    
    def list_all(self) -> List[str]:
        """List all skill names (permanent + temporary)."""
        all_names = set(self.permanent_skills.keys()) | set(self.temporary_skills.keys())
        return sorted(all_names)
    
    def list_permanent(self) -> List[str]:
        """List permanent skill names."""
        return sorted(self.permanent_skills.keys())
    
    def list_temporary(self) -> List[str]:
        """List temporary skill names."""
        return sorted(self.temporary_skills.keys())
    
    def get_all_skills(self) -> List[Skill]:
        """Get all skills (permanent + temporary)."""
        all_skills = {**self.permanent_skills, **self.temporary_skills}
        return list(all_skills.values())
    
    def get_tool_definitions(self) -> List[Dict]:
        """Get all skills as OpenAI tool definitions."""
        return [skill.to_tool_definition() for skill in self.get_all_skills()]
    
    def get_skill_summaries(self) -> List[Dict]:
        """Get brief summaries of all skills for display."""
        return [
            {"name": skill.name, "description": skill.description}
            for skill in self.get_all_skills()
        ]
    
    def delete_permanent(self, skill_name: str):
        """Delete a permanent skill (from memory and disk)."""
        if skill_name in self.permanent_skills:
            del self.permanent_skills[skill_name]
            path = self.storage_path / f"{skill_name}.json"
            if path.exists():
                path.unlink()
            logger.info(f"[SkillLibrary] Deleted permanent skill: {skill_name}")

