"""
Static Task Split for Evolution Experiments
============================================

Task sets for evolution experiments.
"""

# Full task set (30 tasks)
FULL_TASKS = [
    # Hard (6 tasks - 0/4 runs)
    "file_context__duplicates_searching",
    "desktop_template__budget_computation",
    "votenet__requirements_writing",
    "file_property__time_classification",
    "papers__author_folders",
    "threestudio__requirements_completion",
    
    # Medium (8 tasks - 1-3/4 runs)
    "folder_structure__structure_mirror",
    "folder_structure__structure_analysis",
    "threestudio__output_analysis",
    "votenet__debugging",
    "desktop_template__contact_information",
    "desktop_template__file_arrangement",
    "file_context__uppercase",
    "legal_document__solution_tracing",
    
    # Easy (16 tasks - 4/4 runs)
    "desktop__music_report",
    "desktop__project_management",
    "desktop__timeline_extraction",
    "file_context__file_merging",
    "file_context__file_splitting",
    "file_context__pattern_matching",
    "file_property__size_classification",
    "legal_document__dispute_review",
    "legal_document__individual_comments",
    "papers__find_math_paper",
    "papers__organize_legacy_papers",
    "student_database__duplicate_name",
    "student_database__english_talent",
    "student_database__gradebased_score",
    "threestudio__code_locating",
    "votenet__dataset_comparison",
]

# Quick test set - tasks likely to use skills (copy/move)
EVOLUTION_TASKS = [
    "papers__author_folders",          # Likely: copy/move files to author folders
    "file_context__file_merging",      # Likely: read multiple + write combined
]


def get_evolution_tasks() -> list[str]:
    """Get tasks for the evolution experiment."""
    return EVOLUTION_TASKS.copy()


def get_full_tasks() -> list[str]:
    """Get all tasks."""
    return FULL_TASKS.copy()


def get_all_evolution_tasks() -> list[str]:
    """Get all tasks involved in the evolution experiment."""
    return EVOLUTION_TASKS.copy()
