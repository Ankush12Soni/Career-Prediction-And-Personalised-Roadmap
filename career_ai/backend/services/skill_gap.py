"""
Skill Gap Analysis Service
Compares student skills against career requirements and produces
a readiness score + detailed breakdown.
"""

import os
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "career_knowledge_dataset.csv")

_df = None

def _load():
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)

def analyze_gap(career_name: str, student_skills: list[str]) -> dict:
    """
    Returns a full skill gap report for a specific career.

    Args:
        career_name: exact career title from the knowledge base.
        student_skills: list of skill keywords the student has.

    Returns dict with:
        career, required_skills, matched_skills, missing_skills,
        readiness_score (0-100), roadmap_steps
    """
    _load()

    row = _df[_df["career"].str.lower() == career_name.lower()]
    if row.empty:
        return {"error": f"Career '{career_name}' not found in knowledge base."}

    row = row.iloc[0]
    required = [s.strip() for s in str(row["skills"]).split(",")]

    student_lower  = set(s.lower() for s in student_skills)
    required_lower = [s.lower() for s in required]

    matched = [r for r in required if r.lower() in student_lower]
    missing = [r for r in required if r.lower() not in student_lower]

    readiness = round(len(matched) / len(required) * 100, 1) if required else 0

    # Parse roadmap into ordered steps
    raw_roadmap = str(row["roadmap"])
    steps = [s.strip() for s in raw_roadmap.replace("→", "->").split("->")]

    # Mark which steps the student can skip (already has skill)
    step_status = []
    for step in steps:
        completed = any(skill.lower() in step.lower() for skill in student_skills)
        step_status.append({"step": step, "completed": completed})

    return {
        "career":          row["career"],
        "domain":          row["domain"],
        "required_skills": required,
        "matched_skills":  matched,
        "missing_skills":  missing,
        "readiness_score": readiness,
        "roadmap_steps":   step_status,
    }

def bulk_analyze(careers: list[str], student_skills: list[str]) -> list[dict]:
    """Analyze skill gap for multiple careers at once."""
    return [analyze_gap(c, student_skills) for c in careers]
