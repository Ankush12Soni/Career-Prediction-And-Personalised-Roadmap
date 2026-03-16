"""Pydantic models for request / response validation."""

from pydantic import BaseModel, Field
from typing import Optional

class QuizScores(BaseModel):
    python_interest:         int = Field(ge=0, le=10)
    java_interest:           int = Field(ge=0, le=10)
    javascript_interest:     int = Field(ge=0, le=10)
    cpp_interest:            int = Field(ge=0, le=10)
    mobile_dev_interest:     int = Field(ge=0, le=10)
    linear_algebra:          int = Field(ge=0, le=10)
    statistics_probability:  int = Field(ge=0, le=10)
    discrete_math:           int = Field(ge=0, le=10)
    calculus:                int = Field(ge=0, le=10)
    system_design_interest:  int = Field(ge=0, le=10)
    networking_interest:     int = Field(ge=0, le=10)
    security_interest:       int = Field(ge=0, le=10)
    cloud_interest:          int = Field(ge=0, le=10)
    design_interest:         int = Field(ge=0, le=10)
    product_thinking:        int = Field(ge=0, le=10)
    communication_skill:     int = Field(ge=0, le=10)
    leadership:              int = Field(ge=0, le=10)
    research_interest:       int = Field(ge=0, le=10)
    analytical_skill:        int = Field(ge=0, le=10)
    curiosity_learning:      int = Field(ge=0, le=10)

class RecommendRequest(BaseModel):
    domain: str
    quiz_scores: QuizScores
    top_n: Optional[int] = 5

class SkillGapRequest(BaseModel):
    career: str
    student_skills: list[str]

class RoadmapRequest(BaseModel):
    career: str
    missing_skills: list[str]
