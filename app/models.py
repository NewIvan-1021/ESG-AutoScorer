from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class SubCriterionScore(BaseModel):
    title: str
    max_score: float
    score: Optional[float] = None
    rationale: Optional[str] = None

class CriterionScore(BaseModel):
    title: str
    max_score: float
    score: Optional[float] = None
    sub_criteria: List[SubCriterionScore] = Field(default_factory=list)

class SectionScore(BaseModel):
    title: str
    max_score: float
    score: Optional[float] = None
    ai_comment: Optional[str] = None
    criteria: List[CriterionScore] = Field(default_factory=list)

class BreakdownItem(BaseModel):
    id: str
    raw_score: Optional[float] = 0.0
    raw_max_score: Optional[float] = 0.0
    ai_comment: Optional[str] = None
    sections: List[SectionScore] = Field(default_factory=list)

class TotalsScore(BaseModel):
    report: Optional[float] = None
    media: Optional[float] = None
    final: Optional[float] = None

class ScoringResult(BaseModel):
    company: Optional[str] = None
    overview_comment: Optional[str] = None
    strengths: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    improvements: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    breakdown: List[BreakdownItem] = Field(default_factory=list)
    totals: Optional[TotalsScore] = None
