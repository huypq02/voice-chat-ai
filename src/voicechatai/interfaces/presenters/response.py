from __future__ import annotations

from pydantic import BaseModel, Field


class FAQCandidateResponse(BaseModel):
	faq_id: str
	question: str
	answer: str
	score: float


class FAQDecisionResponse(BaseModel):
	decision: str
	answer: str | None = None
	clarification_questions: list[str] = Field(default_factory=list)
	top1_score: float = 0.0
	top2_score: float = 0.0
	margin: float = 0.0
	effective_hit_threshold: float = 0.0


