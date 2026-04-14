from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from voicechatai.application.services.retrieval_service import RetrievalService
from voicechatai.domain.entities.faq_entry import FAQEntry

DecisionType = Literal["hit", "clarify", "fallback"]


@dataclass(slots=True)
class RAGDecisionPolicy:
	"""Threshold and ambiguity policy for FAQ retrieval decisions."""

	hit_threshold: float = 0.76
	ambiguity_threshold: float = 0.65
	min_margin: float = 0.06
	short_query_token_max: int = 3
	short_query_hit_boost: float = 0.03
	clarification_top_n: int = 2


@dataclass(slots=True)
class RAGDecision:
	"""Result of evaluating a query against FAQ retrieval candidates."""

	decision: DecisionType
	top_candidate: FAQEntry | None = None
	clarification_candidates: list[FAQEntry] = field(default_factory=list)
	top1_score: float = 0.0
	top2_score: float = 0.0
	margin: float = 0.0
	effective_hit_threshold: float = 0.0


@dataclass(slots=True)
class RAGService:
	"""Applies retrieval and threshold policy for FAQ routing."""

	retrieval_service: RetrievalService
	policy: RAGDecisionPolicy = field(default_factory=RAGDecisionPolicy)

	def evaluate_faq_decision(self, query_text: str) -> RAGDecision:
		candidates = self.retrieval_service.retrieve_faq_candidates(query_text)
		if not candidates:
			return RAGDecision(decision="fallback")

		top1 = candidates[0]
		top2 = candidates[1] if len(candidates) > 1 else None
		top1_score = top1.score
		top2_score = top2.score if top2 is not None else 0.0
		margin = top1_score - top2_score if top2 is not None else top1_score

		effective_hit_threshold = self.policy.hit_threshold
		if self._token_count(query_text) <= self.policy.short_query_token_max:
			effective_hit_threshold += self.policy.short_query_hit_boost

		if top1_score >= effective_hit_threshold and margin >= self.policy.min_margin:
			return RAGDecision(
				decision="hit",
				top_candidate=top1,
				top1_score=top1_score,
				top2_score=top2_score,
				margin=margin,
				effective_hit_threshold=effective_hit_threshold,
			)

		if top1_score >= self.policy.ambiguity_threshold:
			return RAGDecision(
				decision="clarify",
				top_candidate=top1,
				clarification_candidates=candidates[: self.policy.clarification_top_n],
				top1_score=top1_score,
				top2_score=top2_score,
				margin=margin,
				effective_hit_threshold=effective_hit_threshold,
			)

		return RAGDecision(
			decision="fallback",
			top_candidate=top1,
			top1_score=top1_score,
			top2_score=top2_score,
			margin=margin,
			effective_hit_threshold=effective_hit_threshold,
		)

	@staticmethod
	def _token_count(text: str) -> int:
		# str.split() with no args already strips whitespace and drops empty tokens.
		return len(text.split())


