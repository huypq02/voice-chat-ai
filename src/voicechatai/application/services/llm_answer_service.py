from __future__ import annotations

from dataclasses import dataclass

from voicechatai.domain.entities.faq_entry import FAQEntry
from voicechatai.domain.ports.llm_port import LLMPort

_SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Use the FAQ context below to answer the user's question concisely and accurately. "
    "If the context does not contain enough information, say so honestly.\n\n"
    "Context:\n{context}"
)


@dataclass(slots=True)
class LLMAnswerService:
    """Generates a free-form answer for queries that miss the FAQ threshold (MISS path)."""

    llm: LLMPort

    def generate_answer(self, query: str, top_candidate: FAQEntry | None) -> str:
        context = self._build_context(top_candidate)
        system_prompt = _SYSTEM_PROMPT.format(context=context)
        return self.llm.generate(system_prompt=system_prompt, user_message=query)

    @staticmethod
    def _build_context(top_candidate: FAQEntry | None) -> str:
        if top_candidate is None:
            return "(no relevant FAQ entries found)"
        return f"Q: {top_candidate.question}\nA: {top_candidate.answer}"
