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

_SYSTEM_PROMPT_WITH_INTENT = """\
You are a helpful customer support voice assistant.
Use the FAQ context below to answer the user's question concisely and accurately.
If the context does not contain enough information, say so honestly.

Context:
{context}

{history_section}

## Conversation Intent Rules
After your response, append on the very last line:
<intent>CONTINUE</intent>

Replace CONTINUE with:
- STOP  if you asked the user a direct question and they have not given a meaningful \
answer after 3 or more exchanges, OR the user has been silent/meaningless 2+ times in a row
- REPLAY  if the user's last message contained only filler words \
(e.g. "umm", "uh", "yeah yeah", "okay okay") with no real content
- CONTINUE  for all other cases\
"""

_CLOSING_PROMPT = (
    "The conversation is ending. "
    "Generate a warm, brief closing statement in 1-2 sentences. "
    "Do not append an <intent> tag."
)


@dataclass(slots=True)
class LLMAnswerService:
    """Generates a free-form answer for queries that miss the FAQ threshold (MISS path)."""

    llm: LLMPort

    def generate_answer(self, query: str, top_candidate: FAQEntry | None) -> str:
        context = self._build_context(top_candidate)
        system_prompt = _SYSTEM_PROMPT.format(context=context)
        return self.llm.generate(system_prompt=system_prompt, user_message=query)

    def generate_answer_with_intent(
        self,
        query: str,
        top_candidate: FAQEntry | None,
        history: list[tuple[str, str]],
    ) -> str:
        context = self._build_context(top_candidate)
        history_section = _build_history_section(history)
        system_prompt = _SYSTEM_PROMPT_WITH_INTENT.format(
            context=context,
            history_section=history_section,
        )
        return self.llm.generate(system_prompt=system_prompt, user_message=query)

    def generate_closing(self) -> str:
        return self.llm.generate(
            system_prompt="You are a helpful voice assistant.",
            user_message=_CLOSING_PROMPT,
        )

    @staticmethod
    def _build_context(top_candidate: FAQEntry | None) -> str:
        if top_candidate is None:
            return "(no relevant FAQ entries found)"
        return f"Q: {top_candidate.question}\nA: {top_candidate.answer}"


def _build_history_section(history: list[tuple[str, str]]) -> str:
    if not history:
        return ""
    lines = "\n".join(f"{role}: {text}" for role, text in history[-6:])
    return f"Conversation history:\n{lines}"
