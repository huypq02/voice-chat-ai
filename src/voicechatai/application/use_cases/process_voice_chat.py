from __future__ import annotations

from dataclasses import dataclass, field

from voicechatai.application.services.llm_answer_service import LLMAnswerService
from voicechatai.application.services.rag_service import RAGDecision, RAGService
from voicechatai.domain.ports.stt_port import STTPort
from voicechatai.domain.ports.tts_port import TTSPort


@dataclass(slots=True)
class VoiceChatResult:
    """Outcome of one full voice pipeline execution."""

    transcript: str
    answer: str
    audio_bytes: bytes        # TTS-synthesized answer as MP3
    decision: str             # "hit" | "clarify" | "fallback"
    top1_score: float = 0.0
    clarification_questions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProcessVoiceChat:
    """Orchestrates the full pipeline: STT → RAG → HIT/MISS/clarify → TTS."""

    stt: STTPort
    rag_service: RAGService
    llm_answer_service: LLMAnswerService
    tts: TTSPort

    def execute(self, audio_bytes: bytes) -> VoiceChatResult:
        # Step 1: Speech → Text
        transcript = self.stt.transcribe(audio_bytes)

        # Step 2: Embed + query Chroma + threshold decision
        decision = self.rag_service.evaluate_faq_decision(transcript)

        # Step 3: Resolve answer from decision
        answer, clarification_questions = self._resolve_answer(transcript, decision)

        # Step 4: Text → Audio
        audio_out = self.tts.synthesize(answer)

        return VoiceChatResult(
            transcript=transcript,
            answer=answer,
            audio_bytes=audio_out,
            decision=decision.decision,
            top1_score=decision.top1_score,
            clarification_questions=clarification_questions,
        )

    def _resolve_answer(self, transcript: str, decision: RAGDecision) -> tuple[str, list[str]]:
        if decision.decision == "hit":
            return decision.top_candidate.answer, []

        if decision.decision == "clarify":
            questions = [c.question for c in decision.clarification_candidates if c.question.strip()]
            return _build_clarification_prompt(questions), questions

        # fallback/miss: generate answer with LLM using top candidate as context
        answer = self.llm_answer_service.generate_answer(transcript, decision.top_candidate)
        return answer, []


def _build_clarification_prompt(questions: list[str]) -> str:
    """Build a spoken clarification question from a list of FAQ candidates."""
    if not questions:
        return "Could you please clarify your question?"
    formatted = ", or ".join(f'"{q}"' for q in questions)
    return f"Did you mean: {formatted}?"
