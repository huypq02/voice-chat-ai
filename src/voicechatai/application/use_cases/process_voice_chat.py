from __future__ import annotations

from dataclasses import dataclass, field

from voicechatai.application.services.intent_service import IntentService
from voicechatai.application.services.llm_answer_service import LLMAnswerService
from voicechatai.application.services.rag_service import RAGDecision, RAGService
from voicechatai.domain.entities.conversation_state import ConversationState
from voicechatai.domain.ports.stt_port import STTPort
from voicechatai.domain.ports.tts_port import TTSPort

_REPLAY_PHRASE = "Are you still there?"


@dataclass(slots=True)
class VoiceChatResult:
    transcript: str
    answer: str
    audio_bytes: bytes        # TTS-synthesized answer as MP3
    decision: str             # "hit" | "clarify" | "fallback"
    intent: str               # "CONTINUE" | "REPLAY" | "STOP"
    top1_score: float = 0.0
    clarification_questions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProcessVoiceChat:
    stt: STTPort
    rag_service: RAGService
    llm_answer_service: LLMAnswerService
    tts: TTSPort
    intent_service: IntentService = field(default_factory=IntentService)

    def execute(self, audio_bytes: bytes, state: ConversationState) -> VoiceChatResult:
        # Silence fast-path: no audio → increment streak, check REPLAY threshold
        if not audio_bytes:
            state.record_silence()
            if self.intent_service.should_replay_on_silence(state.silence_streak):
                state.reset_silence()
                audio_out = self.tts.synthesize(_REPLAY_PHRASE)
                return VoiceChatResult(
                    transcript="",
                    answer=_REPLAY_PHRASE,
                    audio_bytes=audio_out,
                    decision="silence",
                    intent="REPLAY",
                )
            return VoiceChatResult(
                transcript="", answer="", audio_bytes=b"", decision="silence", intent="CONTINUE"
            )

        transcript = self.stt.transcribe(audio_bytes)

        # Empty transcript also counts as silence
        if not transcript.strip():
            state.record_silence()
            if self.intent_service.should_replay_on_silence(state.silence_streak):
                state.reset_silence()
                audio_out = self.tts.synthesize(_REPLAY_PHRASE)
                return VoiceChatResult(
                    transcript="",
                    answer=_REPLAY_PHRASE,
                    audio_bytes=audio_out,
                    decision="silence",
                    intent="REPLAY",
                )
            return VoiceChatResult(
                transcript="", answer="", audio_bytes=b"", decision="silence", intent="CONTINUE"
            )

        state.record_user_turn(transcript)
        decision = self.rag_service.evaluate_faq_decision(transcript)
        answer, clarification_questions, intent = self._resolve_answer(transcript, decision, state)

        if intent == "STOP":
            closing = self.llm_answer_service.generate_closing()
            audio_out = self.tts.synthesize(closing)
            state.record_assistant_turn(closing)
            return VoiceChatResult(
                transcript=transcript,
                answer=closing,
                audio_bytes=audio_out,
                decision=decision.decision,
                intent="STOP",
                top1_score=decision.top1_score,
            )

        state.record_assistant_turn(answer)
        audio_out = self.tts.synthesize(answer)
        return VoiceChatResult(
            transcript=transcript,
            answer=answer,
            audio_bytes=audio_out,
            decision=decision.decision,
            intent=intent,
            top1_score=decision.top1_score,
            clarification_questions=clarification_questions,
        )

    def _resolve_answer(
        self, transcript: str, decision: RAGDecision, state: ConversationState
    ) -> tuple[str, list[str], str]:
        if decision.decision == "hit":
            return decision.top_candidate.answer, [], "CONTINUE"

        if decision.decision == "clarify":
            questions = [c.question for c in decision.clarification_candidates if c.question.strip()]
            return _build_clarification_prompt(questions), questions, "CONTINUE"

        # fallback: LLM generates answer and self-reports intent
        raw = self.llm_answer_service.generate_answer_with_intent(
            transcript, decision.top_candidate, state.history
        )
        answer, intent = self.intent_service.parse(raw)
        return answer, [], intent


def _build_clarification_prompt(questions: list[str]) -> str:
    if not questions:
        return "Could you please clarify your question?"
    formatted = ", or ".join(f'"{q}"' for q in questions)
    return f"Did you mean: {formatted}?"
