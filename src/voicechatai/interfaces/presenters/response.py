from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceChatResponse(BaseModel):
	"""Full pipeline response: transcript + answer text + TTS audio."""

	statuscode: int
	transcript: str | None = None
	answer: str | None = None
	audio_b64: str | None = None        # base64-encoded MP3 from TTS
	decision: str | None = None         # "hit" | "clarify" | "fallback"
	top1_score: float = 0.0
	clarification_questions: list[str] = Field(default_factory=list)
	error: str | None = None


