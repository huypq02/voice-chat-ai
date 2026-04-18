from __future__ import annotations

import base64

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from voicechatai.domain.ports.llm_port import LLMGenerationError
from voicechatai.domain.ports.stt_port import STTError
from voicechatai.domain.ports.tts_port import TTSError
from voicechatai.interfaces.presenters.response import VoiceChatResponse


class TranscriptionRequest(BaseModel):
	"""Request body carrying base64-encoded raw audio bytes."""

	model_config = ConfigDict(
		json_schema_extra={"example": {"audio_bytes_b64": "UklGRi4AAAAS..."}}
	)

	audio_bytes_b64: str


def register_routes(app: FastAPI) -> None:
	"""Register API routes for the voice chat pipeline."""

	@app.post("/voice-chat", status_code=200)
	async def voice_chat(request: TranscriptionRequest) -> VoiceChatResponse:
		"""
		Full pipeline: audio → STT → embed → Chroma → threshold → HIT(FAQ) / MISS(LLM) → TTS → audio.

		Returns the transcript, the answer text, and the TTS audio as base64-encoded MP3.
		"""
		process_voice_chat = app.state.process_voice_chat
		if process_voice_chat is None:
			boot_error = app.state.voice_chat_boot_error or "Voice chat pipeline is not initialized."
			return VoiceChatResponse(statuscode=503, error=boot_error)

		try:
			audio_bytes = base64.b64decode(request.audio_bytes_b64)
		except ValueError as exc:
			return VoiceChatResponse(statuscode=400, error=f"Invalid audio input: {exc}")

		try:
			result = process_voice_chat.execute(audio_bytes)
		except (STTError, LLMGenerationError, TTSError) as exc:
			return VoiceChatResponse(statuscode=500, error=str(exc))

		return VoiceChatResponse(
			statuscode=200,
			transcript=result.transcript,
			answer=result.answer,
			audio_b64=base64.b64encode(result.audio_bytes).decode(),
			decision=result.decision,
			top1_score=result.top1_score,
			clarification_questions=result.clarification_questions,
		)

	@app.get("/health")
	async def health_check() -> dict[str, str]:
		"""Health check endpoint."""
		return {"status": "ok", "service": "voice-chat-ai"}
