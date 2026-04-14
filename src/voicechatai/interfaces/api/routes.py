from __future__ import annotations

import base64
import pathlib
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, field_validator

from voicechatai.application.services.rag_service import RAGDecision
from voicechatai.domain.ports.llm_port import LLMGenerationError
from voicechatai.domain.ports.stt_port import STTError
from voicechatai.domain.ports.tts_port import TTSError
from voicechatai.interfaces.presenters.response import FAQDecisionResponse, VoiceChatResponse


class TranscriptionRequest(BaseModel):
	"""Request body carrying base64-encoded raw audio bytes."""

	model_config = ConfigDict(
		json_schema_extra={"example": {"audio_bytes_b64": "UklGRi4AAAAS..."}}
	)

	audio_bytes_b64: str


class TranscribeAndRouteResponse(BaseModel):
	"""Response from the full STT → FAQ decision pipeline."""

	statuscode: int
	transcript: str | None = None
	faq_decision: FAQDecisionResponse | None = None
	error: str | None = None


class IndexFAQSamplesRequest(BaseModel):
	faq_json_path: str = "data/faqs/faqs.json"

	@field_validator("faq_json_path")
	@classmethod
	def _no_path_traversal(cls, v: str) -> str:
		# Reject absolute paths and any '..' components to prevent path traversal.
		p = pathlib.PurePosixPath(v)
		if p.is_absolute() or ".." in p.parts:
			raise ValueError("faq_json_path must be a relative path without '..'")
		return v


class IndexFAQSamplesResponse(BaseModel):
	statuscode: int
	indexed_count: int = 0
	preview: list[dict[str, str]] | None = None
	error: str | None = None


def _transcribe_or_error(app: FastAPI, audio_bytes: bytes) -> tuple[str | None, str | None]:
	process_voice_query = app.state.process_voice_query
	if process_voice_query is None:
		boot_error = app.state.stt_boot_error or "STT adapter is not initialized."
		return None, boot_error

	try:
		transcript = process_voice_query.execute(audio_bytes)
		return transcript, None
	except STTError as exc:
		return None, str(exc)


def _to_faq_decision_response(decision: RAGDecision) -> FAQDecisionResponse:
	answer = None
	if decision.top_candidate and decision.decision in {"hit", "clarify"}:
		answer = decision.top_candidate.answer

	clarification_questions: list[str] = []
	if decision.decision == "clarify":
		clarification_questions = [
			c.question for c in decision.clarification_candidates if c.question.strip()
		]

	return FAQDecisionResponse(
		decision=decision.decision,
		answer=answer,
		clarification_questions=clarification_questions,
		top1_score=decision.top1_score,
		top2_score=decision.top2_score,
		margin=decision.margin,
		effective_hit_threshold=decision.effective_hit_threshold,
	)


def _route_or_error(app: FastAPI, transcript: str) -> tuple[FAQDecisionResponse | None, str | None]:
	rag_service = app.state.rag_service
	if rag_service is None:
		boot_error = app.state.rag_boot_error or "RAG service is not initialized."
		return None, boot_error

	decision = rag_service.evaluate_faq_decision(transcript)
	return _to_faq_decision_response(decision), None


def register_routes(app: FastAPI) -> None:
	"""Register API routes for the voice chat pipeline."""

	@app.post("/transcribe-and-route", status_code=200)
	async def transcribe_and_route(request: TranscriptionRequest) -> TranscribeAndRouteResponse:
		"""Run the full pipeline: STT → embedding → Chroma query → FAQ decision."""
		try:
			audio_bytes = base64.b64decode(request.audio_bytes_b64)
		except ValueError as exc:
			return TranscribeAndRouteResponse(statuscode=400, error=f"Invalid audio input: {exc}")

		transcript, stt_error = _transcribe_or_error(app, audio_bytes)
		if stt_error is not None or transcript is None:
			statuscode = 503 if app.state.process_voice_query is None else 400
			return TranscribeAndRouteResponse(statuscode=statuscode, error=stt_error)

		faq_decision, route_error = _route_or_error(app, transcript)
		if route_error is not None:
			return TranscribeAndRouteResponse(statuscode=200, transcript=transcript, error=route_error)

		return TranscribeAndRouteResponse(statuscode=200, transcript=transcript, faq_decision=faq_decision)

	@app.post("/index-faq-samples", status_code=200)
	async def index_faq_samples(request: IndexFAQSamplesRequest) -> IndexFAQSamplesResponse:
		"""Index FAQ entries into Chroma (run once to populate the vector store)."""
		faq_indexer = app.state.faq_indexer
		if faq_indexer is None:
			boot_error = app.state.rag_boot_error or "FAQ indexer is not initialized."
			return IndexFAQSamplesResponse(statuscode=503, error=boot_error)

		try:
			indexed = faq_indexer.index_from_json(request.faq_json_path)
			return IndexFAQSamplesResponse(
				statuscode=200,
				indexed_count=indexed,
				preview=faq_indexer.last_index_preview or [],
			)
		except Exception as exc:  # noqa: BLE001 - keep endpoint resilient
			return IndexFAQSamplesResponse(statuscode=400, error=str(exc))

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
