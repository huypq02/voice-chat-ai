from __future__ import annotations

import os

from fastapi import FastAPI

from voicechatai.application.services.llm_answer_service import LLMAnswerService
from voicechatai.application.services.rag_service import RAGService
from voicechatai.application.services.retrieval_service import RetrievalService
from voicechatai.application.use_cases.process_voice_chat import ProcessVoiceChat
from voicechatai.domain.ports.stt_port import STTError
from voicechatai.infrastructure.embedding.openai_embedder import OpenAIEmbedder
from voicechatai.infrastructure.llm.openai_llm import OpenAILLM
from voicechatai.infrastructure.stt.whisper_stt import WhisperSTT
from voicechatai.infrastructure.tts.elevenlabs_tts import ElevenLabsTTS
from voicechatai.infrastructure.vector_store.chroma_store import ChromaStore
from voicechatai.interfaces.api import routes, websocket_routes


def create_app() -> FastAPI:
	"""Create and configure the FastAPI application with real adapters."""
	app = FastAPI(
		title="Voice Chat AI",
		description="Real-time voice assistant with Whisper STT, FAQ retrieval, LLM fallback, and TTS.",
		version="0.1.0",
	)

	stt_adapter: WhisperSTT | None = None
	stt_boot_error: str | None = None
	rag_service: RAGService | None = None
	rag_boot_error: str | None = None
	process_voice_chat: ProcessVoiceChat | None = None
	voice_chat_boot_error: str | None = None

	# STT adapter — used by the full pipeline.
	try:
		stt_adapter = WhisperSTT()
	except STTError as exc:
		stt_boot_error = str(exc)

	# Retrieval stack (embedding + vector store + RAG service).
	try:
		embedder = OpenAIEmbedder()
		vector_store = ChromaStore()
		retrieval_service = RetrievalService(embedder=embedder, vector_store=vector_store)
		rag_service = RAGService(retrieval_service=retrieval_service)
	except Exception as exc:  # noqa: BLE001
		rag_boot_error = str(exc)

	# Full pipeline: STT → RAG → HIT/MISS/clarify → TTS.
	# Requires STT, RAG, LLM (OpenAI), and TTS (ElevenLabs) to all be available.
	try:
		if stt_adapter is None:
			raise RuntimeError(stt_boot_error or "STT adapter failed to initialize.")
		if rag_service is None:
			raise RuntimeError(rag_boot_error or "RAG service failed to initialize.")

		llm = OpenAILLM()
		tts = ElevenLabsTTS()
		llm_answer_service = LLMAnswerService(llm=llm)
		process_voice_chat = ProcessVoiceChat(
			stt=stt_adapter,
			rag_service=rag_service,
			llm_answer_service=llm_answer_service,
			tts=tts,
		)
	except Exception as exc:  # noqa: BLE001
		voice_chat_boot_error = str(exc)

	app.state.rag_service = rag_service
	app.state.rag_boot_error = rag_boot_error
	app.state.process_voice_chat = process_voice_chat
	app.state.voice_chat_boot_error = voice_chat_boot_error

	# API keys for per-connection realtime STT adapters (created fresh each WS session).
	app.state.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY", "")
	app.state.assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY", "")

	routes.register_routes(app)
	websocket_routes.register_ws_routes(app)

	return app


app = create_app()
