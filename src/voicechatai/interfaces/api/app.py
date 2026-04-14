from __future__ import annotations

from fastapi import FastAPI

from voicechatai.application.services.llm_answer_service import LLMAnswerService
from voicechatai.application.services.rag_service import RAGService
from voicechatai.application.services.retrieval_service import RetrievalService
from voicechatai.application.use_cases.ingest_documents import FAQSampleIndexer
from voicechatai.application.use_cases.process_voice_chat import ProcessVoiceChat
from voicechatai.application.use_cases.process_voice_query import ProcessVoiceQuery
from voicechatai.domain.ports.stt_port import STTError
from voicechatai.infrastructure.embedding.minilm_embedder import MiniLMEmbedder
from voicechatai.infrastructure.llm.openai_llm import OpenAILLM
from voicechatai.infrastructure.stt.whisper_stt import WhisperSTT
from voicechatai.infrastructure.tts.elevenlabs_tts import ElevenLabsTTS
from voicechatai.infrastructure.vector_store.chroma_store import ChromaStore
from voicechatai.interfaces.api import routes


def create_app() -> FastAPI:
	"""Create and configure the FastAPI application with real adapters."""
	app = FastAPI(
		title="Voice Chat AI",
		description="Real-time voice assistant with Whisper STT, FAQ retrieval, LLM fallback, and TTS.",
		version="0.1.0",
	)

	stt_adapter: WhisperSTT | None = None
	process_voice_query: ProcessVoiceQuery | None = None
	stt_boot_error: str | None = None
	rag_service: RAGService | None = None
	rag_boot_error: str | None = None
	faq_indexer: FAQSampleIndexer | None = None
	process_voice_chat: ProcessVoiceChat | None = None
	voice_chat_boot_error: str | None = None

	# STT adapter — used by both the partial pipeline and the full pipeline.
	try:
		stt_adapter = WhisperSTT()
		process_voice_query = ProcessVoiceQuery(stt=stt_adapter)
	except STTError as exc:
		stt_boot_error = str(exc)

	# Retrieval stack (embedding + vector store + RAG service).
	try:
		embedder = MiniLMEmbedder()
		vector_store = ChromaStore()
		retrieval_service = RetrievalService(embedder=embedder, vector_store=vector_store)
		rag_service = RAGService(retrieval_service=retrieval_service)
		faq_indexer = FAQSampleIndexer(embedder=embedder, vector_store=vector_store)
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

	app.state.process_voice_query = process_voice_query
	app.state.stt_boot_error = stt_boot_error
	app.state.rag_service = rag_service
	app.state.rag_boot_error = rag_boot_error
	app.state.faq_indexer = faq_indexer
	app.state.process_voice_chat = process_voice_chat
	app.state.voice_chat_boot_error = voice_chat_boot_error

	routes.register_routes(app)

	return app


app = create_app()
