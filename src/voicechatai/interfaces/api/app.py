from __future__ import annotations

from fastapi import FastAPI

from voicechatai.application.services.rag_service import RAGService
from voicechatai.application.services.retrieval_service import RetrievalService
from voicechatai.application.use_cases.ingest_documents import FAQSampleIndexer
from voicechatai.application.use_cases.process_voice_query import ProcessVoiceQuery
from voicechatai.domain.ports.stt_port import STTError
from voicechatai.infrastructure.embedding.minilm_embedder import MiniLMEmbedder
from voicechatai.infrastructure.stt.whisper_stt import WhisperSTT
from voicechatai.infrastructure.vector_store.chroma_store import ChromaStore


def create_app() -> FastAPI:
	"""Create and configure the FastAPI application with real adapters."""
	app = FastAPI(
		title="Voice Chat AI",
		description="Real-time voice assistant with Whisper STT, retrieval, and TTS.",
		version="0.1.0",
	)

	process_voice_query: ProcessVoiceQuery | None = None
	stt_boot_error: str | None = None
	rag_service: RAGService | None = None
	rag_boot_error: str | None = None
	faq_indexer: FAQSampleIndexer | None = None

	# Initialize STT at startup, but keep API booting even if local deps are missing.
	try:
		stt_adapter = WhisperSTT()
		process_voice_query = ProcessVoiceQuery(stt=stt_adapter)
	except STTError as exc:
		stt_boot_error = str(exc)

	# Initialize retrieval + decision stack, but keep API available if deps are missing.
	try:
		embedder = MiniLMEmbedder()
		vector_store = ChromaStore()
		retrieval_service = RetrievalService(embedder=embedder, vector_store=vector_store)
		rag_service = RAGService(retrieval_service=retrieval_service)
		faq_indexer = FAQSampleIndexer(embedder=embedder, vector_store=vector_store)
	except Exception as exc:  # noqa: BLE001 - boot should not fail hard on optional stack
		rag_boot_error = str(exc)

	# Store use cases in app state for route injection.
	app.state.process_voice_query = process_voice_query
	app.state.stt_boot_error = stt_boot_error
	app.state.rag_service = rag_service
	app.state.rag_boot_error = rag_boot_error
	app.state.faq_indexer = faq_indexer

	# Mount routes.
	from voicechatai.interfaces.api import routes

	routes.register_routes(app)

	return app


app = create_app()

