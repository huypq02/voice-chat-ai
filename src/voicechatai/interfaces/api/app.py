from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (four levels up from this file: src/voicechatai/interfaces/api/)
load_dotenv(Path(__file__).parents[4] / ".env")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from voicechatai.application.services.intent_service import IntentService
from voicechatai.application.services.llm_answer_service import LLMAnswerService
from voicechatai.application.services.rag_service import RAGService
from voicechatai.application.services.retrieval_service import RetrievalService
from voicechatai.application.use_cases.process_voice_chat import ProcessVoiceChat
from voicechatai.domain.ports.stt_port import STTError
from voicechatai.infrastructure.embedding.gemini_embedder import GeminiEmbedder
from voicechatai.infrastructure.llm.gemini_llm import GeminiLLM
from voicechatai.infrastructure.stt.whisper_stt import WhisperSTT
from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS
from voicechatai.infrastructure.vector_store.chroma_store import ChromaStore
from voicechatai.interfaces.api import routes, websocket_routes

logger = logging.getLogger(__name__)

_FAQS_PATH = Path(__file__).parents[4] / "data" / "faqs" / "faqs.json"


def _index_faqs_if_empty(embedder: GeminiEmbedder, vector_store: ChromaStore) -> None:
    """Embed and upsert all FAQs into ChromaDB if the collection is empty."""
    if not _FAQS_PATH.exists():
        logger.warning("FAQs file not found at %s — skipping indexing.", _FAQS_PATH)
        return

    count = vector_store._collection.count()
    if count > 0:
        logger.info("ChromaDB already has %d entries — skipping re-index.", count)
        return

    faqs = json.loads(_FAQS_PATH.read_text(encoding="utf-8"))
    ids, documents, metadatas, embeddings = [], [], [], []

    for faq in faqs:
        text = faq["question"]
        vector = embedder.embed_text(text)
        ids.append(faq["id"])
        documents.append(text)
        metadatas.append({"question": faq["question"], "answer": faq["answer"]})
        embeddings.append(vector)

    vector_store.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    logger.info("Indexed %d FAQs into ChromaDB.", len(faqs))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application with real adapters."""

    stt_adapter: WhisperSTT | None = None
    stt_boot_error: str | None = None
    rag_service: RAGService | None = None
    rag_boot_error: str | None = None
    embedder: GeminiEmbedder | None = None
    vector_store: ChromaStore | None = None
    process_voice_chat: ProcessVoiceChat | None = None
    voice_chat_boot_error: str | None = None

    # STT adapter
    try:
        stt_adapter = WhisperSTT()
    except STTError as exc:
        stt_boot_error = str(exc)

    # Retrieval stack
    try:
        embedder = GeminiEmbedder()
        vector_store = ChromaStore()
        retrieval_service = RetrievalService(embedder=embedder, vector_store=vector_store)
        rag_service = RAGService(retrieval_service=retrieval_service)
    except Exception as exc:  # noqa: BLE001
        rag_boot_error = str(exc)

    # Full pipeline
    try:
        if stt_adapter is None:
            raise RuntimeError(stt_boot_error or "STT adapter failed to initialize.")
        if rag_service is None:
            raise RuntimeError(rag_boot_error or "RAG service failed to initialize.")

        llm = GeminiLLM()
        tts = GeminiTTS()
        llm_answer_service = LLMAnswerService(llm=llm)
        process_voice_chat = ProcessVoiceChat(
            stt=stt_adapter,
            rag_service=rag_service,
            llm_answer_service=llm_answer_service,
            tts=tts,
            intent_service=IntentService(),
        )
    except Exception as exc:  # noqa: BLE001
        voice_chat_boot_error = str(exc)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Auto-index FAQs on startup if ChromaDB is empty
        if embedder is not None and vector_store is not None:
            try:
                _index_faqs_if_empty(embedder, vector_store)
            except Exception as exc:  # noqa: BLE001
                logger.error("FAQ indexing failed: %s", exc)
        yield

    app = FastAPI(
        title="Voice Chat AI",
        description="Real-time voice assistant with Whisper STT, FAQ retrieval, LLM fallback, and TTS.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.rag_service = rag_service
    app.state.rag_boot_error = rag_boot_error
    app.state.process_voice_chat = process_voice_chat
    app.state.voice_chat_boot_error = voice_chat_boot_error

    routes.register_routes(app)
    websocket_routes.register_ws_routes(app)

    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()
