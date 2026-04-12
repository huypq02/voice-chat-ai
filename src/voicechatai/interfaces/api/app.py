from __future__ import annotations

from fastapi import FastAPI

from voicechatai.application.use_cases.process_voice_query import ProcessVoiceQuery
from voicechatai.domain.ports.stt_port import STTError
from voicechatai.infrastructure.stt.whisper_stt import WhisperSTT


def create_app() -> FastAPI:
	"""Create and configure the FastAPI application with real adapters."""
	app = FastAPI(
		title="Voice Chat AI",
		description="Real-time voice assistant with Whisper STT, retrieval, and TTS.",
		version="0.1.0",
	)

	process_voice_query: ProcessVoiceQuery | None = None
	stt_boot_error: str | None = None

	# Initialize STT at startup, but keep API booting even if local deps are missing.
	try:
		stt_adapter = WhisperSTT()
		process_voice_query = ProcessVoiceQuery(stt=stt_adapter)
	except STTError as exc:
		stt_boot_error = str(exc)

	# Store use cases in app state for route injection.
	app.state.process_voice_query = process_voice_query
	app.state.stt_boot_error = stt_boot_error

	# Mount routes.
	from voicechatai.interfaces.api import routes

	routes.register_routes(app)

	return app


app = create_app()

