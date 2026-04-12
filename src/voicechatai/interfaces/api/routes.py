from __future__ import annotations

import base64
import pathlib
import struct
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from voicechatai.domain.ports.stt_port import STTError


class TranscriptionRequest(BaseModel):
	"""Request body for transcription endpoint."""
	audio_bytes_b64: str

	class Config:
		json_schema_extra = {
			"example": {
				"audio_bytes_b64": "UklGRi4AAAAS...",  # Base64-encoded audio bytes
			}
		}


class TranscriptionResponse(BaseModel):
	"""Response from transcription endpoint."""

	statuscode: int
	transcript: str | None = None
	error: str | None = None


PARTIAL_EVERY_CHUNKS = 4
SILENCE_END_STREAK = 3
MIN_AUDIO_BYTES_FOR_TRANSCRIBE = 512


@dataclass
class StreamState:
	buffer: bytearray
	chunk_count: int = 0
	silence_streak: int = 0
	last_partial_text: str | None = None


def _is_pcm16_silence(audio_bytes: bytes, threshold: int = 300) -> bool:
	"""Return True when average PCM16 amplitude is below a small threshold."""
	if len(audio_bytes) < 2:
		return True

	# Keep even-length to avoid struct errors on malformed frames.
	even_length = len(audio_bytes) - (len(audio_bytes) % 2)
	if even_length <= 0:
		return True

	samples = struct.unpack(f"<{even_length // 2}h", audio_bytes[:even_length])
	avg_abs = sum(abs(sample) for sample in samples) / len(samples)
	return avg_abs < threshold


def _decode_audio_chunk_payload(payload: dict[str, Any]) -> tuple[bytes | None, str | None]:
	audio_b64 = payload.get("audio_b64")
	if not isinstance(audio_b64, str):
		return None, "payload.audio_b64 is required."

	try:
		return base64.b64decode(audio_b64), None
	except ValueError as exc:
		return None, f"Invalid base64 audio: {exc}"


def _build_event(event_type: str, session_id: str, seq: int, payload: dict[str, Any]) -> dict[str, Any]:
	return {
		"type": event_type,
		"session_id": session_id,
		"seq": seq,
		"payload": payload,
	}


async def _send_error(
	websocket: WebSocket,
	session_id: str,
	seq: int,
	code: str,
	message: str,
	retryable: bool,
) -> None:
	await websocket.send_json(
		_build_event(
			"error",
			session_id,
			seq,
			{
				"code": code,
				"message": message,
				"retryable": retryable,
			},
		)
	)


async def _emit_partial_if_ready(
	app: FastAPI,
	websocket: WebSocket,
	state: StreamState,
	session_id: str,
	seq: int,
) -> None:
	if state.chunk_count % PARTIAL_EVERY_CHUNKS != 0:
		return

	if len(state.buffer) < MIN_AUDIO_BYTES_FOR_TRANSCRIBE:
		return

	transcript, error = _transcribe_or_error(app, bytes(state.buffer))
	if error is not None:
		await _send_error(websocket, session_id, seq, "STT_FAILURE", error, True)
		return

	if transcript and transcript != state.last_partial_text:
		state.last_partial_text = transcript
		await websocket.send_json(
			_build_event(
				"partial_transcript",
				session_id,
				seq,
				{
					"text": transcript,
					"is_final": False,
				},
			)
		)


async def _finalize_turn(
	app: FastAPI,
	websocket: WebSocket,
	state: StreamState,
	session_id: str,
	seq: int,
) -> None:
	if len(state.buffer) < MIN_AUDIO_BYTES_FOR_TRANSCRIBE:
		state.buffer.clear()
		state.chunk_count = 0
		state.silence_streak = 0
		state.last_partial_text = None
		return

	transcript, error = _transcribe_or_error(app, bytes(state.buffer))
	if error is not None:
		code = "STT_UNAVAILABLE" if app.state.process_voice_query is None else "STT_FAILURE"
		retryable = app.state.process_voice_query is not None
		await _send_error(websocket, session_id, seq, code, error, retryable)
	else:
		await websocket.send_json(
			_build_event(
				"final_transcript",
				session_id,
				seq,
				{
					"text": transcript,
					"is_final": True,
				},
			)
		)

	state.buffer.clear()
	state.chunk_count = 0
	state.silence_streak = 0
	state.last_partial_text = None


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


def register_routes(app: FastAPI) -> None:
	"""Register API routes for the voice chat pipeline."""

	@app.post("/transcribe", status_code=200)
	async def transcribe_audio(request: TranscriptionRequest) -> TranscriptionResponse:
		"""
		Transcribe raw audio bytes using the local Whisper adapter.

		Request body:
		- audio_bytes_b64: Base64-encoded raw audio bytes

		Returns:
		- status_code: HTTP status
		- transcript: Transcribed text on success
		- error: Error message on failure
		"""
		try:
			audio_bytes = base64.b64decode(request.audio_bytes_b64)
		except ValueError as exc:
			return TranscriptionResponse(
				statuscode=400,
				error=f"Invalid audio input: {exc}",
			)

		transcript, error = _transcribe_or_error(app, audio_bytes)
		if error is not None:
			statuscode = 503 if app.state.process_voice_query is None else 400
			return TranscriptionResponse(
				statuscode=statuscode,
				error=error,
			)

		return TranscriptionResponse(statuscode=200, transcript=transcript)

	@app.websocket("/ws/stt")
	async def websocket_stt(websocket: WebSocket) -> None:
		"""WebSocket endpoint for streaming STT voice chat."""
		await websocket.accept()
		session_id = str(uuid4())
		seq = 0
		state = StreamState(buffer=bytearray())

		await websocket.send_json(
			_build_event("connected", session_id, seq, {"message": "WebSocket STT session established."})
		)

		try:
			while True:
				message = await websocket.receive_json()
				msg_type = message.get("type")
				seq = int(message.get("seq", seq + 1))

				if msg_type == "ping":
					await websocket.send_json(_build_event("pong", session_id, seq, {}))
					continue

				if msg_type == "turn_end":
					await _finalize_turn(app, websocket, state, session_id, seq)
					continue

				if msg_type == "close":
					await _finalize_turn(app, websocket, state, session_id, seq)
					await websocket.close(code=1000)
					break

				if msg_type != "audio_chunk":
					await _send_error(
						websocket,
						session_id,
						seq,
						"BAD_MESSAGE",
						f"Unsupported message type: {msg_type}",
						False,
					)
					continue

				payload = message.get("payload") or {}
				audio_bytes, decode_error = _decode_audio_chunk_payload(payload)
				if decode_error is not None or audio_bytes is None:
					await _send_error(websocket, session_id, seq, "BAD_AUDIO", decode_error or "Invalid audio.", True)
					continue

				state.buffer.extend(audio_bytes)
				state.chunk_count += 1

				codec = str(payload.get("codec", "pcm16")).lower()
				if codec == "pcm16" and _is_pcm16_silence(audio_bytes):
					state.silence_streak += 1
				else:
					state.silence_streak = 0

				await _emit_partial_if_ready(app, websocket, state, session_id, seq)

				if state.silence_streak >= SILENCE_END_STREAK:
					await _finalize_turn(app, websocket, state, session_id, seq)
		except WebSocketDisconnect:
			return

	@app.get("/health")
	async def health_check() -> dict[str, str]:
		"""Health check endpoint."""
		return {"status": "ok", "service": "voice-chat-ai"}

	@app.get("/stt-test", response_class=HTMLResponse)
	async def stt_test_ui() -> HTMLResponse:
		"""Serve the browser-based WebSocket STT test frontend."""
		html_path = pathlib.Path(__file__).parent / "static" / "index.html"
		return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

