from __future__ import annotations

import asyncio
import base64
import json
import struct

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from voicechatai.application.use_cases.process_voice_chat import ProcessVoiceChat
from voicechatai.domain.entities.conversation_state import ConversationState


def _encode_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
    """Wrap raw PCM16 bytes in a WAV container so Whisper can decode it."""
    data_size = len(pcm_bytes)
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16,
        1, channels, sample_rate, byte_rate, block_align, bits,
        b"data", data_size,
    )
    return header + pcm_bytes


def register_ws_routes(app: FastAPI) -> None:
    """Register WebSocket routes."""

    @app.websocket("/ws/voice-chat")
    async def ws_voice_chat(websocket: WebSocket) -> None:
        """
        Full pipeline over WebSocket.

        Protocol (client → server):
          binary frame          Raw PCM16 audio chunk (16 kHz, mono)
          {"type":"turn_end"}   Signals end of speech — triggers pipeline

        Protocol (server → client):
          {"type":"connected"}                          On connect
          {"type":"processing"}                         Pipeline started
          {"type":"result", "transcript":...,           Full pipeline result
           "answer":..., "audio_b64":...,
           "decision":..., "intent":...,
           "top1_score":..., "clarification_questions":[...]}
          {"type":"session_end", "answer":...}          On STOP intent
          {"type":"error", "message":...}               On failure
        """
        await websocket.accept()

        process_voice_chat: ProcessVoiceChat | None = websocket.app.state.process_voice_chat
        if process_voice_chat is None:
            error = websocket.app.state.voice_chat_boot_error or "Pipeline not initialized."
            await websocket.send_json({"type": "error", "message": error})
            await websocket.close(code=1011)
            return

        await websocket.send_json({"type": "connected"})

        session_state = ConversationState()
        audio_chunks: list[bytes] = []

        try:
            while True:
                message = await websocket.receive()

                if message["type"] == "websocket.disconnect":
                    break

                if message.get("bytes"):
                    audio_chunks.append(message["bytes"])
                    continue

                if message.get("text"):
                    try:
                        payload = json.loads(message["text"])
                    except Exception:
                        await websocket.send_json({"type": "error", "message": "Invalid JSON."})
                        continue

                    if payload.get("type") == "turn_end":
                        await websocket.send_json({"type": "processing"})

                        # Empty audio chunks = silence turn
                        if audio_chunks:
                            wav_bytes = _encode_wav(b"".join(audio_chunks))
                        else:
                            wav_bytes = b""
                        audio_chunks = []

                        loop = asyncio.get_event_loop()
                        try:
                            result = await loop.run_in_executor(
                                None, process_voice_chat.execute, wav_bytes, session_state
                            )
                        except Exception as exc:
                            await websocket.send_json({"type": "error", "message": str(exc)})
                            continue

                        # STOP: send closing statement and end the session
                        if result.intent == "STOP":
                            await websocket.send_json({
                                "type": "session_end",
                                "answer": result.answer,
                                "audio_b64": base64.b64encode(result.audio_bytes).decode(),
                            })
                            await websocket.close(code=1000)
                            return

                        # REPLAY or CONTINUE: send result normally
                        # Client skips rendering answer text on REPLAY (just plays audio prompt)
                        await websocket.send_json({
                            "type": "result",
                            "transcript": result.transcript,
                            "answer": result.answer,
                            "audio_b64": base64.b64encode(result.audio_bytes).decode(),
                            "decision": result.decision,
                            "intent": result.intent,
                            "top1_score": result.top1_score,
                            "clarification_questions": result.clarification_questions,
                        })

        except WebSocketDisconnect:
            pass
