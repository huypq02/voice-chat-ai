from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from voicechatai.domain.ports.realtime_stt_port import (
    RealtimeSTTError,
    RealtimeSTTPort,
)
from voicechatai.infrastructure.stt.assemblyai_stt import AssemblyAIRealtimeSTT
from voicechatai.infrastructure.stt.deepgram_stt import DeepgramRealtimeSTT

_SUPPORTED_PROVIDERS = {"deepgram", "assemblyai"}


def _create_adapter(provider: str, state: Any) -> RealtimeSTTPort | None:
    """Instantiate a realtime STT adapter for the requested provider."""
    if provider == "deepgram":
        return DeepgramRealtimeSTT(api_key=getattr(state, "deepgram_api_key", None))
    if provider == "assemblyai":
        return AssemblyAIRealtimeSTT(api_key=getattr(state, "assemblyai_api_key", None))
    return None


def register_ws_routes(app: FastAPI) -> None:
    """Register WebSocket routes for real-time STT streaming."""

    @app.websocket("/ws/transcribe")
    async def ws_transcribe(websocket: WebSocket, provider: str = "deepgram") -> None:
        """
        Stream audio chunks to a real-time STT provider and receive transcript events.

        Query params:
          provider — "deepgram" (default) or "assemblyai"

        Protocol:
          Client  → server : binary frames (raw audio chunks, e.g. 16-bit PCM @ 16 kHz)
          Server  → client : JSON  {"text": "...", "is_final": true|false}
          On error         : JSON  {"error": "..."} then socket closes
        """
        await websocket.accept()

        if provider not in _SUPPORTED_PROVIDERS:
            await websocket.send_json(
                {"error": f"Unknown provider '{provider}'. Choose from: {sorted(_SUPPORTED_PROVIDERS)}"}
            )
            await websocket.close(code=1008)
            return

        try:
            adapter = _create_adapter(provider, websocket.app.state)
        except RealtimeSTTError as exc:
            await websocket.send_json({"error": str(exc)})
            await websocket.close(code=1011)
            return

        if adapter is None:
            await websocket.send_json({"error": "Adapter could not be created."})
            await websocket.close(code=1011)
            return

        try:
            await adapter.connect()
        except RealtimeSTTError as exc:
            await websocket.send_json({"error": str(exc)})
            await websocket.close(code=1011)
            return

        async def _forward_transcripts() -> None:
            """Read transcript events from the adapter and push to the client."""
            async for transcript in adapter.transcripts():
                try:
                    await websocket.send_json(asdict(transcript))
                except Exception:
                    return  # client disconnected mid-stream

        forward_task = asyncio.create_task(_forward_transcripts())

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                chunk = message.get("bytes")
                if chunk:
                    await adapter.send_audio(chunk)
        except WebSocketDisconnect:
            pass
        finally:
            await adapter.disconnect()
            forward_task.cancel()
            try:
                await forward_task
            except asyncio.CancelledError:
                pass
