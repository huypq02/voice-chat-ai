from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

from voicechatai.domain.ports.realtime_stt_port import (
    RealtimeSTTConfigError,
    RealtimeSTTConnectionError,
    RealtimeSTTError,
    RealtimeSTTPort,
    RealtimeTranscript,
)

_DEFAULT_MODEL = "nova-2"
_DEFAULT_LANGUAGE = "en-US"
_DEFAULT_SAMPLE_RATE = 16_000
_DEFAULT_CHANNELS = 1
_DEFAULT_ENCODING = "linear16"

_SENTINEL = None  # signals end-of-stream in the internal queue


class DeepgramRealtimeSTT(RealtimeSTTPort):
    """Deepgram streaming STT adapter using the async WebSocket API (deepgram-sdk v3).

    Ref: https://developers.deepgram.com/docs/python-sdk-streaming-transcription
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        language: str | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        encoding: str | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not resolved_key:
            raise RealtimeSTTConfigError(
                "Deepgram API key is required. Set DEEPGRAM_API_KEY or pass api_key."
            )

        self._api_key = resolved_key
        self._model = model or os.getenv("DEEPGRAM_MODEL", _DEFAULT_MODEL)
        self._language = language or os.getenv("DEEPGRAM_LANGUAGE", _DEFAULT_LANGUAGE)
        self._sample_rate = sample_rate or int(
            os.getenv("DEEPGRAM_SAMPLE_RATE", str(_DEFAULT_SAMPLE_RATE))
        )
        self._channels = channels or int(
            os.getenv("DEEPGRAM_CHANNELS", str(_DEFAULT_CHANNELS))
        )
        self._encoding = encoding or os.getenv("DEEPGRAM_ENCODING", _DEFAULT_ENCODING)

        self._connection: Any = None
        self._queue: asyncio.Queue[RealtimeTranscript | None] = asyncio.Queue()

    async def connect(self) -> None:
        try:
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
        except ImportError as exc:
            raise RealtimeSTTError(
                "deepgram-sdk is not installed. Run: pip install deepgram-sdk"
            ) from exc

        client = DeepgramClient(api_key=self._api_key)
        self._connection = client.listen.asyncwebsocket.v("1")

        async def _on_transcript(connection: Any, result: Any, **kwargs: Any) -> None:
            try:
                alternatives = result.channel.alternatives
                if not alternatives:
                    return
                text = alternatives[0].transcript
                if text:
                    await self._queue.put(
                        RealtimeTranscript(text=text, is_final=result.is_final)
                    )
            except Exception:
                pass  # malformed Deepgram event — skip silently

        async def _on_error(connection: Any, error: Any, **kwargs: Any) -> None:
            await self._queue.put(_SENTINEL)  # signal transcripts() to end

        self._connection.on(LiveTranscriptionEvents.Transcript, _on_transcript)
        self._connection.on(LiveTranscriptionEvents.Error, _on_error)

        options = LiveOptions(
            model=self._model,
            language=self._language,
            smart_format=True,
            encoding=self._encoding,
            channels=self._channels,
            sample_rate=self._sample_rate,
        )

        started = await self._connection.start(options)
        if not started:
            raise RealtimeSTTConnectionError(
                "Deepgram connection failed to start. Check your API key and options."
            )

    async def disconnect(self) -> None:
        if self._connection is not None:
            await self._connection.finish()
        await self._queue.put(_SENTINEL)  # stop the transcripts() iterator

    async def send_audio(self, chunk: bytes) -> None:
        if self._connection is None:
            raise RealtimeSTTConnectionError("Not connected. Call connect() first.")
        await self._connection.send(chunk)

    async def transcripts(self) -> AsyncGenerator[RealtimeTranscript, None]:  # type: ignore[override]
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                return
            yield item
