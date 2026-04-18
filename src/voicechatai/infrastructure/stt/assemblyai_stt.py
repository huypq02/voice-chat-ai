from __future__ import annotations

import asyncio
import os
import queue as sync_queue
import threading
from collections.abc import AsyncGenerator
from typing import Any

from voicechatai.domain.ports.realtime_stt_port import (
    RealtimeSTTConfigError,
    RealtimeSTTConnectionError,
    RealtimeSTTError,
    RealtimeSTTPort,
    RealtimeTranscript,
)

_DEFAULT_SAMPLE_RATE = 16_000
_SENTINEL = None  # signals end-of-stream in both queues


class AssemblyAIRealtimeSTT(RealtimeSTTPort):
    """AssemblyAI streaming STT adapter, bridging the sync SDK to asyncio.

    The SDK's RealtimeTranscriber is callback-based and runs in a background
    thread. Audio chunks flow in via a sync queue; transcripts flow out via
    an asyncio queue so the WebSocket handler can await them.

    Ref: https://www.assemblyai.com/docs/speech-to-text/streaming
    """

    def __init__(
        self,
        api_key: str | None = None,
        sample_rate: int | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("ASSEMBLYAI_API_KEY", "")
        if not resolved_key:
            raise RealtimeSTTConfigError(
                "AssemblyAI API key is required. Set ASSEMBLYAI_API_KEY or pass api_key."
            )

        self._api_key = resolved_key
        self._sample_rate = sample_rate or int(
            os.getenv("ASSEMBLYAI_SAMPLE_RATE", str(_DEFAULT_SAMPLE_RATE))
        )

        # Sync queue — thread reads audio chunks from here without asyncio overhead.
        self._audio_queue: sync_queue.Queue[bytes | None] = sync_queue.Queue()
        # Async queue — main event loop awaits transcript events here.
        self._out_queue: asyncio.Queue[RealtimeTranscript | None] = asyncio.Queue()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    async def connect(self) -> None:
        try:
            import assemblyai as aai  # noqa: F401 — validate SDK is installed
        except ImportError as exc:
            raise RealtimeSTTError(
                "assemblyai is not installed. Run: pip install assemblyai"
            ) from exc

        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run_transcriber, daemon=True)
        self._thread.start()

    async def disconnect(self) -> None:
        # Send sentinel to stop the audio generator in the background thread.
        self._audio_queue.put_nowait(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=10)

    async def send_audio(self, chunk: bytes) -> None:
        if self._loop is None:
            raise RealtimeSTTConnectionError("Not connected. Call connect() first.")
        self._audio_queue.put_nowait(chunk)

    async def transcripts(self) -> AsyncGenerator[RealtimeTranscript, None]:  # type: ignore[override]
        while True:
            item = await self._out_queue.get()
            if item is _SENTINEL:
                return
            yield item

    # ------------------------------------------------------------------
    # Background thread logic
    # ------------------------------------------------------------------

    def _run_transcriber(self) -> None:
        """Entry point for the background thread that owns the SDK transcriber."""
        import assemblyai as aai

        aai.settings.api_key = self._api_key

        def _on_data(transcript: Any) -> None:
            if not transcript.text:
                return
            is_final = isinstance(transcript, aai.RealtimeFinalTranscript)
            event = RealtimeTranscript(text=transcript.text, is_final=is_final)
            # Bridge sync callback → async queue safely across threads.
            assert self._loop is not None
            self._loop.call_soon_threadsafe(self._out_queue.put_nowait, event)

        def _on_error(error: Any) -> None:
            assert self._loop is not None
            self._loop.call_soon_threadsafe(self._out_queue.put_nowait, _SENTINEL)

        transcriber = aai.RealtimeTranscriber(
            sample_rate=self._sample_rate,
            on_data=_on_data,
            on_error=_on_error,
        )

        try:
            transcriber.connect()
            transcriber.stream(self._audio_generator())
        except Exception:
            pass  # error already signalled via _on_error or connection failure
        finally:
            try:
                transcriber.close()
            except Exception:
                pass
            # Guarantee the async iterator always terminates.
            assert self._loop is not None
            self._loop.call_soon_threadsafe(self._out_queue.put_nowait, _SENTINEL)

    def _audio_generator(self):
        """Blocking generator consumed by the SDK in the background thread."""
        while True:
            chunk = self._audio_queue.get()  # blocks until audio or sentinel arrives
            if chunk is _SENTINEL:
                return
            yield chunk
