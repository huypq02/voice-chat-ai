from __future__ import annotations

import asyncio
import queue as sync_queue
import threading
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from voicechatai.domain.ports.realtime_stt_port import (
    RealtimeSTTConfigError,
    RealtimeSTTConnectionError,
    RealtimeTranscript,
)
from voicechatai.infrastructure.stt.assemblyai_stt import AssemblyAIRealtimeSTT


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Fake AssemblyAI SDK objects
# ---------------------------------------------------------------------------

class _FakePartialTranscript:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeFinalTranscript:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeRealtimeTranscriber:
    """Drives on_data callbacks synchronously from a list of canned transcripts."""

    def __init__(
        self,
        canned: list[Any],
        sample_rate: int,
        on_data,
        on_error,
    ) -> None:
        self._canned = canned
        self._on_data = on_data
        self._on_error = on_error
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def stream(self, generator) -> None:
        # Consume the audio generator so it terminates naturally.
        for _ in generator:
            pass
        # Fire canned transcript events after all audio is consumed.
        for event in self._canned:
            self._on_data(event)

    def close(self) -> None:
        self.closed = True


def _make_fake_aai_module(canned: list[Any]) -> MagicMock:
    mock_aai = MagicMock()
    mock_aai.settings = MagicMock()
    mock_aai.RealtimeFinalTranscript = _FakeFinalTranscript

    def _transcriber_factory(sample_rate, on_data, on_error):
        return _FakeRealtimeTranscriber(canned, sample_rate, on_data, on_error)

    mock_aai.RealtimeTranscriber = _transcriber_factory
    return mock_aai


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class AssemblyAIRealtimeSTTConstructorTests(unittest.TestCase):
    def test_raises_config_error_when_api_key_missing(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RealtimeSTTConfigError):
                AssemblyAIRealtimeSTT(api_key="")

    def test_accepts_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"ASSEMBLYAI_API_KEY": "env-key"}):
            adapter = AssemblyAIRealtimeSTT()
            self.assertEqual(adapter._api_key, "env-key")

    def test_accepts_api_key_from_constructor(self) -> None:
        adapter = AssemblyAIRealtimeSTT(api_key="explicit-key")
        self.assertEqual(adapter._api_key, "explicit-key")

    def test_default_sample_rate(self) -> None:
        adapter = AssemblyAIRealtimeSTT(api_key="k")
        self.assertEqual(adapter._sample_rate, 16_000)


# ---------------------------------------------------------------------------
# send_audio before connect
# ---------------------------------------------------------------------------

class AssemblyAIRealtimeSTTPreConnectTests(unittest.TestCase):
    def test_send_audio_raises_before_connect(self) -> None:
        adapter = AssemblyAIRealtimeSTT(api_key="key")
        with self.assertRaises(RealtimeSTTConnectionError):
            _run(adapter.send_audio(b"chunk"))


# ---------------------------------------------------------------------------
# Full lifecycle: connect → send_audio → transcripts → disconnect
# ---------------------------------------------------------------------------

class AssemblyAIRealtimeSTTLifecycleTests(unittest.TestCase):
    def _run_full_flow(self, canned: list[Any]) -> list[RealtimeTranscript]:
        """
        Connect, send one audio chunk, disconnect, then collect all transcripts.
        The fake SDK fires canned events after the audio generator is exhausted.
        """
        mock_aai = _make_fake_aai_module(canned)

        async def _test() -> list[RealtimeTranscript]:
            with patch.dict("sys.modules", {"assemblyai": mock_aai}):
                adapter = AssemblyAIRealtimeSTT(api_key="key")
                await adapter.connect()
                await adapter.send_audio(b"pcm-data")
                await adapter.disconnect()

                results: list[RealtimeTranscript] = []
                async for t in adapter.transcripts():
                    results.append(t)
                return results

        return _run(_test())

    def test_partial_transcript_yields_is_final_false(self) -> None:
        results = self._run_full_flow([_FakePartialTranscript("partial text")])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "partial text")
        self.assertFalse(results[0].is_final)

    def test_final_transcript_yields_is_final_true(self) -> None:
        results = self._run_full_flow([_FakeFinalTranscript("final text")])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "final text")
        self.assertTrue(results[0].is_final)

    def test_multiple_transcripts_all_yielded(self) -> None:
        canned = [
            _FakePartialTranscript("hello"),
            _FakeFinalTranscript("hello world"),
        ]
        results = self._run_full_flow(canned)
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].is_final)
        self.assertTrue(results[1].is_final)

    def test_empty_text_transcripts_are_skipped(self) -> None:
        canned = [
            _FakePartialTranscript(""),  # empty → skipped
            _FakeFinalTranscript("hello"),
        ]
        results = self._run_full_flow(canned)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "hello")

    def test_transcripts_stops_after_disconnect(self) -> None:
        results = self._run_full_flow([])  # no canned events
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
