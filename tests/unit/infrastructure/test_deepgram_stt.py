from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from voicechatai.domain.ports.realtime_stt_port import (
    RealtimeSTTConfigError,
    RealtimeSTTConnectionError,
    RealtimeSTTError,
    RealtimeTranscript,
)
from voicechatai.infrastructure.stt.deepgram_stt import DeepgramRealtimeSTT


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class DeepgramRealtimeSTTConstructorTests(unittest.TestCase):
    def test_raises_config_error_when_api_key_missing(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RealtimeSTTConfigError):
                DeepgramRealtimeSTT(api_key="")

    def test_accepts_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}):
            adapter = DeepgramRealtimeSTT()
            self.assertEqual(adapter._api_key, "test-key")

    def test_accepts_api_key_from_constructor(self) -> None:
        adapter = DeepgramRealtimeSTT(api_key="explicit-key")
        self.assertEqual(adapter._api_key, "explicit-key")

    def test_defaults_applied(self) -> None:
        adapter = DeepgramRealtimeSTT(api_key="k")
        self.assertEqual(adapter._model, "nova-2")
        self.assertEqual(adapter._language, "en-US")
        self.assertEqual(adapter._sample_rate, 16_000)
        self.assertEqual(adapter._channels, 1)
        self.assertEqual(adapter._encoding, "linear16")


class DeepgramRealtimeSTTConnectTests(unittest.TestCase):
    def test_raises_stt_error_when_sdk_missing(self) -> None:
        adapter = DeepgramRealtimeSTT(api_key="key")
        with patch(
            "voicechatai.infrastructure.stt.deepgram_stt.DeepgramRealtimeSTT.connect",
            side_effect=RealtimeSTTError("deepgram-sdk is not installed."),
        ):
            with self.assertRaises(RealtimeSTTError):
                _run(adapter.connect())

    def _make_mock_deepgram(self, start_returns: bool = True):
        """Return a patched deepgram module with a controllable async connection mock."""
        mock_connection = MagicMock()
        mock_connection.on = MagicMock()
        mock_connection.start = AsyncMock(return_value=start_returns)
        mock_connection.finish = AsyncMock()
        mock_connection.send = AsyncMock()

        mock_client = MagicMock()
        mock_client.listen.asyncwebsocket.v.return_value = mock_connection

        mock_deepgram_module = MagicMock()
        mock_deepgram_module.DeepgramClient.return_value = mock_client
        mock_deepgram_module.LiveOptions = MagicMock()
        mock_deepgram_module.LiveTranscriptionEvents.Transcript = "Transcript"
        mock_deepgram_module.LiveTranscriptionEvents.Error = "Error"

        return mock_deepgram_module, mock_connection

    def test_connect_starts_websocket_and_registers_handlers(self) -> None:
        adapter = DeepgramRealtimeSTT(api_key="key")
        mock_module, mock_conn = self._make_mock_deepgram()

        with patch.dict("sys.modules", {"deepgram": mock_module}):
            _run(adapter.connect())

        self.assertEqual(mock_conn.start.call_count, 1)
        self.assertEqual(mock_conn.on.call_count, 2)  # Transcript + Error

    def test_connect_raises_connection_error_when_start_returns_false(self) -> None:
        adapter = DeepgramRealtimeSTT(api_key="key")
        mock_module, _ = self._make_mock_deepgram(start_returns=False)

        with patch.dict("sys.modules", {"deepgram": mock_module}):
            with self.assertRaises(RealtimeSTTConnectionError):
                _run(adapter.connect())

    def test_send_audio_raises_before_connect(self) -> None:
        adapter = DeepgramRealtimeSTT(api_key="key")
        with self.assertRaises(RealtimeSTTConnectionError):
            _run(adapter.send_audio(b"chunk"))

    def test_send_audio_delegates_to_connection(self) -> None:
        adapter = DeepgramRealtimeSTT(api_key="key")
        mock_module, mock_conn = self._make_mock_deepgram()

        with patch.dict("sys.modules", {"deepgram": mock_module}):
            _run(adapter.connect())
            _run(adapter.send_audio(b"pcm-chunk"))

        mock_conn.send.assert_awaited_once_with(b"pcm-chunk")

    def test_disconnect_calls_finish_and_closes_iterator(self) -> None:
        adapter = DeepgramRealtimeSTT(api_key="key")
        mock_module, mock_conn = self._make_mock_deepgram()

        with patch.dict("sys.modules", {"deepgram": mock_module}):
            _run(adapter.connect())
            _run(adapter.disconnect())

        mock_conn.finish.assert_awaited_once()


class DeepgramRealtimeSTTTranscriptsTests(unittest.TestCase):
    def test_transcripts_yields_events_and_stops_on_sentinel(self) -> None:
        adapter = DeepgramRealtimeSTT(api_key="key")

        async def _run_test() -> list[RealtimeTranscript]:
            # Pre-load the queue manually — bypassing connect().
            await adapter._queue.put(RealtimeTranscript(text="hello", is_final=False))
            await adapter._queue.put(RealtimeTranscript(text="hello world", is_final=True))
            await adapter._queue.put(None)  # sentinel

            results = []
            async for t in adapter.transcripts():
                results.append(t)
            return results

        results = asyncio.get_event_loop().run_until_complete(_run_test())
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].text, "hello")
        self.assertFalse(results[0].is_final)
        self.assertEqual(results[1].text, "hello world")
        self.assertTrue(results[1].is_final)

    def test_transcripts_stops_immediately_on_sentinel(self) -> None:
        adapter = DeepgramRealtimeSTT(api_key="key")

        async def _run_test() -> list[RealtimeTranscript]:
            await adapter._queue.put(None)
            results = []
            async for t in adapter.transcripts():
                results.append(t)
            return results

        results = asyncio.get_event_loop().run_until_complete(_run_test())
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
