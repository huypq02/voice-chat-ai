from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from voicechatai.domain.ports.tts_port import TTSError


def _make_mock_genai(audio_data: bytes = b"fake-audio") -> tuple[MagicMock, MagicMock]:
    """Return (mock_genai_module, mock_types_module) with a working TTS response."""
    mock_part = MagicMock()
    mock_part.inline_data.data = audio_data

    mock_content = MagicMock()
    mock_content.parts = [mock_part]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content

    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]

    mock_models = MagicMock()
    mock_models.generate_content.return_value = mock_response

    mock_client_instance = MagicMock()
    mock_client_instance.models = mock_models

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client_instance

    mock_types = MagicMock()

    return mock_genai, mock_types


def _build_tts(mock_genai: MagicMock, mock_types: MagicMock, api_key: str = "key"):
    from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS

    with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_types}):
        return GeminiTTS(api_key=api_key)


class GeminiTTSConstructorTests(unittest.TestCase):
    def setUp(self):
        # Ensure fresh import each test
        sys.modules.pop("voicechatai.infrastructure.tts.gemini_tts", None)

    def test_raises_runtime_error_when_package_missing(self) -> None:
        with patch.dict("sys.modules", {"google.genai": None, "google": None}):
            with self.assertRaises(RuntimeError):
                from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS
                GeminiTTS(api_key="key")

    def test_raises_when_api_key_missing(self) -> None:
        mock_genai, mock_types = _make_mock_genai()
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_types}):
            with patch.dict("os.environ", {}, clear=True):
                from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS
                with self.assertRaises(RuntimeError):
                    GeminiTTS(api_key="")

    def test_accepts_api_key_from_constructor(self) -> None:
        mock_genai, mock_types = _make_mock_genai()
        from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_types}):
            tts = GeminiTTS(api_key="explicit-key")
        self.assertEqual(tts.api_key, "explicit-key")

    def test_default_voice_and_model(self) -> None:
        mock_genai, mock_types = _make_mock_genai()
        from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_types}):
            tts = GeminiTTS(api_key="key")
        self.assertEqual(tts.voice, "Kore")
        self.assertEqual(tts.model, "gemini-2.5-flash-preview-tts")

    def test_custom_voice_from_env(self) -> None:
        mock_genai, mock_types = _make_mock_genai()
        from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_types}):
            with patch.dict("os.environ", {"GEMINI_TTS_VOICE": "Puck"}):
                tts = GeminiTTS(api_key="key")
        self.assertEqual(tts.voice, "Puck")


class GeminiTTSSynthesizeTests(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("voicechatai.infrastructure.tts.gemini_tts", None)

    def test_raises_tts_error_on_empty_text(self) -> None:
        mock_genai, mock_types = _make_mock_genai()
        from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_types}):
            tts = GeminiTTS(api_key="key")
        with self.assertRaises(TTSError):
            tts.synthesize("   ")

    def test_returns_audio_bytes_on_success(self) -> None:
        mock_genai, mock_types = _make_mock_genai(b"audio-bytes")
        from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_types}):
            tts = GeminiTTS(api_key="key")
        result = tts.synthesize("Hello world")
        self.assertEqual(result, b"audio-bytes")

    def test_passes_stripped_text_to_api(self) -> None:
        mock_genai, mock_types = _make_mock_genai()
        from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_types}):
            tts = GeminiTTS(api_key="key")
        tts.synthesize("  hello  ")
        mock_genai.Client.return_value.models.generate_content.assert_called_once()
        call_kwargs = mock_genai.Client.return_value.models.generate_content.call_args
        self.assertEqual(call_kwargs.kwargs["contents"], "hello")

    def test_raises_tts_error_on_empty_audio_data(self) -> None:
        mock_genai, mock_types = _make_mock_genai(b"")
        from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_types}):
            tts = GeminiTTS(api_key="key")
        with self.assertRaises(TTSError):
            tts.synthesize("hello")

    def test_wraps_api_exception_in_tts_error(self) -> None:
        mock_genai, mock_types = _make_mock_genai()
        mock_genai.Client.return_value.models.generate_content.side_effect = RuntimeError("api down")
        from voicechatai.infrastructure.tts.gemini_tts import GeminiTTS
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_types}):
            tts = GeminiTTS(api_key="key")
        with self.assertRaises(TTSError):
            tts.synthesize("hello")


if __name__ == "__main__":
    unittest.main()
