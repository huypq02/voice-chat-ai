from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from voicechatai.domain.ports.tts_port import TTSError, TTSPort

# Available voices: Kore, Charon, Fenrir, Aoede, Puck — see Gemini TTS docs.
_DEFAULT_VOICE = "Kore"
_DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"


@dataclass(slots=True)
class GeminiTTS(TTSPort):
    """Google Gemini TTS adapter implementing the TTS port via google-genai SDK."""

    model: str = field(default_factory=lambda: os.getenv("GEMINI_TTS_MODEL", _DEFAULT_MODEL))
    voice: str = field(default_factory=lambda: os.getenv("GEMINI_TTS_VOICE", _DEFAULT_VOICE))
    api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    _client: Any = field(init=False, repr=False)
    _types: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from google import genai  # type: ignore[import]
            from google.genai import types  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("google-genai package is required for GeminiTTS.") from exc

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for GeminiTTS.")

        self._client = genai.Client(api_key=self.api_key)
        self._types = types

    def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (WAV/PCM) via Gemini TTS."""
        content = text.strip()
        if not content:
            raise TTSError("Cannot synthesize empty text.")

        try:
            types = self._types
            response = self._client.models.generate_content(
                model=self.model,
                contents=content,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self.voice,
                            )
                        )
                    ),
                ),
            )
            audio_data: bytes = response.candidates[0].content.parts[0].inline_data.data
            if not audio_data:
                raise TTSError("Gemini TTS returned empty audio data.")
            return audio_data
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Gemini TTS synthesis failed: {exc}") from exc
