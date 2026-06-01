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
        """Convert text to WAV audio bytes via Gemini TTS (returns PCM wrapped in WAV)."""
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
            part = response.candidates[0].content.parts[0].inline_data
            pcm_bytes: bytes = part.data
            if not pcm_bytes:
                raise TTSError("Gemini TTS returned empty audio data.")
            # Gemini TTS returns L16 PCM at 24 kHz — wrap in WAV so browsers can play it
            sample_rate = _parse_sample_rate(part.mime_type) or 24000
            return _pcm_to_wav(pcm_bytes, sample_rate=sample_rate)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Gemini TTS synthesis failed: {exc}") from exc


def _parse_sample_rate(mime_type: str) -> int | None:
    """Extract rate=N from e.g. 'audio/L16;codec=pcm;rate=24000'."""
    for part in mime_type.split(";"):
        part = part.strip()
        if part.startswith("rate="):
            try:
                return int(part[5:])
            except ValueError:
                return None
    return None


def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    import struct
    data_size  = len(pcm)
    byte_rate  = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16,
        1, channels, sample_rate, byte_rate, block_align, bits,
        b"data", data_size,
    )
    return header + pcm
