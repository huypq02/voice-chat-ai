from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from voicechatai.domain.ports.tts_port import TTSError, TTSPort

# Default to ElevenLabs "Rachel" voice — override via ELEVENLABS_VOICE_ID.
_DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
_DEFAULT_MODEL_ID = "eleven_monolingual_v1"


@dataclass(slots=True)
class ElevenLabsTTS(TTSPort):
    """ElevenLabs TTS adapter implementing the TTS port."""

    voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", _DEFAULT_VOICE_ID)
    model_id: str = os.getenv("ELEVENLABS_MODEL_ID", _DEFAULT_MODEL_ID)
    api_key: str | None = os.getenv("ELEVENLABS_API_KEY")
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.client is not None:
            self._client = self.client
            return

        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:
            raise RuntimeError("elevenlabs package is required for ElevenLabsTTS.") from exc

        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabsTTS.")

        self._client = ElevenLabs(api_key=self.api_key)

    def synthesize(self, text: str) -> bytes:
        """Convert text to MP3 audio bytes via ElevenLabs."""
        content = text.strip()
        if not content:
            raise TTSError("Cannot synthesize empty text.")

        try:
            audio_stream = self._client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=content,
                model_id=self.model_id,
            )
            return b"".join(audio_stream)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"ElevenLabs synthesis failed: {exc}") from exc
