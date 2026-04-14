from __future__ import annotations

from abc import ABC, abstractmethod


class TTSError(Exception):
    """Raised when text-to-speech synthesis fails."""


class TTSPort(ABC):
    """Port for text-to-speech synthesis."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (MP3)."""
        raise NotImplementedError
