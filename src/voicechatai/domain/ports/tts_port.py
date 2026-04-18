from __future__ import annotations

from abc import ABC, abstractmethod


class TTSError(Exception):
    """Raised when text-to-speech synthesis fails."""


class TTSPort(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError
