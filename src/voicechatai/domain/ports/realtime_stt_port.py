from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class RealtimeTranscript:
    """A single transcript event from the streaming STT engine."""

    text: str
    is_final: bool  # True = committed utterance, False = in-progress partial


class RealtimeSTTError(Exception):
    """Base exception for real-time STT failures."""


class RealtimeSTTConnectionError(RealtimeSTTError):
    """Raised when the streaming connection cannot be established or is lost."""


class RealtimeSTTConfigError(RealtimeSTTError):
    """Raised when required configuration (API key, params) is missing or invalid."""


class RealtimeSTTPort(ABC):
    """Async streaming interface for real-time speech-to-text.

    Lifecycle: connect() → send_audio() in a loop → disconnect().
    Iterate transcripts() concurrently to receive events.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Open the streaming connection to the STT provider."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the streaming connection and drain pending transcripts."""

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """Forward a raw audio chunk to the provider.

        Raises RealtimeSTTConnectionError if called before connect().
        """

    @abstractmethod
    def transcripts(self) -> AsyncIterator[RealtimeTranscript]:
        """Async iterator that yields transcript events in arrival order.

        Ends naturally after disconnect() drains the queue.
        """
