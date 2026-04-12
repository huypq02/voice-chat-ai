from __future__ import annotations

from abc import ABC, abstractmethod


class STTError(Exception):
	"""Base exception for speech-to-text failures."""


class STTValidationError(STTError):
	"""Raised when audio input does not satisfy the STT contract."""


class STTPort(ABC):
	"""Defines the application-facing speech-to-text contract."""

	@abstractmethod
	def transcribe(self, audio_bytes: bytes) -> str:
		"""Convert raw audio bytes into non-empty transcript text."""


