from __future__ import annotations

from abc import ABC, abstractmethod


class STTError(Exception):
	"""Base exception for speech-to-text failures."""


class STTValidationError(STTError):
	"""Raised when audio input does not satisfy the STT contract."""


class STTPort(ABC):
	@abstractmethod
	def transcribe(self, audio_bytes: bytes) -> str:
