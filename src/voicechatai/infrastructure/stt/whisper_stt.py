from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from voicechatai.domain.ports.stt_port import STTError, STTPort, STTValidationError


class WhisperSTT(STTPort):
	"""Local Whisper adapter that transcribes raw audio bytes with openai-whisper."""

	def __init__(
		self,
		model_name: str | None = None,
		language: str | None = None,
		task: str | None = None,
		fp16: bool | None = None,
		audio_suffix: str | None = None,
	):
		resolved_model_name = (model_name or os.getenv("WHISPER_MODEL", "base")).strip()
		if not resolved_model_name:
			raise ValueError("model_name must contain non-whitespace text.")

		resolved_task = (task or os.getenv("WHISPER_TASK", "transcribe")).strip()
		if resolved_task not in {"transcribe", "translate"}:
			raise ValueError("task must be either 'transcribe' or 'translate'.")

		resolved_audio_suffix = (audio_suffix or os.getenv("WHISPER_AUDIO_SUFFIX", ".wav")).strip()
		if not resolved_audio_suffix:
			raise ValueError("audio_suffix must contain non-whitespace text.")
		if not resolved_audio_suffix.startswith("."):
			resolved_audio_suffix = f".{resolved_audio_suffix}"

		self._language = (language or os.getenv("WHISPER_LANGUAGE") or "").strip() or None
		self._task = resolved_task
		self._fp16 = self._resolve_fp16(fp16)
		self._audio_suffix = resolved_audio_suffix
		self._model = self._load_model(resolved_model_name)

	def transcribe(self, audio_bytes: bytes) -> str:
		if not isinstance(audio_bytes, bytes):
			raise STTValidationError("Audio input must be raw bytes.")

		if not audio_bytes:
			raise STTValidationError("Audio bytes cannot be empty.")

		try:
			options: dict[str, Any] = {
				"task": self._task,
				"fp16": self._fp16,
			}
			if self._language is not None:
				options["language"] = self._language

			if self._audio_suffix == ".wav" and audio_bytes[:4] != b"RIFF":
				# Raw PCM: decode directly to float32 waveform — no temp file needed.
				audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
				result = self._model.transcribe(audio_array, **options)
			else:
				# Encoded audio (RIFF/WAV or other container): write to temp file.
				audio_path: Path | None = None
				try:
					with tempfile.NamedTemporaryFile(delete=False, suffix=self._audio_suffix) as f:
						f.write(audio_bytes)
						audio_path = Path(f.name)
					result = self._model.transcribe(str(audio_path), **options)
				finally:
					if audio_path is not None:
						audio_path.unlink(missing_ok=True)
		except STTValidationError:
			raise
		except Exception as exc:
			raise STTError(f"Whisper transcription failed: {exc}") from exc

		transcript = result.get("text") if isinstance(result, dict) else None
		if not isinstance(transcript, str) or not transcript.strip():
			raise STTError("Whisper transcription produced an empty transcript.")

		return transcript.strip()

	@staticmethod
	def _resolve_fp16(fp16: bool | None) -> bool:
		if fp16 is not None:
			return fp16

		value = os.getenv("WHISPER_FP16", "false").strip().lower()
		return value in {"1", "true", "yes", "on"}

	@staticmethod
	def _load_model(model_name: str) -> Any:
		try:
			whisper = importlib.import_module("whisper")
		except ImportError as exc:
			raise STTError(
				"Whisper is not installed. Install 'openai-whisper' and ensure ffmpeg is available on PATH."
			) from exc

		try:
			return whisper.load_model(model_name)
		except Exception as exc:
			raise STTError(f"Whisper model '{model_name}' could not be loaded.") from exc


