from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from voicechatai.domain.ports.stt_port import STTError, STTPort, STTValidationError

_DEFAULT_MODEL = "base"
_DEFAULT_TASK = "transcribe"
_DEFAULT_AUDIO_SUFFIX = ".wav"
_VALID_TASKS = {"transcribe", "translate"}
_RIFF_HEADER = b"RIFF"
_INT16_MAX = 32768.0


def _resolve_env_str(param: str | None, env_var: str, default: str) -> str:
	"""Return param if provided, else the env var value, else the default. Always stripped."""
	return (param or os.getenv(env_var, default)).strip()


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
		resolved_model = _resolve_env_str(model_name, "WHISPER_MODEL", _DEFAULT_MODEL)
		if not resolved_model:
			raise ValueError("model_name must contain non-whitespace text.")

		resolved_task = _resolve_env_str(task, "WHISPER_TASK", _DEFAULT_TASK)
		if resolved_task not in _VALID_TASKS:
			raise ValueError(f"task must be one of {_VALID_TASKS}.")

		resolved_suffix = _resolve_env_str(audio_suffix, "WHISPER_AUDIO_SUFFIX", _DEFAULT_AUDIO_SUFFIX)
		if not resolved_suffix:
			raise ValueError("audio_suffix must contain non-whitespace text.")
		if not resolved_suffix.startswith("."):
			resolved_suffix = f".{resolved_suffix}"

		self._language = _resolve_env_str(language, "WHISPER_LANGUAGE", "") or None
		self._task = resolved_task
		self._fp16 = self._resolve_fp16(fp16)
		self._audio_suffix = resolved_suffix
		self._model = self._load_model(resolved_model)

	def transcribe(self, audio_bytes: bytes) -> str:
		if not isinstance(audio_bytes, bytes):
			raise STTValidationError("Audio input must be raw bytes.")

		if not audio_bytes:
			raise STTValidationError("Audio bytes cannot be empty.")

		try:
			options: dict[str, Any] = {"task": self._task, "fp16": self._fp16}
			if self._language is not None:
				options["language"] = self._language

			if self._is_raw_pcm(audio_bytes):
				# Raw PCM int16 samples: decode to float32 waveform directly — no temp file.
				audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / _INT16_MAX
				result = self._model.transcribe(audio_array, **options)
			else:
				# Container format (RIFF/WAV, MP3, etc.): write to temp file and let ffmpeg decode.
				result = self._transcribe_from_tempfile(audio_bytes, options)
		except STTValidationError:
			raise
		except Exception as exc:
			raise STTError(f"Whisper transcription failed: {exc}") from exc

		transcript = result.get("text") if isinstance(result, dict) else None
		if not isinstance(transcript, str) or not transcript.strip():
			raise STTError("Whisper transcription produced an empty transcript.")

		return transcript.strip()

	def _is_raw_pcm(self, audio_bytes: bytes) -> bool:
		"""Return True when bytes look like headerless raw PCM16 (not a container format)."""
		return self._audio_suffix == _DEFAULT_AUDIO_SUFFIX and audio_bytes[:4] != _RIFF_HEADER

	def _transcribe_from_tempfile(self, audio_bytes: bytes, options: dict[str, Any]) -> Any:
		"""Write audio to a temp file, transcribe, then delete the file."""
		audio_path: Path | None = None
		try:
			with tempfile.NamedTemporaryFile(delete=False, suffix=self._audio_suffix) as f:
				f.write(audio_bytes)
				audio_path = Path(f.name)
			return self._model.transcribe(str(audio_path), **options)
		finally:
			if audio_path is not None:
				audio_path.unlink(missing_ok=True)

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
