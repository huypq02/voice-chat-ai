from __future__ import annotations

import importlib
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

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
		sample_rate: int | None = None,
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
		self._sample_rate = sample_rate or int(os.getenv("WHISPER_SAMPLE_RATE", "16000"))
		self._ffmpeg_exe = self._ensure_ffmpeg_available()
		self._patch_whisper_audio_runner(self._ffmpeg_exe)
		self._model = self._load_model(resolved_model_name)

	def transcribe(self, audio_bytes: bytes) -> str:
		if not isinstance(audio_bytes, bytes):
			raise STTValidationError("Audio input must be raw bytes.")

		if not audio_bytes:
			raise STTValidationError("Audio bytes cannot be empty.")

		audio_path: Path | None = None
		try:
			write_bytes = (
				self._prepend_wav_header(audio_bytes, self._sample_rate)
				if self._audio_suffix == ".wav" and audio_bytes[:4] != b"RIFF"
				else audio_bytes
			)
			with NamedTemporaryFile(delete=False, suffix=self._audio_suffix) as audio_file:
				audio_file.write(write_bytes)
				audio_path = Path(audio_file.name)

			options: dict[str, Any] = {
				"task": self._task,
				"fp16": self._fp16,
			}
			if self._language is not None:
				options["language"] = self._language

			result = self._model.transcribe(str(audio_path), **options)
		except STTValidationError:
			raise
		except Exception as exc:
			raise STTError(f"Whisper transcription failed: {exc}") from exc
		finally:
			if audio_path is not None:
				audio_path.unlink(missing_ok=True)

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

	@staticmethod
	def _ensure_ffmpeg_available() -> str:
		ffmpeg_on_path = shutil.which("ffmpeg")
		if ffmpeg_on_path is not None:
			return ffmpeg_on_path

		# Try to locate ffmpeg from the imageio-ffmpeg bundle and expose it as `ffmpeg`.
		try:
			import imageio_ffmpeg  # type: ignore[import-untyped]

			ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
			ffmpeg_dir = Path(ffmpeg_exe).parent
			shim_dir = Path(tempfile.gettempdir()) / "voice-chat-ai-bin"
			shim_dir.mkdir(parents=True, exist_ok=True)
			shim_file = shim_dir / "ffmpeg.cmd"

			# Whisper executes the literal command "ffmpeg"; on Windows we provide a shim.
			shim_file.write_text(f'@echo off\r\n"{ffmpeg_exe}" %*\r\n', encoding="ascii")

			path = os.environ.get("PATH", "")
			os.environ["PATH"] = os.pathsep.join([str(shim_dir), str(ffmpeg_dir), path])
			if shutil.which("ffmpeg") is None:
				raise RuntimeError("ffmpeg command still not resolvable after PATH update.")
			return ffmpeg_exe
		except Exception:
			raise STTError(
				"ffmpeg is not installed or not available on PATH, "
				"and the imageio-ffmpeg bundle could not provide it. "
				"Install ffmpeg: https://ffmpeg.org/download.html"
			)

	@staticmethod
	def _patch_whisper_audio_runner(ffmpeg_exe: str) -> None:
		"""Patch whisper.audio.run so it always invokes the absolute ffmpeg executable."""
		try:
			whisper_audio = importlib.import_module("whisper.audio")
		except ImportError:
			return

		def _run_with_pinned_ffmpeg(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
			if cmd and cmd[0] == "ffmpeg":
				cmd = [ffmpeg_exe, *cmd[1:]]
			return subprocess.run(cmd, *args, **kwargs)

		setattr(whisper_audio, "run", _run_with_pinned_ffmpeg)

	@staticmethod
	def _prepend_wav_header(pcm_bytes: bytes, sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
		"""Wrap raw PCM bytes in a RIFF/WAV container so ffmpeg can decode them."""
		data_size = len(pcm_bytes)
		byte_rate = sample_rate * num_channels * bits_per_sample // 8
		block_align = num_channels * bits_per_sample // 8
		header = struct.pack(
			"<4sI4s4sIHHIIHH4sI",
			b"RIFF",
			36 + data_size,
			b"WAVE",
			b"fmt ",
			16,
			1,  # PCM
			num_channels,
			sample_rate,
			byte_rate,
			block_align,
			bits_per_sample,
			b"data",
			data_size,
		)
		return header + pcm_bytes


