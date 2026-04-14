from __future__ import annotations

from dataclasses import dataclass

from voicechatai.domain.ports.stt_port import STTError, STTPort


@dataclass(slots=True)
class ProcessVoiceQuery:
	"""Executes the Sprint 1 local speech-to-text happy path."""

	stt: STTPort

	def execute(self, audio_bytes: bytes) -> str:
		try:
			transcript = self.stt.transcribe(audio_bytes)
		except STTError:
			raise
		except Exception as exc:
			raise STTError("Speech-to-text processing failed.") from exc

		if not isinstance(transcript, str) or not transcript.strip():
			raise STTError("Speech-to-text produced an empty transcript.")

		return transcript


