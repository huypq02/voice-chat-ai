import unittest

from voicechatai.application.use_cases.process_voice_query import ProcessVoiceQuery
from voicechatai.domain.ports.stt_port import STTError, STTPort


class StubSTT(STTPort):
    def __init__(self, transcript: str = "hello"):
        self.transcript = transcript
        self.calls = []

    def transcribe(self, audio_bytes: bytes) -> str:
        self.calls.append(audio_bytes)
        return self.transcript


class FailingSTT(STTPort):
    def transcribe(self, audio_bytes: bytes) -> str:
        raise STTError("adapter failure")


class ExplodingSTT(STTPort):
    def transcribe(self, audio_bytes: bytes) -> str:
        raise RuntimeError("unexpected failure")


class BlankTranscriptSTT(STTPort):
    def transcribe(self, audio_bytes: bytes) -> str:
        return "   "


class ProcessVoiceQuerySTTTests(unittest.TestCase):
    def test_execute_invokes_stt_port_and_returns_transcript(self) -> None:
        stub = StubSTT(transcript="transcribed text")
        use_case = ProcessVoiceQuery(stt=stub)

        transcript = use_case.execute(b"audio")

        self.assertEqual(transcript, "transcribed text")
        self.assertEqual(stub.calls, [b"audio"])

    def test_execute_propagates_controlled_stt_failures(self) -> None:
        use_case = ProcessVoiceQuery(stt=FailingSTT())

        with self.assertRaises(STTError):
            use_case.execute(b"audio")

    def test_execute_wraps_unexpected_failures_as_stt_error(self) -> None:
        use_case = ProcessVoiceQuery(stt=ExplodingSTT())

        with self.assertRaises(STTError):
            use_case.execute(b"audio")

    def test_execute_rejects_blank_transcripts(self) -> None:
        use_case = ProcessVoiceQuery(stt=BlankTranscriptSTT())

        with self.assertRaises(STTError):
            use_case.execute(b"audio")


if __name__ == "__main__":
    unittest.main()
