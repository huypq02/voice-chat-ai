import unittest
from typing import Any, cast
from unittest.mock import patch

from voicechatai.domain.ports.stt_port import STTError, STTValidationError
from voicechatai.infrastructure.stt.whisper_stt import WhisperSTT


class FakeWhisperModel:
    def __init__(self, result: Any = None, error: Exception | None = None):
        self.result = result if result is not None else {"text": "  transcribed text  "}
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def transcribe(self, audio_path: str, **kwargs: Any) -> Any:
        self.calls.append((audio_path, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


class FakeWhisperModule:
    def __init__(self, model: FakeWhisperModel | None = None, load_error: Exception | None = None):
        self.model = model or FakeWhisperModel()
        self.load_error = load_error
        self.load_calls: list[str] = []

    def load_model(self, model_name: str) -> FakeWhisperModel:
        self.load_calls.append(model_name)
        if self.load_error is not None:
            raise self.load_error
        return self.model


class WhisperSTTTests(unittest.TestCase):
    def test_constructor_rejects_blank_model_name(self) -> None:
        with self.assertRaises(ValueError):
            WhisperSTT(model_name="   ")

    def test_constructor_rejects_invalid_task(self) -> None:
        with self.assertRaises(ValueError):
            WhisperSTT(task="summarize")

    def test_constructor_raises_controlled_error_when_ffmpeg_is_missing(self) -> None:
        with patch("voicechatai.infrastructure.stt.whisper_stt.shutil.which", return_value=None):
            with self.assertRaises(STTError):
                WhisperSTT()

    def test_constructor_raises_controlled_error_when_whisper_is_missing(self) -> None:
        with patch(
            "voicechatai.infrastructure.stt.whisper_stt.shutil.which",
            return_value="C:/ffmpeg.exe",
        ), patch(
            "voicechatai.infrastructure.stt.whisper_stt.importlib.import_module",
            side_effect=ImportError("missing whisper"),
        ):
            with self.assertRaises(STTError):
                WhisperSTT()

    def test_transcribe_uses_whisper_model_and_cleans_up_temp_file(self) -> None:
        fake_module = FakeWhisperModule()

        with patch(
            "voicechatai.infrastructure.stt.whisper_stt.shutil.which",
            return_value="C:/ffmpeg.exe",
        ), patch(
            "voicechatai.infrastructure.stt.whisper_stt.importlib.import_module",
            return_value=fake_module,
        ):
            adapter = WhisperSTT(model_name="tiny", language="en", fp16=False, audio_suffix="wav")

            transcript = adapter.transcribe(b"audio-bytes")

        self.assertEqual(transcript, "transcribed text")
        self.assertEqual(fake_module.load_calls, ["tiny"])
        self.assertEqual(len(fake_module.model.calls), 1)

        audio_path, options = fake_module.model.calls[0]
        self.assertTrue(audio_path.endswith(".wav"))
        self.assertFalse(__import__("pathlib").Path(audio_path).exists())
        self.assertEqual(
            options,
            {
                "task": "transcribe",
                "fp16": False,
                "language": "en",
            },
        )

    def test_transcribe_rejects_empty_audio(self) -> None:
        fake_module = FakeWhisperModule()

        with patch(
            "voicechatai.infrastructure.stt.whisper_stt.shutil.which",
            return_value="C:/ffmpeg.exe",
        ), patch(
            "voicechatai.infrastructure.stt.whisper_stt.importlib.import_module",
            return_value=fake_module,
        ):
            adapter = WhisperSTT()

        with self.assertRaises(STTValidationError):
            adapter.transcribe(b"")

    def test_transcribe_rejects_non_bytes_input(self) -> None:
        fake_module = FakeWhisperModule()

        with patch(
            "voicechatai.infrastructure.stt.whisper_stt.shutil.which",
            return_value="C:/ffmpeg.exe",
        ), patch(
            "voicechatai.infrastructure.stt.whisper_stt.importlib.import_module",
            return_value=fake_module,
        ):
            adapter = WhisperSTT()

        with self.assertRaises(STTValidationError):
            adapter.transcribe(cast(bytes, None))

    def test_constructor_wraps_model_load_failures(self) -> None:
        fake_module = FakeWhisperModule(load_error=RuntimeError("bad model"))

        with patch(
            "voicechatai.infrastructure.stt.whisper_stt.shutil.which",
            return_value="C:/ffmpeg.exe",
        ), patch(
            "voicechatai.infrastructure.stt.whisper_stt.importlib.import_module",
            return_value=fake_module,
        ):
            with self.assertRaises(STTError):
                WhisperSTT(model_name="broken")

    def test_transcribe_wraps_model_failures(self) -> None:
        fake_module = FakeWhisperModule(model=FakeWhisperModel(error=RuntimeError("decode failed")))

        with patch(
            "voicechatai.infrastructure.stt.whisper_stt.shutil.which",
            return_value="C:/ffmpeg.exe",
        ), patch(
            "voicechatai.infrastructure.stt.whisper_stt.importlib.import_module",
            return_value=fake_module,
        ):
            adapter = WhisperSTT()

        with self.assertRaises(STTError):
            adapter.transcribe(b"audio-bytes")

    def test_transcribe_rejects_blank_whisper_output(self) -> None:
        fake_module = FakeWhisperModule(model=FakeWhisperModel(result={"text": "   "}))

        with patch(
            "voicechatai.infrastructure.stt.whisper_stt.shutil.which",
            return_value="C:/ffmpeg.exe",
        ), patch(
            "voicechatai.infrastructure.stt.whisper_stt.importlib.import_module",
            return_value=fake_module,
        ):
            adapter = WhisperSTT()

        with self.assertRaises(STTError):
            adapter.transcribe(b"audio-bytes")


if __name__ == "__main__":
    unittest.main()
