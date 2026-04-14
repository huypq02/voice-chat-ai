import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from voicechatai.domain.ports.stt_port import STTError, STTValidationError
from voicechatai.infrastructure.stt.whisper_stt import WhisperSTT


class FakeWhisperModel:
    def __init__(self, result: Any = None, error: Exception | None = None):
        self.result = result if result is not None else {"text": "  transcribed text  "}
        self.error = error
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def transcribe(self, audio: Any, **kwargs: Any) -> Any:
        self.calls.append((audio, kwargs))
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


def _make_adapter(fake_module: FakeWhisperModule, **kwargs: Any) -> WhisperSTT:
    """Construct a WhisperSTT with the whisper import patched out."""
    with patch(
        "voicechatai.infrastructure.stt.whisper_stt.importlib.import_module",
        return_value=fake_module,
    ):
        return WhisperSTT(**kwargs)


class WhisperSTTTests(unittest.TestCase):
    def test_constructor_rejects_blank_model_name(self) -> None:
        with self.assertRaises(ValueError):
            WhisperSTT(model_name="   ")

    def test_constructor_rejects_invalid_task(self) -> None:
        with self.assertRaises(ValueError):
            WhisperSTT(task="summarize")

    def test_constructor_raises_controlled_error_when_whisper_is_missing(self) -> None:
        with patch(
            "voicechatai.infrastructure.stt.whisper_stt.importlib.import_module",
            side_effect=ImportError("missing whisper"),
        ):
            with self.assertRaises(STTError):
                WhisperSTT()

    def test_transcribe_riff_wav_uses_temp_file_path(self) -> None:
        """RIFF/WAV bytes are written to a temp file and the path is passed to the model."""
        fake_module = FakeWhisperModule()
        adapter = _make_adapter(fake_module, model_name="tiny", language="en", fp16=False, audio_suffix="wav")

        riff_bytes = b"RIFF" + b"\x00" * 40  # minimal RIFF stub
        transcript = adapter.transcribe(riff_bytes)

        self.assertEqual(transcript, "transcribed text")
        self.assertEqual(fake_module.load_calls, ["tiny"])
        self.assertEqual(len(fake_module.model.calls), 1)

        audio_arg, options = fake_module.model.calls[0]
        self.assertIsInstance(audio_arg, str)
        self.assertTrue(audio_arg.endswith(".wav"))
        self.assertFalse(Path(audio_arg).exists())  # temp file cleaned up
        self.assertEqual(options, {"task": "transcribe", "fp16": False, "language": "en"})

    def test_transcribe_raw_pcm_passes_numpy_array_to_model(self) -> None:
        """Raw PCM bytes (no RIFF header) are decoded to float32 and passed directly."""
        fake_module = FakeWhisperModule()
        adapter = _make_adapter(fake_module, model_name="tiny", fp16=False, audio_suffix="wav")

        # Two int16 samples: 16384 → 0.5, -16384 → -0.5
        pcm_bytes = b"\x00\x40\x00\xc0"
        transcript = adapter.transcribe(pcm_bytes)

        self.assertEqual(transcript, "transcribed text")
        audio_arg, options = fake_module.model.calls[0]
        self.assertIsInstance(audio_arg, np.ndarray)
        self.assertEqual(audio_arg.dtype, np.float32)
        np.testing.assert_allclose(audio_arg, [0.5, -0.5], atol=1e-5)
        self.assertEqual(options["task"], "transcribe")

    def test_transcribe_rejects_empty_audio(self) -> None:
        adapter = _make_adapter(FakeWhisperModule())
        with self.assertRaises(STTValidationError):
            adapter.transcribe(b"")

    def test_transcribe_rejects_non_bytes_input(self) -> None:
        adapter = _make_adapter(FakeWhisperModule())
        with self.assertRaises(STTValidationError):
            adapter.transcribe(cast(bytes, None))

    def test_constructor_wraps_model_load_failures(self) -> None:
        fake_module = FakeWhisperModule(load_error=RuntimeError("bad model"))
        with self.assertRaises(STTError):
            _make_adapter(fake_module, model_name="broken")

    def test_transcribe_wraps_model_failures(self) -> None:
        fake_module = FakeWhisperModule(model=FakeWhisperModel(error=RuntimeError("decode failed")))
        adapter = _make_adapter(fake_module)
        with self.assertRaises(STTError):
            adapter.transcribe(b"audio-bytes")

    def test_transcribe_rejects_blank_whisper_output(self) -> None:
        fake_module = FakeWhisperModule(model=FakeWhisperModel(result={"text": "   "}))
        adapter = _make_adapter(fake_module)
        with self.assertRaises(STTError):
            adapter.transcribe(b"audio-bytes")


if __name__ == "__main__":
    unittest.main()
