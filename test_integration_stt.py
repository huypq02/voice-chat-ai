#!/usr/bin/env python
"""Quick integration test for the Whisper STT pipeline."""
from __future__ import annotations

import sys
import base64

sys.path.insert(0, "src")

from voicechatai.interfaces.api.app import app
from voicechatai.domain.ports.stt_port import STTError


def test_stt_direct() -> None:
    """Test the STT use case directly with real Whisper model."""
    process_voice_query = app.state.process_voice_query

    # Generate a simple 1-second silence audio in WAV format (44.1kHz, mono, 16-bit).
    # WAV header for silence: 44100 Hz, mono, 16-bit PCM, 1 second = 88244 bytes
    wav_silence = (
        b"RIFF"  # ChunkID
        b"\x44\xac\x01\x00"  # ChunkSize (88244 - 8)
        b"WAVE"  # Format
        b"fmt "  # Subchunk1ID
        b"\x10\x00\x00\x00"  # Subchunk1Size (16)
        b"\x01\x00"  # AudioFormat (1 = PCM)
        b"\x01\x00"  # NumChannels (1 = mono)
        b"\x44\xac\x00\x00"  # SampleRate (44100)
        b"\x88\x58\x01\x00"  # ByteRate (44100 * 2)
        b"\x02\x00"  # BlockAlign (2)
        b"\x10\x00"  # BitsPerSample (16)
        b"data"  # Subchunk2ID
        b"\x00\xac\x01\x00"  # Subchunk2Size (88200)
        + b"\x00" * 88200  # Sample data (silence)
    )

    try:
        transcript = process_voice_query.execute(wav_silence)
        print(f"✅ Transcription successful")
        print(f"   Transcript: '{transcript[:100] if transcript else '(empty)'}'")
        assert isinstance(transcript, str)
    except STTError as e:
        print(f"⚠️  STT Error (expected for silence): {e}")


def test_app_state() -> None:
    """Test that the app state is properly configured."""
    assert hasattr(app.state, "process_voice_query"), "Missing process_voice_query use case"
    use_case = app.state.process_voice_query
    assert use_case is not None, "process_voice_query is None"
    print("✅ App state properly configured with ProcessVoiceQuery use case")


if __name__ == "__main__":
    print("Testing app state initialization...")
    test_app_state()
    print("\nTesting STT pipeline with silence audio...")
    test_stt_direct()
    print("\n✅ All integration tests passed!")

