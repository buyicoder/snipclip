"""Tests for transcription module."""

import json
import struct
from pathlib import Path
import pytest
from snipclip.transcriber import transcribe, Segment, detect_device


class TestSegment:
    def test_segment_fields(self):
        seg = Segment(start=1.0, end=2.5, text="hello world", confidence=0.95)
        assert seg.start == 1.0
        assert seg.end == 2.5
        assert seg.text == "hello world"
        assert seg.confidence == 0.95

    def test_segment_json_roundtrip(self):
        seg = Segment(start=0.5, end=3.0, text="test", confidence=0.8)
        data = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "confidence": seg.confidence,
        }
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["start"] == 0.5
        assert parsed["end"] == 3.0
        assert parsed["text"] == "test"
        assert parsed["confidence"] == 0.8


class TestDetectDevice:
    def test_detect_device_returns_str(self):
        device = detect_device()
        assert isinstance(device, str)
        assert device in ("cpu", "cuda")

    def test_detect_device_cpu_when_torch_not_installed(self):
        """Returns cpu when torch is not installed at all."""
        # This is the actual behavior on systems without torch
        assert detect_device() == "cpu"

    def test_detect_device_cpu_when_no_torch(self, mocker):
        mocker.patch.dict("sys.modules", {"torch": None})
        # Should gracefully fall back to cpu
        assert detect_device() == "cpu"


def _make_minimal_wav(path: Path, duration_sec: float = 1.0) -> None:
    """Create a minimal valid WAV file (16kHz, 16-bit, mono, silence)."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))     # chunk size
        f.write(struct.pack("<H", 1))      # PCM
        f.write(struct.pack("<H", 1))      # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))  # byte rate
        f.write(struct.pack("<H", 2))      # block align
        f.write(struct.pack("<H", 16))     # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


class TestTranscribe:
    def test_transcribe_mocked(self, mocker, tmp_path):
        """Transcribe with mocked faster-whisper model."""
        audio_path = tmp_path / "test.wav"
        _make_minimal_wav(audio_path)

        mock_model = mocker.MagicMock()
        mock_segments = [
            mocker.MagicMock(start=0.0, end=1.0, text="hello", no_speech_prob=0.1),
            mocker.MagicMock(start=1.0, end=2.0, text="world", no_speech_prob=0.2),
        ]
        mock_model.transcribe.return_value = (mock_segments, {"language": "en"})
        mocker.patch(
            "snipclip.transcriber.WhisperModel", return_value=mock_model
        )

        segments = transcribe(audio_path, model_size="tiny")

        assert len(segments) == 2
        assert segments[0].text == "hello"
        assert segments[0].start == 0.0
        assert segments[0].end == 1.0
        assert segments[1].text == "world"

    def test_transcribe_skips_silence(self, mocker, tmp_path):
        """Segments with high no_speech_prob are filtered out."""
        audio_path = tmp_path / "test.wav"
        _make_minimal_wav(audio_path)

        mock_model = mocker.MagicMock()
        mock_segments = [
            mocker.MagicMock(start=0.0, end=1.0, text="speech", no_speech_prob=0.05),
            mocker.MagicMock(start=1.0, end=2.0, text="", no_speech_prob=0.95),
            mocker.MagicMock(start=2.0, end=3.0, text="more", no_speech_prob=0.1),
        ]
        mock_model.transcribe.return_value = (mock_segments, {"language": "en"})
        mocker.patch(
            "snipclip.transcriber.WhisperModel", return_value=mock_model
        )

        segments = transcribe(audio_path, model_size="tiny")

        assert len(segments) == 2
        assert segments[0].text == "speech"
        assert segments[1].text == "more"

    def test_transcribe_raises_on_missing_audio(self, tmp_path):
        """Raises FileNotFoundError for missing audio file."""
        with pytest.raises(FileNotFoundError):
            transcribe(tmp_path / "nope.wav")

    def test_transcribe_respects_device_param(self, mocker, tmp_path):
        """Passes device parameter to WhisperModel."""
        audio_path = tmp_path / "test.wav"
        _make_minimal_wav(audio_path)

        mock_model = mocker.MagicMock()
        mock_model.transcribe.return_value = ([], {"language": "en"})
        mock_whisper = mocker.patch(
            "snipclip.transcriber.WhisperModel", return_value=mock_model
        )

        transcribe(audio_path, model_size="base", device="cpu")

        mock_whisper.assert_called_once_with(
            "base", device="cpu", compute_type="int8"
        )
