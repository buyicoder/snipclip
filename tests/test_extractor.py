"""Tests for audio extraction module."""

import json
from pathlib import Path
import pytest
from snipclip.extractor import extract_audio
from tests.conftest import ffmpeg_required


class TestExtractAudioUnit:
    def test_raises_on_nonexistent_input(self, tmp_path):
        """Raises FileNotFoundError for missing video."""
        with pytest.raises(FileNotFoundError):
            extract_audio(tmp_path / "nope.mp4", tmp_path / "out.wav")

    def test_calls_ffmpeg_with_correct_args(self, mocker, tmp_path):
        """Calls FFmpeg with correct arguments for 16kHz mono WAV."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0)

        # Create a fake input file
        input_file = tmp_path / "test.mp4"
        input_file.write_bytes(b"fake video")
        output_file = tmp_path / "audio.wav"

        mocker.patch("snipclip.extractor.get_ffmpeg_path", return_value=Path("/bin/ffmpeg"))

        result = extract_audio(input_file, output_file)

        assert result == output_file
        # Verify ffmpeg was called
        call_args = mock_run.call_args[0][0]
        assert "-vn" in call_args
        assert "-acodec" in call_args
        assert "pcm_s16le" in call_args
        assert "-ar" in call_args
        assert "16000" in call_args
        assert "-ac" in call_args
        assert "1" in call_args

    def test_extract_audio_creates_parent_dirs(self, mocker, tmp_path):
        """Creates output directory if it doesn't exist."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0)

        input_file = tmp_path / "test.mp4"
        input_file.write_bytes(b"fake")
        output_file = tmp_path / "subdir" / "nested" / "audio.wav"

        mocker.patch("snipclip.extractor.get_ffmpeg_path", return_value=Path("/bin/ffmpeg"))

        extract_audio(input_file, output_file)
        assert output_file.parent.exists()


class TestExtractAudioIntegration:
    @ffmpeg_required
    def test_extracts_wav_from_video(self, test_video, tmp_path):
        """Extract audio and verify output is 16kHz mono WAV."""
        output = tmp_path / "audio.wav"
        result = extract_audio(test_video, output)

        assert result == output
        assert result.exists()
        assert result.stat().st_size > 0

    @ffmpeg_required
    def test_overwrites_existing_file(self, test_video, tmp_path):
        """Overwrites existing output file."""
        output = tmp_path / "audio.wav"
        output.write_bytes(b"garbage")

        result = extract_audio(test_video, output)

        assert result.exists()
        assert result.stat().st_size > 100
