"""Tests for subtitle generation module."""

from pathlib import Path
import pytest
from snipclip.subtitler import generate_srt, burn_subtitles, Segment
from tests.conftest import ffmpeg_required


class TestGenerateSrt:
    def test_generates_valid_srt(self, tmp_path):
        """Generate SRT from segments and verify file content."""
        segments = [
            Segment(start=0.0, end=2.5, text="Hello world", confidence=0.95),
            Segment(start=3.0, end=5.5, text="This is a test", confidence=0.90),
        ]
        output = tmp_path / "test.srt"

        result = generate_srt(segments, output)

        assert result == output
        assert result.exists()

        content = result.read_text()
        assert content.startswith("1\n")
        assert "00:00:00,000 --> 00:00:02,500" in content
        assert "Hello world" in content
        assert "2\n" in content
        assert "00:00:03,000 --> 00:00:05,500" in content
        assert "This is a test" in content

    def test_empty_segments_produces_empty_srt(self, tmp_path):
        """Empty segment list produces empty SRT file."""
        output = tmp_path / "empty.srt"
        result = generate_srt([], output)
        assert result.exists()
        assert result.read_text() == ""

    def test_srt_timestamp_formatting(self, tmp_path):
        """Verify SRT timestamp format is HH:MM:SS,mmm."""
        segments = [Segment(start=3661.5, end=3662.0, text="one hour in", confidence=1.0)]
        output = tmp_path / "time.srt"
        generate_srt(segments, output)
        content = output.read_text()
        assert "01:01:01,500 --> 01:01:02,000" in content

    def test_multiline_text_preserved(self, tmp_path):
        """Multi-word text is preserved in SRT."""
        segments = [
            Segment(start=0.0, end=1.0,
                    text="This is a longer sentence with many words",
                    confidence=0.99),
        ]
        output = tmp_path / "long.srt"
        generate_srt(segments, output)
        content = output.read_text()
        assert "This is a longer sentence with many words" in content


class TestBurnSubtitles:
    @ffmpeg_required
    def test_burns_subtitles_into_video(self, test_video, tmp_path):
        """Burn subtitles and verify output exists."""
        segments = [
            Segment(start=0.0, end=2.0, text="Test subtitle", confidence=0.9),
        ]
        output = tmp_path / "burned.mp4"

        result = burn_subtitles(test_video, segments, output)

        assert result == output
        assert result.exists()
        assert result.stat().st_size > 0

    def test_raises_on_nonexistent_video(self, tmp_path):
        """Raises FileNotFoundError for missing video."""
        with pytest.raises(FileNotFoundError):
            burn_subtitles(
                tmp_path / "nope.mp4",
                [Segment(start=0.0, end=1.0, text="test", confidence=1.0)],
                tmp_path / "out.mp4",
            )
