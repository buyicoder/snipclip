"""Tests for video cutting module."""

import json
from pathlib import Path
import pytest
from snipclip.cutter import cut_video, TimeRange
from tests.conftest import ffmpeg_required, ffprobe_required


class TestTimeRange:
    def test_time_range_serializable(self):
        """TimeRange can be serialized to/from JSON."""
        tr = TimeRange(start=1.5, end=3.5)
        data = {"start": tr.start, "end": tr.end}
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["start"] == 1.5
        assert parsed["end"] == 3.5

    def test_duration_property(self):
        """duration property returns end - start."""
        tr = TimeRange(start=2.0, end=5.0)
        assert tr.duration == 3.0


class TestCutVideoValidation:
    def test_raises_on_empty_segments(self, tmp_path):
        """Raises ValueError for empty segment list."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        with pytest.raises(ValueError, match="empty"):
            cut_video(video, [], tmp_path / "out.mp4", mode="keep")

    def test_raises_on_invalid_segment(self, tmp_path):
        """Raises ValueError when start >= end."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        with pytest.raises(ValueError, match="start.*end"):
            cut_video(
                video,
                [TimeRange(start=5.0, end=1.0)],
                tmp_path / "out.mp4",
                mode="keep",
            )

    def test_raises_on_negative_start(self, tmp_path):
        """Raises ValueError when start < 0."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        with pytest.raises(ValueError):
            cut_video(
                video,
                [TimeRange(start=-1.0, end=2.0)],
                tmp_path / "out.mp4",
                mode="keep",
            )

    def test_raises_on_nonexistent_file(self, tmp_path):
        """Raises FileNotFoundError for missing video."""
        with pytest.raises(FileNotFoundError):
            cut_video(
                tmp_path / "nope.mp4",
                [TimeRange(start=1.0, end=2.0)],
                tmp_path / "out.mp4",
            )


class TestCutVideoUnit:
    def test_cut_keep_writes_concat_file(self, mocker, tmp_path):
        """keep mode writes proper concat file and invokes ffmpeg."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0)

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video data")

        mocker.patch("snipclip.cutter.get_ffmpeg_path", return_value=Path("/bin/ffmpeg"))

        output = tmp_path / "out.mp4"
        segments = [TimeRange(start=1.0, end=3.0), TimeRange(start=4.0, end=5.0)]

        result = cut_video(video, segments, output, mode="keep")

        assert result == output
        # Verify ffmpeg concat invocation
        call_args = mock_run.call_args[0][0]
        assert "-f" in call_args
        assert "concat" in call_args
        assert "-c" in call_args
        assert "copy" in call_args


class TestCutVideoIntegration:
    @ffmpeg_required
    @ffprobe_required
    def test_keep_single_segment(self, test_video, tmp_path):
        """Keep one segment from the middle of the video."""
        output = tmp_path / "output.mp4"
        segments = [TimeRange(start=1.0, end=3.0)]

        result = cut_video(test_video, segments, output, mode="keep")

        assert result == output
        assert result.exists()
        assert result.stat().st_size > 0

    @ffmpeg_required
    @ffprobe_required
    def test_keep_multiple_segments(self, test_video, tmp_path):
        """Keep two disjoint segments."""
        output = tmp_path / "output.mp4"
        segments = [
            TimeRange(start=0.5, end=1.5),
            TimeRange(start=3.0, end=4.0),
        ]

        result = cut_video(test_video, segments, output, mode="keep")
        assert result.exists()
