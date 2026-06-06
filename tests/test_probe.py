"""Tests for video probe module."""

import json
from pathlib import Path
import pytest
from snipclip.probe import probe_video, VideoInfo
from tests.conftest import ffprobe_required


class TestVideoInfo:
    def test_video_info_fields(self):
        """VideoInfo has all expected namedtuple fields."""
        info = VideoInfo(
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            video_codec="h264",
            audio_codec="aac",
            sample_rate=44100,
            bitrate=5000000,
        )
        assert info.duration == 10.0
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30.0
        assert info.video_codec == "h264"
        assert info.audio_codec == "aac"
        assert info.sample_rate == 44100
        assert info.bitrate == 5000000

    def test_video_info_is_serializable(self):
        """VideoInfo can be serialized to JSON."""
        info = VideoInfo(
            duration=5.0, width=640, height=480, fps=30.0,
            video_codec="h264", audio_codec="aac",
            sample_rate=44100, bitrate=1000000,
        )
        data = info._asdict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["duration"] == 5.0
        assert parsed["width"] == 640


class TestProbeVideoUnit:
    def test_raises_on_nonexistent_file(self, tmp_path):
        """Raises FileNotFoundError for nonexistent video."""
        bad_path = tmp_path / "does_not_exist.mp4"
        with pytest.raises(FileNotFoundError):
            probe_video(bad_path)

    def test_parses_ffprobe_output(self, mocker):
        """Correctly parses FFprobe JSON output."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0, stdout=json.dumps({
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "10.5",
                    "avg_frame_rate": "30000/1001",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "48000",
                },
            ],
            "format": {"bit_rate": "5000000"},
        }))
        # Mock the ffprobe path resolution
        mocker.patch("snipclip.probe.get_ffprobe_path", return_value=Path("/bin/ffprobe"))
        # Mock Path.exists for the input file check
        mocker.patch.object(Path, "exists", return_value=True)

        info = probe_video(Path("/fake/video.mp4"))

        assert info.duration == pytest.approx(10.5)
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == pytest.approx(29.97, abs=0.1)
        assert info.video_codec == "h264"
        assert info.audio_codec == "aac"
        assert info.sample_rate == 48000
        assert info.bitrate == 5000000


class TestProbeVideoIntegration:
    @ffprobe_required
    def test_returns_video_info_for_valid_file(self, test_video):
        """Probe a real test video and verify all fields are populated."""
        info = probe_video(test_video)

        assert isinstance(info, VideoInfo)
        assert info.duration > 0
        assert info.duration == pytest.approx(5.0, abs=0.5)
        assert info.width > 0
        assert info.height > 0
        assert info.fps > 0
        assert len(info.video_codec) > 0
