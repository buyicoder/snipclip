"""Tests for FFmpeg discovery helper."""

import subprocess
from pathlib import Path
import pytest
from snipclip._ffmpeg import (
    find_ffmpeg,
    find_ffprobe,
    get_ffmpeg_path,
    get_ffprobe_path,
    check_ffmpeg,
    FFmpegNotFoundError,
)


class TestFindFfmpeg:
    def test_find_ffmpeg_in_path(self, mocker):
        """ffmpeg is found when it's in PATH."""
        mock_which = mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")
        result = find_ffmpeg()
        assert result == Path("/usr/bin/ffmpeg")
        mock_which.assert_called_once_with("ffmpeg")

    def test_find_ffmpeg_not_in_path_falls_back_to_local(self, mocker):
        """When not in PATH, checks ~/.snipclip/bin/ffmpeg."""
        mocker.patch("shutil.which", return_value=None)
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "home", return_value=Path("/home/user"))

        result = find_ffmpeg()

        expected = Path("/home/user/.snipclip/bin/ffmpeg")
        assert str(result) == str(expected)

    def test_find_ffmpeg_not_found_anywhere(self, mocker):
        """Raises FFmpegNotFoundError when ffmpeg is nowhere."""
        mocker.patch("shutil.which", return_value=None)
        mocker.patch.object(Path, "exists", return_value=False)

        with pytest.raises(FFmpegNotFoundError) as exc:
            find_ffmpeg()
        assert "snipclip setup" in str(exc.value)


class TestFindFfprobe:
    def test_find_ffprobe_in_path(self, mocker):
        mock_which = mocker.patch("shutil.which", return_value="/usr/bin/ffprobe")
        result = find_ffprobe()
        assert result == Path("/usr/bin/ffprobe")


class TestGetPaths:
    def test_get_ffmpeg_path_caches_result(self, mocker):
        # Clear cache before test
        import snipclip._ffmpeg as mod
        mod._ffmpeg_path = None

        mock_find = mocker.patch.object(
            mod, "find_ffmpeg", return_value=Path("/bin/ffmpeg")
        )
        try:
            result1 = get_ffmpeg_path()
            result2 = get_ffmpeg_path()
            assert result1 == Path("/bin/ffmpeg")
            assert result2 == Path("/bin/ffmpeg")
            mock_find.assert_called_once()  # cached on second call
        finally:
            # Reset global cache to avoid polluting other tests
            mod._ffmpeg_path = None


class TestCheckFfmpeg:
    def test_check_ffmpeg_runs_version(self, mocker):
        """check_ffmpeg returns True when ffmpeg -version succeeds."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0, stdout="ffmpeg version 6.0")

        result = check_ffmpeg(Path("/bin/ffmpeg"))

        assert result is True
        mock_run.assert_called_once_with(
            [str(Path("/bin/ffmpeg")), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_check_ffmpeg_fails_on_bad_binary(self, mocker):
        mock_run = mocker.patch("subprocess.run")
        mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 10)

        result = check_ffmpeg(Path("/bin/ffmpeg"))

        assert result is False
