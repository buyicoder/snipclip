"""Tests for CLI interface."""

import json
from pathlib import Path
import pytest
from click.testing import CliRunner
from snipclip.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestCliHelp:
    def test_help_shows_commands(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "probe" in result.output
        assert "cut" in result.output
        assert "transcribe" in result.output
        assert "subtitle" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestCliProbe:
    def test_probe_nonexistent_file(self, runner, tmp_path):
        result = runner.invoke(
            main, ["probe", str(tmp_path / "nope.mp4")]
        )
        assert result.exit_code != 0

    @pytest.mark.skip(reason="Needs FFprobe and test video")
    def test_probe_real_video(self, runner):
        """Integration: probe a real video file."""
        pass


class TestCliCut:
    def test_cut_requires_keep_or_remove(self, runner, test_video):
        """Cut requires --keep or --remove flag."""
        result = runner.invoke(main, ["cut", str(test_video)])
        assert result.exit_code != 0
        assert "--keep" in result.output or "--remove" in result.output

    def test_cut_cannot_use_both(self, runner, test_video, tmp_path):
        """Cannot use --keep and --remove together."""
        seg_file = tmp_path / "seg.json"
        seg_file.write_text('[{"start": 0, "end": 1}]')
        result = runner.invoke(
            main,
            [
                "cut", str(test_video),
                "--keep", str(seg_file),
                "--remove", str(seg_file),
            ],
        )
        assert result.exit_code != 0


class TestCliSubtitle:
    def test_subtitle_with_bad_transcript(self, runner, test_video, tmp_path):
        """Shows error with malformed transcript."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        result = runner.invoke(
            main,
            ["subtitle", str(test_video), str(bad_file)],
        )
        assert result.exit_code != 0


class TestCliSetup:
    def test_setup_help(self, runner):
        result = runner.invoke(main, ["setup", "--help"])
        assert result.exit_code == 0
