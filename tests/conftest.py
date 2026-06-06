"""Shared test fixtures for SnipClip."""

import shutil
from pathlib import Path
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and functional."""
    return shutil.which("ffmpeg") is not None


def _ffprobe_available() -> bool:
    """Check if FFprobe is installed and functional."""
    return shutil.which("ffprobe") is not None


ffmpeg_required = pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="FFmpeg not installed. Run: snipclip setup",
)
ffprobe_required = pytest.mark.skipif(
    not _ffprobe_available(),
    reason="FFprobe not installed. Run: snipclip setup",
)


@pytest.fixture(scope="session")
def fixtures_dir():
    """Path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def test_video(fixtures_dir):
    """Path to the standard test video (created by setup step)."""
    path = fixtures_dir / "test_video.mp4"
    if not path.exists():
        pytest.skip("Test video not found. Run: python scripts/make_fixtures.py")
    return path
