"""Shared test fixtures for SnipClip."""

from pathlib import Path
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
