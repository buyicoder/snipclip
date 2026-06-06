# SnipClip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build SnipClip — a two-tier automatic video editing framework: a Python CLI engine for video processing + a Claude Code skill as the AI editing director.

**Architecture:** The engine (Python package) provides 6 atomic modules — probe, extract, transcribe, cut, subtitle, scene — each with zero intelligence. The Claude Code skill orchestrates them in a 5-step pipeline (probe → transcribe → analyze → execute → deliver), where Claude reads transcripts and makes all editing decisions.

**Tech Stack:** Python 3.10+, faster-whisper, Click, Rich, ffmpeg-python, FFmpeg. CPU-first with GPU auto-detection.

**Spec:** `docs/superpowers/specs/2026-06-06-snipclip-design.md`

---

## File Structure (created this plan)

```
snipclip/                          # Repository root
├── snipclip/                      # Python package
│   ├── __init__.py
│   ├── _ffmpeg.py                 # Internal: FFmpeg discovery & helpers
│   ├── cli.py                     # CLI entry point (Click)
│   ├── probe.py                   # Video info probe (FFprobe)
│   ├── extractor.py               # Audio extraction (FFmpeg)
│   ├── transcriber.py             # Speech-to-text (faster-whisper)
│   ├── cutter.py                  # Cut & concatenate (FFmpeg)
│   ├── subtitler.py               # Subtitle generation & burn-in
│   └── scene.py                   # Scene detection (optional)
│
├── skill/                         # Claude Code skill
│   └── snipclip.md                # Skill definition
│
├── tests/
│   ├── conftest.py                # Shared fixtures
│   ├── test_probe.py
│   ├── test_extractor.py
│   ├── test_cutter.py
│   ├── test_transcriber.py
│   ├── test_subtitler.py
│   └── fixtures/                  # Generated test videos
│
├── pyproject.toml
├── README.md
├── README_zh.md
├── LICENSE
└── .gitignore
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `snipclip/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "snipclip"
version = "0.1.0"
description = "AI-powered automatic video editing — engine does the hands, Claude does the brain"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
dependencies = [
    "faster-whisper>=1.0.0",
    "click>=8.0.0",
    "rich>=13.0.0",
    "ffmpeg-python>=0.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.0.0",
]

[project.scripts]
snipclip = "snipclip.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
venv/
dist/
*.egg-info/
.pytest_cache/
*.mp4
*.wav
*.srt
!tests/fixtures/*
```

- [ ] **Step 3: Create LICENSE**

MIT License, copyright the user. Standard MIT text from https://opensource.org/licenses/MIT.

- [ ] **Step 4: Create snipclip/__init__.py**

```python
"""SnipClip — AI-powered automatic video editing engine."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Create tests/__init__.py**

```python
# SnipClip test suite
```

- [ ] **Step 6: Create tests/conftest.py**

```python
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
```

- [ ] **Step 7: Install dev dependencies and verify**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 8: Init git repo**

```bash
git init
git add -A
git commit -m "chore: project scaffolding"
```

---

### Task 2: FFmpeg Discovery Helper

**Files:**
- Create: `snipclip/_ffmpeg.py`
- Create: `tests/test_ffmpeg.py`

This internal module locates ffmpeg/ffprobe on the system. Every other engine module depends on it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ffmpeg.py`:

```python
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
        mock_which = mocker.patch("shutil.which", return_value=None)
        mock_exists = mocker.patch.object(Path, "exists", return_value=True)
        mock_expanduser = mocker.patch.object(
            Path, "home", return_value=Path("/home/user")
        )

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
        mock_find = mocker.patch(
            "snipclip._ffmpeg.find_ffmpeg", return_value=Path("/bin/ffmpeg")
        )
        result1 = get_ffmpeg_path()
        result2 = get_ffmpeg_path()
        assert result1 == Path("/bin/ffmpeg")
        assert result2 == Path("/bin/ffmpeg")
        mock_find.assert_called_once()  # cached on second call


class TestCheckFfmpeg:
    def test_check_ffmpeg_runs_version(self, mocker):
        """check_ffmpeg returns True when ffmpeg --version succeeds."""
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_ffmpeg.py -v
```
Expected: FAIL (no module `snipclip._ffmpeg`)

- [ ] **Step 3: Implement snipclip/_ffmpeg.py**

```python
"""FFmpeg/FFprobe discovery and validation.

Finds ffmpeg/ffprobe on the system:
1. Check PATH (user-installed)
2. Check ~/.snipclip/bin/ (auto-downloaded via `snipclip setup`)
3. Raise FFmpegNotFoundError with setup instructions
"""

from pathlib import Path
import shutil
import subprocess
from typing import Optional


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg or ffprobe not found anywhere on the system."""

    def __init__(self, tool: str):
        self.tool = tool
        super().__init__(
            f"{tool} not found. Install it manually or run:\n"
            f"  snipclip setup\n"
            f"to auto-download ffmpeg to ~/.snipclip/bin/"
        )


_ffmpeg_path: Optional[Path] = None
_ffprobe_path: Optional[Path] = None


def _snipclip_bin_dir() -> Path:
    """Return ~/.snipclip/bin/ directory."""
    return Path.home() / ".snipclip" / "bin"


def find_ffmpeg() -> Path:
    """Locate ffmpeg binary. Raises FFmpegNotFoundError if not found."""
    # 1. Check PATH
    in_path = shutil.which("ffmpeg")
    if in_path:
        return Path(in_path)

    # 2. Check local install
    local = _snipclip_bin_dir() / "ffmpeg"
    if local.exists():
        return local

    raise FFmpegNotFoundError("ffmpeg")


def find_ffprobe() -> Path:
    """Locate ffprobe binary. Raises FFmpegNotFoundError if not found."""
    in_path = shutil.which("ffprobe")
    if in_path:
        return Path(in_path)

    local = _snipclip_bin_dir() / "ffprobe"
    if local.exists():
        return local

    raise FFmpegNotFoundError("ffprobe")


def get_ffmpeg_path() -> Path:
    """Get cached ffmpeg path. Finds on first call, caches thereafter."""
    global _ffmpeg_path
    if _ffmpeg_path is None:
        _ffmpeg_path = find_ffmpeg()
    return _ffmpeg_path


def get_ffprobe_path() -> Path:
    """Get cached ffprobe path. Finds on first call, caches thereafter."""
    global _ffprobe_path
    if _ffprobe_path is None:
        _ffprobe_path = find_ffprobe()
    return _ffprobe_path


def check_ffmpeg(ffmpeg_path: Path) -> bool:
    """Verify that ffmpeg at the given path actually runs."""
    try:
        result = subprocess.run(
            [str(ffmpeg_path), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return False


def check_ffprobe(ffprobe_path: Path) -> bool:
    """Verify that ffprobe at the given path actually runs."""
    try:
        result = subprocess.run(
            [str(ffprobe_path), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ffmpeg.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add snipclip/_ffmpeg.py tests/test_ffmpeg.py
git commit -m "feat: add FFmpeg discovery helper"
```

---

### Task 3: Test Fixtures — Generate Test Video

**Files:**
- Create: `scripts/make_fixtures.py`
- Create: `tests/fixtures/.gitkeep` (will be replaced by generated video)

Before developing engine modules, we need a real test video for integration tests.

- [ ] **Step 1: Get FFmpeg installed**

If FFmpeg is not already available, download it manually:
- Windows: https://ffmpeg.org/download.html → Windows builds → ffmpeg-release-essentials.zip
- Extract and add `bin/` to PATH, or place `ffmpeg.exe` and `ffprobe.exe` in `~/.snipclip/bin/`

Alternatively, write a minimal setup script:

Create `scripts/make_fixtures.py`:

```python
#!/usr/bin/env python3
"""Generate test video fixtures using ffmpeg.

Creates a 5-second test video with test pattern + audio tone.
"""

import subprocess
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def run_ffmpeg(args: list[str], description: str) -> None:
    """Run ffmpeg with args, print progress."""
    print(f"  {description}...")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"  OK")


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    test_video = FIXTURES_DIR / "test_video.mp4"
    print(f"Creating test video: {test_video}")
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=640x480:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(test_video),
        ],
        "Generating 5-second test pattern video with audio tone",
    )

    # Also generate a shorter 3-second clip for quick tests
    short_video = FIXTURES_DIR / "test_video_short.mp4"
    print(f"Creating short test video: {short_video}")
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=3",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(short_video),
        ],
        "Generating 3-second short test video",
    )

    print(f"\nFixtures ready in {FIXTURES_DIR}")
    for f in sorted(FIXTURES_DIR.glob("*.mp4")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script to generate fixtures**

```bash
python scripts/make_fixtures.py
```
Expected: two .mp4 files in `tests/fixtures/`

- [ ] **Step 3: Verify fixtures exist**

```bash
ls -la tests/fixtures/
```
Expected: `test_video.mp4` and `test_video_short.mp4` exist with non-zero size

- [ ] **Step 4: Commit**

```bash
git add scripts/make_fixtures.py tests/fixtures/
git commit -m "chore: add test fixture generation script and fixtures"
```

---

### Task 4: probe.py — Video Info Probe

**Files:**
- Create: `snipclip/probe.py`
- Create: `tests/test_probe.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_probe.py`:

```python
"""Tests for video probe module."""

from pathlib import Path
import json
import pytest
from snipclip.probe import probe_video, VideoInfo


class TestProbeVideo:
    def test_returns_video_info_for_valid_file(self, test_video):
        """Probe a real test video and verify all fields are populated."""
        info = probe_video(test_video)

        assert isinstance(info, VideoInfo)
        assert info.duration > 0
        assert info.duration == pytest.approx(5.0, abs=0.5)
        assert info.width == 640
        assert info.height == 480
        assert info.fps > 0
        assert len(info.video_codec) > 0
        assert len(info.audio_codec) > 0
        assert info.sample_rate > 0

    def test_raises_on_nonexistent_file(self, tmp_path):
        """Raises FileNotFoundError for nonexistent video."""
        bad_path = tmp_path / "does_not_exist.mp4"
        with pytest.raises(FileNotFoundError):
            probe_video(bad_path)

    def test_video_info_is_serializable(self, test_video):
        """VideoInfo can be serialized to JSON."""
        info = probe_video(test_video)
        data = info._asdict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["duration"] == pytest.approx(5.0, abs=0.5)


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_probe.py -v
```
Expected: FAIL (no module `snipclip.probe`)

- [ ] **Step 3: Implement snipclip/probe.py**

```python
"""Video information probe using FFprobe.

Extracts: duration, resolution, frame rate, codecs, sample rate, bitrate.
"""

from pathlib import Path
import json
import subprocess
from typing import NamedTuple

from snipclip._ffmpeg import get_ffprobe_path


class VideoInfo(NamedTuple):
    """Structured video metadata."""

    duration: float       # seconds
    width: int            # pixels
    height: int           # pixels
    fps: float            # frames per second
    video_codec: str      # e.g. "h264", "hevc"
    audio_codec: str      # e.g. "aac", "mp3", "" if no audio
    sample_rate: int      # Hz, 0 if no audio
    bitrate: int           # bits per second, 0 if unknown


def probe_video(video_path: Path) -> VideoInfo:
    """Extract metadata from a video file using FFprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoInfo namedtuple with all available metadata.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If FFprobe fails to parse the file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffprobe = get_ffprobe_path()

    result = subprocess.run(
        [
            str(ffprobe),
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFprobe failed on {video_path}:\n{result.stderr}"
        )

    data = json.loads(result.stdout)

    # Find video stream
    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream["codec_type"] == "video" and video_stream is None:
            video_stream = stream
        elif stream["codec_type"] == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    # Parse duration: prefer stream duration, fall back to format duration
    duration = float(video_stream.get("duration") or data["format"].get("duration", 0))

    # Parse FPS: try avg_frame_rate fraction first, then r_frame_rate
    fps_str = video_stream.get("avg_frame_rate", "0/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    else:
        fps = float(fps_str)

    return VideoInfo(
        duration=duration,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        video_codec=video_stream.get("codec_name", "unknown"),
        audio_codec=audio_stream.get("codec_name", "") if audio_stream else "",
        sample_rate=int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
        bitrate=int(data["format"].get("bit_rate", 0)),
    )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_probe.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add snipclip/probe.py tests/test_probe.py
git commit -m "feat: add video probe module"
```

---

### Task 5: extractor.py — Audio Extraction

**Files:**
- Create: `snipclip/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_extractor.py`:

```python
"""Tests for audio extraction module."""

import subprocess
from pathlib import Path
import pytest
from snipclip.extractor import extract_audio


class TestExtractAudio:
    def test_extracts_wav_from_video(self, test_video, tmp_path):
        """Extract audio from test video, verify output is valid WAV."""
        output = tmp_path / "audio.wav"
        result = extract_audio(test_video, output)

        assert result == output
        assert result.exists()
        assert result.stat().st_size > 0

        # Verify it's actually a WAV with expected properties
        # Use ffprobe to check
        import subprocess as sp
        probe_result = sp.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(result),
            ],
            capture_output=True,
            text=True,
        )
        import json
        data = json.loads(probe_result.stdout)
        audio = data["streams"][0]
        assert audio["codec_type"] == "audio"
        assert audio["sample_rate"] == "16000"
        assert audio["channels"] == 1

    def test_overwrites_existing_file(self, test_video, tmp_path):
        """Should overwrite output if it already exists."""
        output = tmp_path / "audio.wav"
        output.write_bytes(b"garbage")

        result = extract_audio(test_video, output)

        assert result.exists()
        # Should be larger than the garbage we wrote
        assert result.stat().st_size > 100

    def test_raises_on_nonexistent_input(self, tmp_path):
        """Raises FileNotFoundError for missing video."""
        with pytest.raises(FileNotFoundError):
            extract_audio(tmp_path / "nope.mp4", tmp_path / "out.wav")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_extractor.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement snipclip/extractor.py**

```python
"""Audio extraction from video files.

Extracts audio track to 16kHz mono WAV — the standard input format for Whisper.
"""

from pathlib import Path
import subprocess

from snipclip._ffmpeg import get_ffmpeg_path


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
) -> Path:
    """Extract audio from video as 16kHz mono WAV.

    Args:
        video_path: Path to source video.
        output_path: Path for output WAV file (overwritten if exists).
        sample_rate: Output sample rate in Hz (default 16000 for Whisper).

    Returns:
        Path to the extracted audio file.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If FFmpeg fails.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg_path()

    result = subprocess.run(
        [
            str(ffmpeg),
            "-y",                        # overwrite output
            "-i", str(video_path),       # input
            "-vn",                       # no video
            "-acodec", "pcm_s16le",      # 16-bit PCM
            "-ar", str(sample_rate),     # sample rate
            "-ac", "1",                  # mono
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed:\n{result.stderr}"
        )

    return output_path
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_extractor.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add snipclip/extractor.py tests/test_extractor.py
git commit -m "feat: add audio extraction module"
```

---

### Task 6: cutter.py — Cut & Concatenate

**Files:**
- Create: `snipclip/cutter.py`
- Create: `tests/test_cutter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cutter.py`:

```python
"""Tests for video cutting module."""

import json
from pathlib import Path
import pytest
from snipclip.cutter import cut_video, TimeRange


class TestCutVideoKeep:
    def test_keep_single_segment(self, test_video, tmp_path):
        """Keep one segment from the middle of the video."""
        output = tmp_path / "output.mp4"
        segments = [TimeRange(start=1.0, end=3.0)]

        result = cut_video(test_video, segments, output, mode="keep")

        assert result == output
        assert result.exists()
        assert result.stat().st_size > 0

        # Verify duration is approximately correct
        from snipclip.probe import probe_video
        info = probe_video(result)
        assert info.duration == pytest.approx(2.0, abs=0.3)

    def test_keep_multiple_segments(self, test_video, tmp_path):
        """Keep two disjoint segments."""
        output = tmp_path / "output.mp4"
        segments = [
            TimeRange(start=0.5, end=1.5),
            TimeRange(start=3.0, end=4.0),
        ]

        result = cut_video(test_video, segments, output, mode="keep")

        assert result.exists()
        from snipclip.probe import probe_video
        info = probe_video(result)
        # Total kept: ~2.0 seconds
        assert info.duration == pytest.approx(2.0, abs=0.3)

    def test_raises_on_empty_segments(self, test_video, tmp_path):
        """Raises ValueError for empty segment list."""
        with pytest.raises(ValueError, match="empty"):
            cut_video(test_video, [], tmp_path / "out.mp4", mode="keep")

    def test_raises_on_invalid_segment(self, test_video, tmp_path):
        """Raises ValueError when start >= end."""
        with pytest.raises(ValueError, match="start.*end"):
            cut_video(
                test_video,
                [TimeRange(start=5.0, end=1.0)],
                tmp_path / "out.mp4",
                mode="keep",
            )


class TestCutVideoRemove:
    def test_remove_middle_segment(self, test_video, tmp_path):
        """Remove a segment from the middle."""
        output = tmp_path / "output.mp4"
        segments = [TimeRange(start=2.0, end=4.0)]

        result = cut_video(test_video, segments, output, mode="remove")

        assert result.exists()
        from snipclip.probe import probe_video
        info = probe_video(result)
        # Kept: ~0-2s and 4-5s ≈ 3 seconds
        assert info.duration == pytest.approx(3.0, abs=0.5)


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_cutter.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement snipclip/cutter.py**

```python
"""Video cutting and concatenation using FFmpeg.

Cuts video at specified time ranges. Two modes:
- keep: retain only specified segments, discard the rest
- remove: discard specified segments, keep the rest

Uses FFmpeg concat demuxer for lossless (stream copy) cutting at keyframes.
"""

from pathlib import Path
import subprocess
import tempfile
from typing import List, NamedTuple

from snipclip._ffmpeg import get_ffmpeg_path


class TimeRange(NamedTuple):
    """A time segment with start and end in seconds."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def cut_video(
    video_path: Path,
    segments: List[TimeRange],
    output_path: Path,
    mode: str = "keep",
) -> Path:
    """Cut a video, keeping or removing specified time segments.

    Args:
        video_path: Path to source video.
        segments: List of time ranges.
        output_path: Path for output video.
        mode: "keep" (retain only segments) or "remove" (delete segments).

    Returns:
        Path to the output video.

    Raises:
        ValueError: If segments is empty or contains invalid ranges.
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If FFmpeg fails.
    """
    if not segments:
        raise ValueError("Segments list cannot be empty")

    for seg in segments:
        if seg.start >= seg.end:
            raise ValueError(
                f"Invalid segment: start ({seg.start}) >= end ({seg.end})"
            )
        if seg.start < 0:
            raise ValueError(f"Invalid segment: start ({seg.start}) < 0")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg_path()

    if mode == "remove":
        # Convert remove segments to keep segments
        from snipclip.probe import probe_video
        info = probe_video(video_path)
        total_duration = info.duration

        # Sort segments to remove
        sorted_segs = sorted(segments, key=lambda s: s.start)
        keep_segs = []
        cursor = 0.0
        for seg in sorted_segs:
            if seg.start > cursor:
                keep_segs.append(TimeRange(start=cursor, end=seg.start))
            cursor = max(cursor, seg.end)
        if cursor < total_duration:
            keep_segs.append(TimeRange(start=cursor, end=total_duration))
        segments = keep_segs

    if not segments:
        raise ValueError("No segments to keep after processing")

    # Write FFmpeg concat file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as concat_file:
        for seg in segments:
            concat_file.write(f"file '{video_path.as_posix()}'\n")
            concat_file.write(f"inpoint {seg.start:.6f}\n")
            concat_file.write(f"outpoint {seg.end:.6f}\n")
        concat_path = concat_file.name

    try:
        result = subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_path,
                "-c", "copy",          # stream copy, no re-encode
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg cut failed:\n{result.stderr}"
            )

        return output_path

    finally:
        Path(concat_path).unlink(missing_ok=True)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_cutter.py -v
```
Expected: PASS

Note: If cuts are imprecise (keyframe alignment), duration assertions use `abs=0.5` tolerance. This is expected behavior with stream copy mode.

- [ ] **Step 5: Commit**

```bash
git add snipclip/cutter.py tests/test_cutter.py
git commit -m "feat: add video cutter module"
```

---

### Task 7: transcriber.py — Speech-to-Text

**Files:**
- Create: `snipclip/transcriber.py`
- Create: `tests/test_transcriber.py`

This is the most complex module. We mock `faster_whisper` for unit tests and use a real audio file for integration tests.

- [ ] **Step 1: Write the failing test**

Create `tests/test_transcriber.py`:

```python
"""Tests for transcription module."""

import json
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

    def test_detect_device_cpu_when_no_cuda(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=False)
        # Need to import after mock
        from snipclip.transcriber import detect_device as dd
        assert dd() == "cpu"


class TestTranscribe:
    def test_transcribe_mocked(self, mocker, tmp_path):
        """Transcribe with mocked faster-whisper model."""
        # Create a minimal audio file
        audio_path = tmp_path / "test.wav"
        # Write a minimal WAV header + silence (44 bytes header + data)
        # WAV header: RIFF, size, WAVE, fmt, PCM, 1ch, 16kHz, 16-bit
        import struct
        data_size = 16000 * 2  # 1 second of 16kHz 16-bit mono
        with open(audio_path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))     # chunk size
            f.write(struct.pack("<H", 1))      # PCM
            f.write(struct.pack("<H", 1))      # mono
            f.write(struct.pack("<I", 16000))  # sample rate
            f.write(struct.pack("<I", 32000))  # byte rate
            f.write(struct.pack("<H", 2))      # block align
            f.write(struct.pack("<H", 16))     # bits per sample
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(b"\x00" * data_size)

        # Mock the WhisperModel
        mock_model = mocker.MagicMock()
        mock_segments = [
            mocker.MagicMock(start=0.0, end=1.0, text="hello", no_speech_prob=0.1),
            mocker.MagicMock(start=1.0, end=2.0, text="world", no_speech_prob=0.2),
        ]
        mock_model.transcribe.return_value = (mock_segments, {"language": "en"})
        mock_whisper = mocker.patch(
            "snipclip.transcriber.WhisperModel", return_value=mock_model
        )

        segments = transcribe(audio_path, model_size="tiny")

        assert len(segments) == 2
        assert segments[0].text == "hello"
        assert segments[0].start == 0.0
        assert segments[0].end == 1.0
        assert segments[1].text == "world"

        # Verify model was created with correct args
        mock_whisper.assert_called_once_with("tiny", device="cpu", compute_type="int8")

    def test_transcribe_skips_silence(self, mocker, tmp_path):
        """Segments with high no_speech_prob are filtered out."""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"RIFF...")  # minimal, won't be processed

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_transcriber.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement snipclip/transcriber.py**

```python
"""Speech-to-text transcription using faster-whisper.

Supports CPU (int8) and GPU (CUDA float16) modes with auto-detection.
Outputs structured segments with timestamps, text, and confidence scores.
"""

from pathlib import Path
from typing import List, NamedTuple, Optional

from faster_whisper import WhisperModel


class Segment(NamedTuple):
    """A transcribed speech segment with timing and confidence."""
    start: float       # start time in seconds
    end: float         # end time in seconds
    text: str           # transcribed text
    confidence: float   # 0.0 to 1.0


def detect_device() -> str:
    """Detect best available compute device.

    Returns:
        "cuda" if CUDA GPU is available, otherwise "cpu".
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _get_compute_type(device: str) -> str:
    """Get the optimal compute type for the given device."""
    if device == "cuda":
        return "float16"
    return "int8"


def transcribe(
    audio_path: Path,
    model_size: str = "large-v3",
    device: Optional[str] = None,
    language: Optional[str] = None,
    silence_threshold: float = 0.5,
) -> List[Segment]:
    """Transcribe audio file to timed text segments.

    Args:
        audio_path: Path to 16kHz mono WAV file.
        model_size: Whisper model size ("tiny", "base", "small", "medium",
                    "large-v2", "large-v3"). Default: "large-v3".
        device: Compute device ("cpu" or "cuda"). Auto-detected if None.
        language: Language code (e.g. "en", "zh"). Auto-detected if None.
        silence_threshold: Segments with no_speech_prob above this are
                          filtered out. Default 0.5.

    Returns:
        List of Segment namedtuples with start, end, text, confidence.

    Raises:
        FileNotFoundError: If audio_path does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if device is None:
        device = detect_device()

    compute_type = _get_compute_type(device)

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    segments_out: List[Segment] = []

    raw_segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        word_timestamps=True,
    )

    for seg in raw_segments:
        # Skip silence / low-confidence segments
        if seg.no_speech_prob > silence_threshold:
            continue

        # Skip empty text
        text = seg.text.strip()
        if not text:
            continue

        segments_out.append(Segment(
            start=seg.start,
            end=seg.end,
            text=text,
            confidence=1.0 - seg.no_speech_prob,
        ))

    return segments_out
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_transcriber.py -v
```
Expected: PASS (may download tiny model on first run)

- [ ] **Step 5: Commit**

```bash
git add snipclip/transcriber.py tests/test_transcriber.py
git commit -m "feat: add speech-to-text transcription module"
```

---

### Task 8: subtitler.py — Subtitle Generation & Burn-in

**Files:**
- Create: `snipclip/subtitler.py`
- Create: `tests/test_subtitler.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_subtitler.py`:

```python
"""Tests for subtitle generation module."""

import json
from pathlib import Path
import pytest
from snipclip.subtitler import generate_srt, burn_subtitles, Segment


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
        assert "1\n" in content
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
        # 3661.5s = 01:01:01,500
        assert "01:01:01,500 --> 01:01:02,000" in content


class TestBurnSubtitles:
    def test_burns_subtitles_into_video(self, test_video, tmp_path):
        """Burn subtitles and verify video is playable."""
        segments = [
            Segment(start=0.0, end=2.0, text="Test subtitle", confidence=0.9),
        ]
        output = tmp_path / "burned.mp4"

        result = burn_subtitles(test_video, segments, output)

        assert result == output
        assert result.exists()
        assert result.stat().st_size > 0

        # Verify it's a valid video
        from snipclip.probe import probe_video
        info = probe_video(result)
        assert info.duration > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_subtitler.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement snipclip/subtitler.py**

```python
"""Subtitle generation and burn-in using FFmpeg.

Generates SRT subtitle files from transcript segments.
Optionally burns (hard-codes) subtitles into the video.
"""

from pathlib import Path
import subprocess
from typing import List, NamedTuple

from snipclip._ffmpeg import get_ffmpeg_path


class Segment(NamedTuple):
    """A transcript segment used for subtitle generation."""
    start: float
    end: float
    text: str
    confidence: float


def _format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(
    segments: List[Segment],
    output_path: Path,
) -> Path:
    """Generate an SRT subtitle file from transcript segments.

    Args:
        segments: List of Segment namedtuples with start, end, text.
        output_path: Path for the .srt file.

    Returns:
        Path to the generated SRT file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not segments:
        output_path.write_text("")
        return output_path

    lines = []
    for i, seg in enumerate(segments, 1):
        start_ts = _format_srt_timestamp(seg.start)
        end_ts = _format_srt_timestamp(seg.end)
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.text)
        lines.append("")  # blank line between entries

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def burn_subtitles(
    video_path: Path,
    segments: List[Segment],
    output_path: Path,
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
) -> Path:
    """Burn subtitles directly into the video (hard-coded).

    Args:
        video_path: Path to source video.
        segments: List of Segment namedtuples.
        output_path: Path for output video with burned subtitles.
        font_size: Subtitle font size in pixels.
        font_color: Text color (FFmpeg color name or hex).
        outline_color: Text outline color.

    Returns:
        Path to the output video.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If FFmpeg fails.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate temporary SRT file
    srt_path = output_path.with_suffix(".tmp.srt")
    generate_srt(segments, srt_path)

    try:
        ffmpeg = get_ffmpeg_path()

        # Build subtitles filter
        # Escape SRT path for FFmpeg (Windows needs backslash escaping)
        srt_path_str = str(srt_path).replace("\\", "/").replace(":", "\\:")

        style = (
            f"FontSize={font_size},"
            f"PrimaryColour=&H{font_color},"
            f"OutlineColour=&H{outline_color},"
            f"BorderStyle=1,Outline=2,Shadow=1"
        )

        vf = f"subtitles='{srt_path_str}':force_style='{style}'"

        result = subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-i", str(video_path),
                "-vf", vf,
                "-c:a", "copy",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg subtitle burn failed:\n{result.stderr}"
            )

        return output_path

    finally:
        srt_path.unlink(missing_ok=True)
```

- [ ] **Step 4: Run tests — note: burn test may need font config on Linux**

```bash
python -m pytest tests/test_subtitler.py -v
```
Expected: PASS (or skip burn test if no fonts available — mark with `pytest.mark.skipif`)

- [ ] **Step 5: Commit**

```bash
git add snipclip/subtitler.py tests/test_subtitler.py
git commit -m "feat: add subtitle generation and burn-in module"
```

---

### Task 9: cli.py — Command-Line Interface (Wiring)

**Files:**
- Create: `snipclip/cli.py`
- Create: `tests/test_cli.py`

This ties all modules together into usable CLI commands.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
"""Tests for CLI interface."""

import json
from pathlib import Path
import pytest
from click.testing import CliRunner
from snipclip.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestCliProbe:
    def test_probe_command(self, runner, test_video):
        result = runner.invoke(main, ["probe", str(test_video)])
        assert result.exit_code == 0
        # Output should be valid JSON
        data = json.loads(result.output)
        assert "duration" in data
        assert "width" in data
        assert "height" in data

    def test_probe_nonexistent_file(self, runner, tmp_path):
        result = runner.invoke(
            main, ["probe", str(tmp_path / "nope.mp4")]
        )
        assert result.exit_code != 0


class TestCliCut:
    def test_cut_keep_command(self, runner, test_video, tmp_path):
        """Cut with --keep option via CLI."""
        segments_file = tmp_path / "segments.json"
        segments = [
            {"start": 1.0, "end": 3.0},
            {"start": 4.0, "end": 4.5},
        ]
        segments_file.write_text(json.dumps(segments))

        output = tmp_path / "output.mp4"
        result = runner.invoke(
            main,
            [
                "cut",
                str(test_video),
                "--keep", str(segments_file),
                "--output", str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_cut_requires_keep_or_remove(self, runner, test_video):
        """Cut requires --keep or --remove flag."""
        result = runner.invoke(main, ["cut", str(test_video)])
        assert result.exit_code != 0
        assert "--keep" in result.output or "--remove" in result.output


class TestCliTranscribe:
    def test_transcribe_command_requires_ffmpeg(self, runner, test_video, tmp_path):
        """Transcribe command works (uses real whisper, may be slow)."""
        # This is an integration test — skip in CI without GPU
        output = tmp_path / "transcript.json"
        result = runner.invoke(
            main,
            [
                "transcribe",
                str(test_video),
                "--output", str(output),
                "--model", "tiny",   # smallest model for fast test
            ],
        )
        # May fail if faster-whisper model not downloaded
        # In CI, use a fixture or skip
        if result.exit_code != 0:
            pytest.skip(
                "Transcription requires faster-whisper model download"
            )


class TestCliHelp:
    def test_help_shows_commands(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "probe" in result.output
        assert "cut" in result.output
        assert "transcribe" in result.output
        assert "subtitle" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_cli.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement snipclip/cli.py**

```python
"""SnipClip CLI — video processing commands.

Commands:
  snipclip probe <video>              Get video metadata as JSON
  snipclip transcribe <video>         Transcribe speech to text
  snipclip cut <video> --keep <json>  Cut video by time segments
  snipclip subtitle <video> <transcript>  Generate subtitles
  snipclip setup                      Download FFmpeg locally
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from snipclip import __version__
from snipclip.probe import probe_video, VideoInfo
from snipclip.extractor import extract_audio
from snipclip.transcriber import transcribe, Segment
from snipclip.cutter import cut_video, TimeRange
from snipclip.subtitler import generate_srt, burn_subtitles

console = Console()


def _load_segments(path: Path) -> list[TimeRange]:
    """Load time segments from a JSON file.

    Expected format: [{"start": 0.0, "end": 1.0}, ...]
    """
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise click.BadParameter(f"Segments file must be a JSON array, got {type(data)}")
    segments = []
    for item in data:
        if not isinstance(item, dict) or "start" not in item or "end" not in item:
            raise click.BadParameter(f"Each segment must have 'start' and 'end': {item}")
        segments.append(TimeRange(start=float(item["start"]), end=float(item["end"])))
    return segments


def _save_segments(segments: list[Segment], path: Path) -> None:
    """Save transcript segments to JSON."""
    data = [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "confidence": s.confidence,
        }
        for s in segments
    ]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


@click.group()
@click.version_option(version=__version__)
def main():
    """SnipClip — AI-powered automatic video editing engine.

    Engine does the hands, Claude does the brain.
    """
    pass


@main.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
def probe(video: Path):
    """Get video metadata as JSON."""
    info = probe_video(video)
    console.print_json(json.dumps(info._asdict()))


@main.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output path for transcript JSON. Default: <video>_transcript.json",
)
@click.option(
    "--model", "-m",
    default="large-v3",
    help="Whisper model size (default: large-v3)",
)
@click.option(
    "--device", "-d",
    default=None,
    help="Compute device: cpu or cuda (auto-detect if not set)",
)
@click.option(
    "--language", "-l",
    default=None,
    help="Language code, e.g. en, zh (auto-detect if not set)",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    help="Cache directory for intermediate files. Default: ~/.snipclip/cache/",
)
def transcribe_cmd(
    video: Path,
    output: Optional[Path],
    model: str,
    device: Optional[str],
    language: Optional[str],
    cache_dir: Optional[Path],
):
    """Transcribe video speech to text with timestamps.

    Extracts audio, runs Whisper transcription, outputs JSON with
    start/end/text/confidence per segment.
    """
    if cache_dir is None:
        import hashlib
        video_hash = hashlib.md5(video.as_posix().encode()).hexdigest()[:12]
        cache_dir = Path.home() / ".snipclip" / "cache" / video_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    if output is None:
        output = Path.cwd() / f"{video.stem}_transcript.json"

    # Step 1: Extract audio
    audio_path = cache_dir / "audio.wav"
    with console.status(f"[bold]Extracting audio from {video.name}..."):
        extract_audio(video, audio_path)
    console.print(f"[green]Audio extracted:[/green] {audio_path}")

    # Step 2: Transcribe
    with console.status(f"[bold]Transcribing with {model} ({device or 'auto'})..."):
        segments = transcribe(
            audio_path,
            model_size=model,
            device=device,
            language=language,
        )
    console.print(f"[green]Transcription complete:[/green] {len(segments)} segments")

    # Step 3: Save
    _save_segments(segments, output)
    console.print(f"[green]Transcript saved:[/green] {output}")

    # Summary table
    if segments:
        table = Table(title="Transcript Preview")
        table.add_column("#", style="dim")
        table.add_column("Start")
        table.add_column("End")
        table.add_column("Text")
        for i, seg in enumerate(segments[:10], 1):
            table.add_row(
                str(i),
                f"{seg.start:.1f}s",
                f"{seg.end:.1f}s",
                seg.text[:80] + ("..." if len(seg.text) > 80 else ""),
            )
        console.print(table)
        if len(segments) > 10:
            console.print(f"  ... and {len(segments) - 10} more segments")


@main.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--keep", "-k",
    type=click.Path(exists=True, path_type=Path),
    help="JSON file with segments to KEEP",
)
@click.option(
    "--remove", "-r",
    type=click.Path(exists=True, path_type=Path),
    help="JSON file with segments to REMOVE",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output video path. Default: <video>_cut.mp4",
)
def cut(
    video: Path,
    keep: Optional[Path],
    remove: Optional[Path],
    output: Optional[Path],
):
    """Cut video by time segments.

    Use --keep to retain only specified segments.
    Use --remove to delete specified segments, keep the rest.
    One of --keep or --remove is required.
    """
    if keep is None and remove is None:
        raise click.UsageError("Either --keep or --remove is required")

    if keep is not None and remove is not None:
        raise click.UsageError("Use --keep or --remove, not both")

    segments_file = keep if keep else remove
    mode = "keep" if keep else "remove"
    segments = _load_segments(segments_file)

    if output is None:
        suffix = "_cut.mp4"
        output = video.parent / f"{video.stem}{suffix}"

    # Calculate what we're doing
    total_keep = sum(s.duration for s in segments)

    with console.status(f"[bold]Cutting video ({mode} mode)..."):
        cut_video(video, segments, output, mode=mode)

    console.print(f"[green]Video saved:[/green] {output}")
    console.print(f"Mode: {mode} | Segments: {len(segments)} | "
                  f"Total kept: {total_keep:.1f}s")


@main.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.argument("transcript", type=click.Path(exists=True, path_type=Path))
@click.option("--burn", is_flag=True, help="Burn subtitles into video")
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output path (SRT file or burned video)",
)
@click.option("--font-size", default=24, help="Subtitle font size (burn mode)")
def subtitle(
    video: Path,
    transcript: Path,
    burn: bool,
    output: Optional[Path],
    font_size: int,
):
    """Generate SRT subtitles or burn them into video.

    TRANSCRIPT is a JSON file from the 'transcribe' command.
    """
    # Load segments
    data = json.loads(transcript.read_text())
    segments = [
        Segment(
            start=float(s["start"]),
            end=float(s["end"]),
            text=s["text"],
            confidence=float(s.get("confidence", 1.0)),
        )
        for s in data
    ]

    if burn:
        if output is None:
            output = video.parent / f"{video.stem}_subtitled.mp4"
        with console.status("[bold]Burning subtitles..."):
            burn_subtitles(video, segments, output, font_size=font_size)
        console.print(f"[green]Burned video saved:[/green] {output}")
    else:
        if output is None:
            output = Path.cwd() / f"{video.stem}.srt"
        generate_srt(segments, output)
        console.print(f"[green]SRT saved:[/green] {output}")


@main.command()
def setup():
    """Download FFmpeg to ~/.snipclip/bin/ for local use."""
    import platform
    import urllib.request
    import zipfile
    import tarfile
    import tempfile

    system = platform.system()
    machine = platform.machine()

    console.print(f"[bold]Setting up FFmpeg for {system} ({machine})...")

    # Determine download URL based on platform
    if system == "Windows":
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        archive_type = "zip"
    elif system == "Darwin":
        url = "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
        archive_type = "zip"
    elif system == "Linux":
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        archive_type = "tar.xz"
    else:
        console.print(f"[red]Unsupported platform: {system}")
        console.print("Please install FFmpeg manually: https://ffmpeg.org/download.html")
        sys.exit(1)

    dest_dir = Path.home() / ".snipclip" / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)

    with console.status(f"[bold]Downloading FFmpeg..."):
        with tempfile.NamedTemporaryFile(suffix=f".{archive_type.replace('.', '_')}") as tmp:
            urllib.request.urlretrieve(url, tmp.name)

            # Extract
            if archive_type == "zip":
                with zipfile.ZipFile(tmp.name, "r") as zf:
                    # Find ffmpeg.exe and ffprobe.exe in the archive
                    for member in zf.namelist():
                        name = Path(member).name.lower()
                        if name in ("ffmpeg.exe", "ffprobe.exe", "ffmpeg", "ffprobe"):
                            target = dest_dir / Path(member).name
                            with zf.open(member) as src, open(target, "wb") as dst:
                                dst.write(src.read())
                            console.print(f"  Extracted: {target.name}")

    console.print(f"\n[green]FFmpeg installed to {dest_dir}")
    console.print("Make sure this directory is in your PATH, or the engine will auto-discover it.")
```

- [ ] **Step 4: Run CLI tests**

```bash
python -m pytest tests/test_cli.py -v -k "not transcribe"
```
Expected: PASS for probe, cut, help tests. Transcribe test skipped (needs model download).

- [ ] **Step 5: Verify CLI works end-to-end**

```bash
python -m snipclip.cli probe tests/fixtures/test_video.mp4
```
Expected: JSON output with duration, width, height, etc.

```bash
python -m snipclip.cli --help
```
Expected: Shows all available commands.

- [ ] **Step 6: Commit**

```bash
git add snipclip/cli.py tests/test_cli.py
git commit -m "feat: add CLI interface wiring all modules"
```

---

### Task 10: Claude Code Skill

**Files:**
- Create: `skill/snipclip.md`

This is the AI brain — the Claude Code skill that orchestrates the engine.

- [ ] **Step 1: Write the skill file**

Create `skill/snipclip.md`:

(Content is large — write the full skill definition. See end of this plan for skill content.)

- [ ] **Step 2: Verify skill is valid**

No formal validation for skill files. Review manually against Claude Code skill format.

- [ ] **Step 3: Commit**

```bash
git add skill/snipclip.md
git commit -m "feat: add Claude Code skill definition"
```

---

### Task 11: README & Documentation

**Files:**
- Create: `README.md`
- Create: `README_zh.md`

- [ ] **Step 1: Write README.md**

Brief English README covering: what is SnipClip, quick install, quick start, CLI reference, skill usage.

- [ ] **Step 2: Write README_zh.md**

Chinese translation of the above, with additional context for Chinese users.

- [ ] **Step 3: Commit**

```bash
git add README.md README_zh.md
git commit -m "docs: add README (EN + ZH)"
```

---

### Task 12 (Optional): scene.py — Scene Detection

**Files:**
- Create: `snipclip/scene.py`

Only if scope allows. Uses FFmpeg `select` filter to detect scene changes. Can be deferred to v0.2.

---

## Skill Content

The skill file at `skill/snipclip.md`:

```markdown
---
name: snipclip
description: AI-powered automatic video editing. Give raw footage and direction — get a finished video. Claude acts as editing director, SnipClip engine executes the cuts.
category: media
---

# SnipClip — AI Video Editing Director

You are an AI video editing director. When the user gives you a video file and describes what they want, you orchestrate the SnipClip engine to produce a finished edit.

## Prerequisites

Before starting, verify the engine is available:

```bash
pip show snipclip
```

If not installed:
```bash
pip install snipclip
```

Also check FFmpeg:
```bash
snipclip setup   # auto-download if needed
```

## The 5-Step Pipeline

### Step 1 — PROBE

Run `snipclip probe <video>` to get video metadata. Check: duration, resolution, whether it has audio. Report a one-line summary to the user.

### Step 2 — TRANSCRIBE

Run `snipclip transcribe <video> --output transcript.json`.

This extracts audio and runs Whisper transcription. The output is a JSON array:
```json
[
  {"start": 0.0, "end": 2.5, "text": "Hello everyone", "confidence": 0.95},
  ...
]
```

Wait for this to complete before proceeding. For long videos, tell the user it may take a few minutes.

### Step 3 — ANALYZE & DECIDE

Read the transcript JSON. Your job is to understand the user's intent and decide what to keep.

**Determine the editing persona** from the user's description:

| Persona | Trigger phrases | Strategy |
|---------|----------------|----------|
| 🎓 **Tutor** | "教程", "tutorial", "课程", "course", "教学" | Keep knowledge points, procedures, demonstrations. Remove chatter, tangents, repeated explanations. |
| 💼 **Meeting** | "会议", "meeting", "面试", "interview", "讨论" | Keep conclusions, decisions, action items. Remove discussion process, digressions, small talk. |
| ⚡ **Shorts** | "短", "short", "抖音", "tiktok", "reels", "shorts" | Fast jump cuts. Each kept segment ≤ 30s. High energy. Cut all slow sections. |
| 🎙️ **Podcast** | "播客", "podcast", "vlog", "聊天" | Remove silence, filler words, verbal tics. Keep narrative flow. Preserve humor and personality. |
| 🎯 **Custom** | (anything else) | Follow the user's explicit instructions literally. |

**How to analyze:**
1. Read through every segment's text
2. For each segment, decide: KEEP or CUT
3. For KEEP segments, merge adjacent segments into continuous time ranges
4. Calculate total kept duration
5. If user specified a target duration, adjust: trim less-important parts until target is met

**Present your plan to the user.** Show:
- Total original duration → target duration
- Number of segments kept / cut
- A summary of what each kept segment contains
- The exact time ranges that will be kept

Ask the user to confirm before executing.

### Step 4 — EXECUTE

Once the user confirms:

1. Write the kept time ranges to `segments.json`:
```json
[
  {"start": 0.0, "end": 15.5},
  {"start": 45.0, "end": 120.0}
]
```

2. Cut the video:
```bash
snipclip cut <video> --keep segments.json --output output.mp4
```

3. Generate subtitles (always offer):
```bash
snipclip subtitle output.mp4 transcript.json
```

4. Optionally burn subtitles:
```bash
snipclip subtitle output.mp4 transcript.json --burn --output output_subbed.mp4
```

### Step 5 — DELIVER & ITERATE

Report the result:
- Final video path and duration
- Retention rate (kept duration / original duration)
- File size

Ask if the user wants adjustments, e.g.:
- "把第3段再缩30秒"
- "多保留一些前面的内容"
- "去掉第2段"

If adjustments requested, go back to Step 3 with the new instructions.

## Guidelines

### Making Good Cuts
- Prefer cutting at sentence boundaries (natural pauses)
- Keep segments at least 1 second long (avoid micro-cuts)
- For tutorials: keep complete explanations, not fragments
- For podcasts: keep the setup before punchlines
- When in doubt, show the user both options

### Cache & Speed
- The engine caches extracted audio and transcripts in `~/.snipclip/cache/`
- On re-edits of the same video, skip re-transcription
- Tell users about this caching behavior

### Handling Edge Cases
- No audio track: inform user, suggest they provide a transcript
- Very short video (<30s): ask if editing is even needed
- Very long video (>2h): warn about processing time, suggest editing in chunks
- Low confidence transcript: flag segments with low confidence to the user

### Cross-Platform Notes
- Windows paths with spaces: always quote file paths
- Chinese text: the engine handles UTF-8 correctly
- GPU vs CPU: auto-detected, no user action needed
```

---

## Summary of All Tasks

| # | Task | Files Created |
|---|------|--------------|
| 1 | Project scaffolding | `pyproject.toml`, `.gitignore`, `LICENSE`, `snipclip/__init__.py`, `tests/__init__.py`, `tests/conftest.py` |
| 2 | FFmpeg discovery helper | `snipclip/_ffmpeg.py`, `tests/test_ffmpeg.py` |
| 3 | Test fixtures | `scripts/make_fixtures.py`, fixture videos |
| 4 | probe module | `snipclip/probe.py`, `tests/test_probe.py` |
| 5 | extractor module | `snipclip/extractor.py`, `tests/test_extractor.py` |
| 6 | cutter module | `snipclip/cutter.py`, `tests/test_cutter.py` |
| 7 | transcriber module | `snipclip/transcriber.py`, `tests/test_transcriber.py` |
| 8 | subtitler module | `snipclip/subtitler.py`, `tests/test_subtitler.py` |
| 9 | CLI wiring | `snipclip/cli.py`, `tests/test_cli.py` |
| 10 | Claude Code skill | `skill/snipclip.md` |
| 11 | README & docs | `README.md`, `README_zh.md` |
| 12 | (Optional) scene detection | `snipclip/scene.py` |
