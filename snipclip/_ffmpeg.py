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
