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
