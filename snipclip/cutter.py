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
