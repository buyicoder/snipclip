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
    fmt = data.get("format", {})
    duration = float(video_stream.get("duration") or fmt.get("duration", 0))

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
        bitrate=int(fmt.get("bit_rate", 0)),
    )
