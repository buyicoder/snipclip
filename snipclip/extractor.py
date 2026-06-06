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
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed:\n{result.stderr}"
        )

    return output_path
