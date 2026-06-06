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
