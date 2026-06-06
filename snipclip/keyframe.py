"""Keyframe extraction using FFmpeg scene detection.

Extracts representative frames at scene change boundaries.
"""

import subprocess
import json
from pathlib import Path
from typing import List, NamedTuple, Optional

from snipclip._ffmpeg import get_ffmpeg_path, get_ffprobe_path
from snipclip._cv import imwrite as cv_imwrite


class KeyFrame(NamedTuple):
    """A keyframe extracted at a scene change point."""
    time: float           # seconds from video start
    image_path: Path      # path to extracted frame image
    scene_score: float    # FFmpeg scene change score (0-1)


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    threshold: float = 0.3,
    max_frames: int = 50,
) -> List[KeyFrame]:
    """Extract keyframes at scene change points.

    Args:
        video_path: Path to source video.
        output_dir: Directory to save extracted frames.
        threshold: Scene change sensitivity (0.0-1.0). Lower = more frames.
        max_frames: Maximum number of keyframes to extract.

    Returns:
        List of KeyFrame namedtuples.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem

    # Step 1: Detect scene changes using ffprobe
    ffprobe = get_ffprobe_path()
    result = subprocess.run(
        [
            str(ffprobe),
            "-v", "quiet",
            "-print_format", "json",
            "-show_frames",
            "-show_entries", "frame=pkt_pts_time,scene_score",
            "-f", "lavfi",
            f"movie={video_path.as_posix()},select='gt(scene\\,{threshold})'",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    change_times: List[float] = []
    if result.returncode == 0 and result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            for frame in data.get("frames", []):
                t = float(frame.get("pkt_pts_time", 0))
                score = float(frame.get("tags", {}).get("lavfi.scene_score", 0))
                change_times.append(t)
                if len(change_times) >= max_frames:
                    break
        except (json.JSONDecodeError, KeyError):
            pass

    # Step 2: If no scene changes detected, fall back to uniform sampling
    if not change_times:
        # Get video duration
        probe_result = subprocess.run(
            [
                str(ffprobe), "-v", "quiet",
                "-print_format", "json", "-show_format",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if probe_result.returncode == 0:
            dur_data = json.loads(probe_result.stdout)
            duration = float(dur_data.get("format", {}).get("duration", 0))
            if duration > 0:
                interval = duration / min(max_frames, 20)
                change_times = [interval * i + interval / 2 for i in range(min(max_frames, 20))]

    # Step 3: Extract frames at detected times
    ffmpeg = get_ffmpeg_path()
    keyframes: List[KeyFrame] = []

    for i, t in enumerate(change_times):
        frame_path = output_dir / f"{video_name}_kf{i:04d}.jpg"
        result = subprocess.run(
            [
                str(ffmpeg), "-y",
                "-ss", str(t),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(frame_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode == 0 and frame_path.exists():
            keyframes.append(KeyFrame(
                time=t,
                image_path=frame_path,
                scene_score=0.5,  # default for uniform samples
            ))

    return keyframes


def extract_preview_grid(
    video_paths: List[Path],
    output_path: Path,
    cols: int = 4,
    frame_interval: float = 2.0,
) -> Path:
    """Generate a grid preview image from multiple videos.

    Extracts one frame every `frame_interval` seconds and arranges in a grid.

    Args:
        video_paths: List of video files.
        output_path: Output image path.
        cols: Number of columns in the grid.
        frame_interval: Seconds between frames.

    Returns:
        Path to the generated preview image.
    """
    import cv2
    import numpy as np

    frames: List[np.ndarray] = []

    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        step = int(fps * frame_interval)
        frame_idx = step // 2  # start at middle of first interval

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            # Resize to thumbnail
            h, w = frame.shape[:2]
            thumb_w = 320
            thumb_h = int(h * thumb_w / w)
            thumb = cv2.resize(frame, (thumb_w, thumb_h))
            frames.append(thumb)
            frame_idx += step

        cap.release()

    if not frames:
        raise RuntimeError("No frames could be extracted")

    # Arrange in grid
    rows = (len(frames) + cols - 1) // cols
    thumb_h, thumb_w = frames[0].shape[:2]
    grid = np.ones((rows * thumb_h + (rows - 1) * 4, cols * thumb_w + (cols - 1) * 4, 3), dtype=np.uint8) * 40

    for i, frame in enumerate(frames):
        r, c = i // cols, i % cols
        y = r * (thumb_h + 4)
        x = c * (thumb_w + 4)
        grid[y:y + thumb_h, x:x + thumb_w] = frame

    cv_imwrite(output_path, grid)
    return output_path
