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
