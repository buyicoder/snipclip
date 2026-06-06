"""SnipClip CLI — video processing commands.

Commands:
  snipclip probe <video>              Get video metadata as JSON
  snipclip transcribe <video>         Transcribe speech to text
  snipclip report <path>             Generate segmented素材分析报告 for Claude
  snipclip cut <video> --keep <json>  Cut video by time segments
  snipclip subtitle <video> <transcript>  Generate subtitles
  snipclip index <path>              Index video material with visual analysis
  snipclip preview <path>            Generate thumbnail preview grid
  snipclip setup                      Download FFmpeg locally
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table

from snipclip import __version__
from snipclip.probe import probe_video
from snipclip.extractor import extract_audio
from snipclip.transcriber import transcribe, Segment as TransSegment
from snipclip.cutter import cut_video, TimeRange
from snipclip.subtitler import generate_srt, burn_subtitles, Segment as SubSegment

console = Console()


def _load_segments(path: Path) -> list[TimeRange]:
    """Load time segments from a JSON file.

    Expected format: [{"start": 0.0, "end": 1.0}, ...]
    """
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise click.BadParameter(f"Segments file must be a JSON array")
    segments = []
    for item in data:
        if not isinstance(item, dict) or "start" not in item or "end" not in item:
            raise click.BadParameter(f"Each segment must have 'start' and 'end': {item}")
        segments.append(TimeRange(start=float(item["start"]), end=float(item["end"])))
    return segments


def _save_segments(segments: list[TransSegment], path: Path) -> None:
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
        SubSegment(
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
                    for member in zf.namelist():
                        name = Path(member).name.lower()
                        if name in ("ffmpeg.exe", "ffprobe.exe", "ffmpeg", "ffprobe"):
                            target = dest_dir / Path(member).name
                            with zf.open(member) as src, open(target, "wb") as dst:
                                dst.write(src.read())
                            console.print(f"  Extracted: {target.name}")

    console.print(f"\n[green]FFmpeg installed to {dest_dir}")
    console.print("Make sure this directory is in your PATH, or the engine will auto-discover it.")


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("index.json"),
    help="Output JSON path (default: index.json)",
)
@click.option(
    "--threshold", "-t",
    default=0.3,
    help="Scene change sensitivity 0.0-1.0 (default: 0.3)",
)
@click.option(
    "--max-frames", "-m",
    default=30,
    help="Max keyframes per video (default: 30)",
)
def index(path: Path, output: Path, threshold: float, max_frames: int):
    """Index video material with visual analysis.

    Extracts keyframes, scores quality, detects faces/objects,
    classifies scenes, and outputs a structured JSON index.

    PATH can be a video file or directory of videos.
    """
    from snipclip.indexer import index_material

    paths = [path]

    def progress(current, total, filename):
        console.print(f"  [{current}/{total}] {filename}")

    with console.status("[bold]Indexing video material..."):
        result = index_material(
            paths, output,
            scene_threshold=threshold,
            max_keyframes_per_file=max_frames,
            progress_callback=progress,
        )

    console.print(f"\n[green]Index saved:[/green] {output}")
    console.print(f"Files: {result['total_files']}")
    console.print(f"Total duration: {result['total_duration']:.0f}s = {result['total_duration']/60:.1f} min")
    total_kfs = sum(len(f["keyframes"]) for f in result["files"])
    console.print(f"Keyframes extracted: {total_kfs}")


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("preview.jpg"),
    help="Output image path (default: preview.jpg)",
)
@click.option(
    "--cols", "-c",
    default=4,
    help="Number of columns in the grid (default: 4)",
)
@click.option(
    "--interval", "-i",
    default=2.0,
    help="Seconds between extracted frames (default: 2.0)",
)
def preview(path: Path, output: Path, cols: int, interval: float):
    """Generate a thumbnail preview grid from video files.

    Extracts frames at regular intervals and arranges them in a grid.
    Useful for quickly scanning video content.

    PATH can be a video file or directory of videos.
    """
    from snipclip.indexer import generate_preview

    with console.status("[bold]Generating preview grid..."):
        result = generate_preview([path], output, cols=cols, interval=interval)

    size_kb = result.stat().st_size / 1024
    console.print(f"[green]Preview saved:[/green] {result} ({size_kb:.0f} KB)")


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("report.json"),
    help="Output JSON path (default: report.json)",
)
@click.option(
    "--segment-duration", "-s",
    default=4.0,
    help="Seconds per segment (default: 4.0)",
)
@click.option(
    "--transcript", "-t",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Transcript JSON from transcribe command",
)
def report(path: Path, output: Path, segment_duration: float, transcript: Optional[Path]):
    """Generate a segmented analysis report for Claude Code.

    Divides videos into uniform time segments, tags each with
    visual metadata (quality, faces, scene type) and transcript text,
    producing a structured JSON that Claude Code reads to create
    an editing plan.

    PATH can be a video file or directory of videos.
    """
    from snipclip.segment import generate_segment_report

    # Collect video files
    video_files: List[Path] = []
    if path.is_dir():
        for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"):
            video_files.extend(sorted(path.glob(ext)))
    else:
        video_files = [path]

    # Filter out generated files
    SKIP_PREFIXES = ("merged", "vlog", "output", "index_")
    video_files = [
        vf for vf in video_files
        if not any(vf.name.lower().startswith(pref) for pref in SKIP_PREFIXES)
    ]
    video_files = sorted(set(video_files))

    def progress(current, total, filename):
        console.print(f"  [{current}/{total}] {filename}")

    with console.status(f"[bold]Generating segment report (每{segment_duration}秒一段)..."):
        result = generate_segment_report(
            video_files, output,
            segment_duration=segment_duration,
            transcript_path=transcript,
            progress_callback=progress,
        )

    console.print(f"\n[green]Report saved:[/green] {output}")
    console.print(f"Files: {result['total_files']}")
    console.print(f"Total duration: {result['total_duration']:.0f}s = {result['total_duration']/60:.1f} min")
    console.print(f"Segments: {result['total_segments']} (每段 {segment_duration}s)")
    stats = result.get("stats", {})
    if stats:
        console.print(f"Avg quality: {stats.get('avg_quality', 0):.2f}")
        console.print(f"Segments with faces: {stats.get('segments_with_faces', 0)}")
        console.print(f"Segments with speech: {stats.get('segments_with_speech', 0)}")
        top = stats.get("top_scenes", [])
        if top:
            console.print(f"Top scenes: {', '.join(s['label'] for s in top[:3])}")
