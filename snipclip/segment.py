"""Uniform temporal video segmentation with per-segment tagging.

Divides video into fixed-duration segments (default 4s), extracts
one representative frame per segment, and tags each with:
- Visual: quality score, scene type, face count
- Audio: transcript text (if available)
- Metadata: timestamp, file source

This produces the structured "素材卡片" that Claude Code reads
to produce an editing plan.
"""

import subprocess
import json
from pathlib import Path
from typing import List, NamedTuple, Optional

from snipclip._ffmpeg import get_ffmpeg_path, get_ffprobe_path
from snipclip._cv import imread
from snipclip.quality import _assess_from_image
from snipclip.detector import detect
from snipclip.scene_classifier import classify_frame

import cv2


class Segment(NamedTuple):
    """A fixed-duration video segment with all tags."""
    segment_id: int
    start: float
    end: float
    duration: float
    source_file: str

    # Visual tags
    frame_path: Optional[Path]
    quality_overall: float
    is_blurry: bool
    faces: int
    people_count: int
    scene_tags: List[str]   # top 2 scene labels
    scene_source: str        # "clip" or "heuristic"

    # Audio tags
    transcript: str          # speech in this segment (empty if none)

    # Composite
    score: float             # 0-1 overall interestingness


def segment_video(
    video_path: Path,
    output_dir: Path,
    segment_duration: float = 4.0,
    transcript: Optional[List[dict]] = None,
    time_offset: float = 0.0,
) -> List[Segment]:
    """Divide a video into uniform segments and tag each one.

    Args:
        video_path: Source video file.
        output_dir: Directory for extracted frames.
        segment_duration: Duration of each segment in seconds (default 4s).
        transcript: Optional list of transcript segments (absolute timeline).
        time_offset: This file's start time in the transcript timeline.

    Returns:
        List of Segment namedtuples, one per time slice.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = video_path.stem

    # Get video duration and display dimensions
    ffprobe = get_ffprobe_path()
    result = subprocess.run(
        [str(ffprobe), "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(video_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    duration = 0.0
    if result.returncode == 0:
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))

    if duration <= 0:
        return []

    # Get display dimensions (after rotation) for frame orientation fix
    actual_dims = None
    if result.returncode == 0:
        streams = data.get("streams", [])
        for s in streams:
            if s.get("codec_type") == "video":
                actual_dims = (s.get("width", 0), s.get("height", 0))
                break

    # Calculate segment boundaries
    num_segments = max(1, int(duration / segment_duration))
    actual_seg_dur = duration / num_segments

    ffmpeg = get_ffmpeg_path()
    segments: List[Segment] = []

    for i in range(num_segments):
        start = i * actual_seg_dur
        end = min((i + 1) * actual_seg_dur, duration)
        midpoint = (start + end) / 2

        # Extract frame at segment midpoint
        frame_path = output_dir / f"{video_name}_seg{i:04d}.jpg"
        frame_ok = False
        if not frame_path.exists():
            r = subprocess.run(
                [str(ffmpeg), "-y", "-ss", str(midpoint),
                 "-i", str(video_path),
                 "-vframes", "1", "-q:v", "2",
                 str(frame_path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            )
            frame_ok = (r.returncode == 0 and frame_path.exists())
        else:
            frame_ok = True

        # Fix rotation: phone videos may have rotation metadata that
        # ffprobe reads (reporting display_w x display_h) but FFmpeg
        # single-frame extraction doesn't apply. Detect and fix.
        if frame_ok and actual_dims and frame_path.exists():
            try:
                raw = imread(frame_path)
                if raw is not None:
                    fh, fw = raw.shape[:2]
                    dw, dh = actual_dims
                    # If frame dims are swapped vs probe display dims, rotate
                    if (fw == dh and fh == dw) and fw != fh:
                        corrected = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
                        cv2.imwrite(str(frame_path), corrected)
            except Exception:
                pass

        # Quality
        quality_overall = 0.5
        is_blurry = False
        if frame_ok:
            try:
                img = imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    q = _assess_from_image(img)
                    quality_overall = q.overall
                    is_blurry = q.is_blurry
            except Exception:
                pass

        # Detection
        faces = 0
        people_count = 0
        if frame_ok:
            try:
                det = detect(frame_path)
                faces = det.faces
                people_count = det.people_count
            except Exception:
                pass

        # Scene tags
        scene_labels: List[str] = []
        scene_source = "heuristic"  # default
        if frame_ok:
            try:
                tags = classify_frame(frame_path)
                # CLIP scores are relative (0.2-0.3 range), take top 2 regardless
                # Heuristic scores are absolute (0-1 range), filter low confidence
                is_clip = any(t.source == "clip" for t in tags)
                if is_clip:
                    scene_labels = [t.label for t in tags[:2]]
                    scene_source = "clip"
                else:
                    scene_labels = [t.label for t in tags[:2] if t.confidence > 0.4]
                    scene_source = "heuristic"
            except Exception:
                pass

        # Transcript — map from absolute timeline using time_offset
        seg_transcript = ""
        if transcript:
            abs_start = start + time_offset
            abs_end = end + time_offset
            seg_transcript = " ".join(
                t["text"] for t in transcript
                if max(t["start"], abs_start) < min(t["end"], abs_end)
            )

        # Composite score: quality (40%) + faces (30%) + has_transcript (20%) + not_blurry (10%)
        face_score = min(faces / 5.0, 1.0)
        transcript_score = 0.3 if seg_transcript else 0.0
        blur_penalty = 0.0 if is_blurry else 0.1

        score = round(
            0.4 * quality_overall +
            0.3 * face_score +
            0.2 * transcript_score +
            blur_penalty,
            3,
        )

        segments.append(Segment(
            segment_id=i,
            start=round(start, 1),
            end=round(end, 1),
            duration=round(end - start, 1),
            source_file=video_path.name,
            frame_path=frame_path if frame_ok else None,
            quality_overall=quality_overall,
            is_blurry=is_blurry,
            faces=faces,
            people_count=people_count,
            scene_tags=scene_labels,
            scene_source=scene_source,
            transcript=seg_transcript,
            score=score,
        ))

    return segments


def generate_segment_report(
    video_paths: List[Path],
    output_path: Path,
    segment_duration: float = 4.0,
    transcript_path: Optional[Path] = None,
    progress_callback=None,
) -> dict:
    """Generate a full segment report for one or more video files.

    This is the primary entry point for the '素材分析报告'.

    Args:
        video_paths: List of video files.
        output_path: Where to save the JSON report.
        segment_duration: Seconds per segment (default 4s).
        transcript_path: Optional merged transcript JSON.
        progress_callback: Optional fn(current, total, filename).

    Returns:
        Report dict (also saved as JSON).
    """
    video_paths = sorted(set(video_paths))

    # Load transcript if provided
    transcript: Optional[List[dict]] = None
    if transcript_path and transcript_path.exists():
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

    frames_dir = output_path.parent / f"{output_path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "segment_duration": segment_duration,
        "total_files": len(video_paths),
        "files": [],
    }

    all_segments = []
    total_duration = 0.0
    cumulative_offset = 0.0  # track absolute time offset for transcript mapping

    for i, vp in enumerate(video_paths):
        if progress_callback:
            progress_callback(i, len(video_paths), vp.name)

        segs = segment_video(
            vp, frames_dir / vp.stem,
            segment_duration=segment_duration,
            transcript=transcript,
            time_offset=cumulative_offset,
        )
        all_segments.extend(segs)

        file_entry = {
            "file": vp.name,
            "path": str(vp),
            "segments": [
                {
                    "id": s.segment_id,
                    "start": s.start,
                    "end": s.end,
                    "duration": s.duration,
                    "quality": s.quality_overall,
                    "is_blurry": s.is_blurry,
                    "faces": s.faces,
                    "people_count": s.people_count,
                    "scene_tags": s.scene_tags,
                    "scene_source": s.scene_source,
                    "transcript": s.transcript,
                    "score": s.score,
                    "frame": str(s.frame_path) if s.frame_path else None,
                }
                for s in segs
            ],
        }
        report["files"].append(file_entry)

        # Get duration
        try:
            from snipclip.probe import probe_video
            info = probe_video(vp)
            file_entry["duration"] = info.duration
            total_duration += info.duration
        except Exception:
            file_entry["duration"] = sum(s.duration for s in segs)
            total_duration += file_entry["duration"]

        cumulative_offset += file_entry["duration"]

    if progress_callback:
        progress_callback(len(video_paths), len(video_paths), "done")

    report["total_duration"] = round(total_duration, 1)
    report["total_segments"] = len(all_segments)

    # Overall stats
    if all_segments:
        report["stats"] = {
            "avg_quality": round(sum(s.quality_overall for s in all_segments) / len(all_segments), 3),
            "blurry_ratio": round(sum(1 for s in all_segments if s.is_blurry) / len(all_segments), 3),
            "segments_with_faces": sum(1 for s in all_segments if s.faces > 0),
            "segments_with_speech": sum(1 for s in all_segments if s.transcript),
            "top_scenes": _top_scenes(all_segments),
        }

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _top_scenes(segments: List[Segment], n: int = 5) -> List[dict]:
    """Find the most common scene tags across segments."""
    from collections import Counter
    counter = Counter()
    for s in segments:
        for tag in s.scene_tags:
            counter[tag] += 1
    return [{"label": k, "count": v} for k, v in counter.most_common(n)]
