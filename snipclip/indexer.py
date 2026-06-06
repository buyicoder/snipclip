"""Video material indexer — orchestrates visual analysis pipeline.

Walks through video files/directories, extracts keyframes,
analyzes quality, detects faces/objects, classifies scenes,
and outputs a structured index JSON.
"""

import json
import time
from pathlib import Path
from typing import List, Optional

from snipclip.keyframe import extract_keyframes, extract_preview_grid
from snipclip.quality import assess_quality
from snipclip.detector import detect
from snipclip.scene_classifier import classify_frame
from snipclip.groups import group_similar_frames, describe_scene
from snipclip.probe import probe_video


def index_material(
    paths: List[Path],
    output: Path,
    keyframes_dir: Optional[Path] = None,
    scene_threshold: float = 0.3,
    max_keyframes_per_file: int = 30,
    progress_callback=None,
) -> dict:
    """Index video material into a structured JSON.

    For each video file:
      1. Probe metadata
      2. Extract keyframes at scene changes
      3. For each keyframe: quality, detection, classification
      4. Group similar frames into scenes

    Args:
        paths: List of video file or directory paths.
        output: Path for output index JSON.
        keyframes_dir: Directory for extracted frames (default: <output>.frames/).
        scene_threshold: FFmpeg scene change threshold.
        max_keyframes_per_file: Max keyframes per video.
        progress_callback: Optional fn(current, total, filename).

    Returns:
        The index dict (also written to output).
    """
    video_files: List[Path] = []
    for p in paths:
        if p.is_dir():
            for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"):
                video_files.extend(sorted(p.glob(ext)))
        elif p.suffix.lower() in (".mp4", ".mov", ".avi"):
            video_files.append(p)

    # Filter out previously generated output files
    SKIP_PREFIXES = ("merged", "vlog", "output", "index_")
    video_files = [
        vf for vf in video_files
        if not any(vf.name.lower().startswith(pref) for pref in SKIP_PREFIXES)
    ]
    video_files = sorted(set(video_files))

    if keyframes_dir is None:
        keyframes_dir = output.parent / f"{output.stem}_frames"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_files": len(video_files),
        "total_duration": 0.0,
        "files": [],
    }

    all_keyframe_paths: List[Path] = []

    for i, vf in enumerate(video_files):
        if progress_callback:
            progress_callback(i, len(video_files), vf.name)

        # Probe
        try:
            info = probe_video(vf)
        except Exception:
            continue

        file_entry = {
            "file": vf.name,
            "path": str(vf),
            "duration": info.duration,
            "resolution": f"{info.width}x{info.height}",
            "fps": info.fps,
            "keyframes": [],
        }

        # Extract keyframes
        try:
            kf_dir = keyframes_dir / vf.stem
            kf_dir.mkdir(parents=True, exist_ok=True)
            kfs = extract_keyframes(
                vf, kf_dir, threshold=scene_threshold,
                max_frames=max_keyframes_per_file,
            )
        except Exception:
            kfs = []

        # Analyze each keyframe
        for kf in kfs:
            kf_entry = {
                "time": round(kf.time, 1),
                "image": str(kf.image_path),
            }

            # Quality
            try:
                q = assess_quality(kf.image_path)
                kf_entry["quality"] = {
                    "sharpness": round(q.sharpness, 1),
                    "brightness": round(q.brightness, 1),
                    "contrast": round(q.contrast, 1),
                    "is_blurry": q.is_blurry,
                    "overall": q.overall,
                }
            except Exception:
                kf_entry["quality"] = {"overall": 0.0, "is_blurry": True}

            # Detection
            try:
                det = detect(kf.image_path)
                kf_entry["detection"] = {
                    "faces": det.faces,
                    "people_count": det.people_count,
                    "objects": [
                        {"label": o.label, "confidence": o.confidence}
                        for o in det.objects[:5]
                    ],
                }
            except Exception:
                kf_entry["detection"] = {"faces": 0, "people_count": 0, "objects": []}

            # Scene classification
            try:
                tags = classify_frame(kf.image_path)
                kf_entry["scene_tags"] = [
                    {"label": t.label, "confidence": t.confidence, "source": t.source}
                    for t in tags
                ]
            except Exception:
                kf_entry["scene_tags"] = []

            file_entry["keyframes"].append(kf_entry)
            all_keyframe_paths.append(kf.image_path)

        index["files"].append(file_entry)
        index["total_duration"] += info.duration

    if progress_callback:
        progress_callback(len(video_files), len(video_files), "complete")

    # Scene grouping (across all frames)
    try:
        groups = group_similar_frames(all_keyframe_paths[:200])  # limit for perf
        index["scene_groups"] = [
            {
                "group_id": g.group_id,
                "count": len(g.keyframes),
                "representative": str(g.representative),
                "description": describe_scene(g.avg_color),
                "avg_color_rgb": g.avg_color,
            }
            for g in groups
        ]
    except Exception:
        index["scene_groups"] = []

    # Write output
    output.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


def generate_preview(
    paths: List[Path],
    output: Path,
    cols: int = 4,
    interval: float = 2.0,
) -> Path:
    """Generate a grid preview from video files.

    Args:
        paths: Video file or directory paths.
        output: Output image path.
        cols: Grid columns.
        interval: Seconds between extracted frames.

    Returns:
        Path to preview image.
    """
    video_files: List[Path] = []
    for p in paths:
        if p.is_dir():
            for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
                video_files.extend(sorted(p.glob(ext)))
        elif p.suffix.lower() in (".mp4", ".mov", ".avi"):
            video_files.append(p)

    video_files = sorted(set(video_files))
    return extract_preview_grid(video_files, output, cols=cols, frame_interval=interval)
