# SnipClip Design Spec

**Date**: 2026-06-06
**Status**: Draft
**Author**: Claude Code + User

---

## Overview

SnipClip is an open-source, automatic video editing framework with a two-tier architecture:

- **SnipClip Engine**: A Python package (`pip install snipclip`) that provides atomic video processing capabilities — audio extraction, speech transcription, cutting/concatenation, and subtitle generation. It makes NO intelligent decisions. CPU-first, GPU-optional, cross-platform.
- **SnipClip Skill**: A Claude Code skill that acts as the "AI editing director." It reads transcripts, understands user intent, makes editing decisions, and calls the engine to execute.

**Core philosophy**: Engine does the hands, Claude does the brain.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 SnipClip Skill                    │
│          (Claude Code as editing brain)           │
│                                                   │
│   • Understand user intent                        │
│   • Read transcript, analyze content              │
│   • Make editing decisions (keep/cut segments)    │
│   • Call engine to execute                        │
│   • Preview results, iterate                      │
└──────────────────────┬──────────────────────────┘
                       │ calls
                       ▼
┌─────────────────────────────────────────────────┐
│              SnipClip Engine                      │
│          (Python package, pip install)             │
│                                                   │
│   • Extract audio (FFmpeg)                        │
│   • Transcribe speech (Whisper, CPU/GPU adaptive) │
│   • Cut & concatenate video (FFmpeg)              │
│   • Generate & burn subtitles                     │
│   • Scene detection (optional)                    │
│   • Cross-platform (Windows / macOS / Linux)      │
└─────────────────────────────────────────────────┘
```

---

## Project Structure

```
snipclip/                          # Repository root
├── snipclip/                      # Python package
│   ├── __init__.py
│   ├── cli.py                     # CLI entry point (Click)
│   ├── probe.py                   # Video info probe (FFprobe)
│   ├── extractor.py               # Audio extraction (FFmpeg)
│   ├── transcriber.py             # Speech-to-text (faster-whisper)
│   ├── cutter.py                  # Cut & concatenate (FFmpeg)
│   ├── subtitler.py               # Subtitle generation & burn-in
│   └── scene.py                   # Scene detection (FFmpeg)
│
├── skill/                         # Claude Code skill
│   └── snipclip.md                # Skill definition
│
├── tests/                         # Tests
│   ├── test_probe.py
│   ├── test_extractor.py
│   ├── test_transcriber.py
│   ├── test_cutter.py
│   └── fixtures/                  # Short test video clips
│
├── pyproject.toml                 # Project config + dependencies
├── README.md                      # Bilingual (EN/ZH)
├── README_zh.md
├── LICENSE                        # MIT
└── .gitignore
```

---

## Engine Modules

### Dependencies

```
dependencies = [
    "faster-whisper",    # CPU-friendly, GPU-optional transcription
    "click",             # CLI framework
    "rich",              # Terminal output formatting
    "ffmpeg-python",     # Lightweight FFmpeg Python bindings
]
```

### Module Contracts

#### `probe.py` — Video Info Probe

- **Input**: `video_path: Path`
- **Output**: `VideoInfo { duration, width, height, fps, codec, audio_codec, sample_rate, bitrate }`
- **Dependency**: FFprobe (bundled with FFmpeg)
- **No side effects**

#### `extractor.py` — Audio Extraction

- **Input**: `video_path: Path`
- **Output**: `audio.wav` (16kHz, mono, PCM — Whisper standard input)
- **Dependency**: FFmpeg
- **No side effects** beyond writing one file

#### `transcriber.py` — Speech-to-Text

- **Input**: `audio_path: Path`
- **Output**: `List[Segment]` where `Segment = { start: float, end: float, text: str, confidence: float }`
- **Backend**: `faster-whisper` (CTranslate2, CPU-efficient)
- **GPU support**: Auto-detect CUDA/cuBLAS; fall back to CPU
- **Model**: `large-v3` by default; configurable via `--model` flag
- **Output format**: JSON array of segments

#### `cutter.py` — Cut & Concatenate

- **Input**: `video_path: Path` + `segments: List[TimeRange]` where `TimeRange = { start: float, end: float }`
- **Output**: `output.mp4`
- **Modes**:
  - `keep` — keep only specified segments, discard rest
  - `remove` — remove specified segments, keep rest
- **Strategy**: FFmpeg concat demuxer, cut at keyframes to avoid re-encoding when possible
- **MUST NOT** re-encode audio/video unless necessary (preserve quality)

#### `subtitler.py` — Subtitle Generation

- **Input**: `video_path: Path` + `segments: List[Segment]`
- **Output**: SRT file or burned-in video
- **Options**: font size, position, color, background
- **Dependency**: FFmpeg subtitles filter

#### `scene.py` — Scene Detection (Optional/MVP)

- **Input**: `video_path: Path`
- **Output**: `List[float]` — list of timestamps where scene changes occur
- **Dependency**: FFmpeg `select='gt(scene,0.4)'` filter
- **MVP note**: Can be deferred; not required for core workflow

### `cli.py` — Command-Line Interface

Primary commands:

```bash
snipclip probe video.mp4              # Get video info
snipclip transcribe video.mp4         # Extract audio + transcribe
snipclip cut video.mp4 \
  --keep segments.json                # Cut and concatenate
snipclip cut video.mp4 \
  --remove segments.json              # Remove specified segments
snipclip subtitle video.mp4 \
  transcript.json                     # Generate SRT subtitle
snipclip subtitle video.mp4 \
  transcript.json --burn              # Burn subtitles into video
snipclip setup                        # Auto-download FFmpeg if missing
```

---

## Skill Layer

### Skill File Location

`.claude/skills/snipclip.md` in the repository. Users symlink or copy this into their own `.claude/skills/` directory.

### Skill Workflow (5-step pipeline)

```
User: "trim this meeting video to 3 min, keep only conclusions and decisions"
                         │
                         ▼
Step 1 ── PROBE
  Execute: snipclip probe video.mp4
  Get: duration, resolution, codec, audio format
  Purpose: sanity check, estimate processing time
                         │
                         ▼
Step 2 ── TRANSCRIBE
  Execute: snipclip transcribe video.mp4
  Get: full transcript with per-segment timestamps
  Format: JSON [{start, end, text, confidence}, ...]
                         │
                         ▼
Step 3 ── ANALYZE & DECIDE  (Claude's brain work)
  • Read full transcript
  • Understand user intent ("conclusions + decisions, under 3 min")
  • Mark each segment: KEEP or CUT
  • Merge adjacent KEEP segments → time range list
  • Verify total duration meets target
  • Present edit plan to user, wait for confirmation
                         │
                         ▼
Step 4 ── EXECUTE
  Execute:
    snipclip cut video.mp4 --keep segments.json
    snipclip subtitle video.mp4 transcript.json
  Get: final video + subtitle file
                         │
                         ▼
Step 5 ── DELIVER & ITERATE
  • Report final duration, retention rate
  • Ask if user wants adjustments ("shorten segment 3 by 30s")
  • All intermediate artifacts preserved in cache directory
  • Edits are traceable and replayable
```

### Built-in Editing Personas

The skill includes preset strategies. Claude auto-matches based on user description:

| Persona | Use Case | Core Logic |
|---------|----------|------------|
| 🎓 **Tutor** | Tutorials, courses | Keep knowledge points + procedures; remove chatter and repetition |
| 💼 **Meeting** | Meetings, interviews | Keep conclusions, decisions, action items; remove discussion process |
| ⚡ **Shorts** | Short-form video | Fast-paced jump cuts; each segment ≤ 30s |
| 🎙️ **Podcast** | Podcast, Vlog | Remove silence, filler words; preserve narrative flow |
| 🎯 **Custom** | User-defined | Claude fully follows the user's prompt intent |

### Cache & Traceability

All intermediate outputs are stored in `~/.snipclip/cache/<video_hash>/`:
- `audio.wav` — extracted audio
- `transcript.json` — transcription result
- `segments.json` — Claude's editing decisions
- `output.mp4` — final rendered video

This enables: replay, iteration without re-transcribing, debugging, and user audit of editing decisions.

---

## FFmpeg Installation Strategy

Three tiers, tried in order:

1. **User installed** — use system FFmpeg if already in PATH
2. **Auto-download** — `snipclip setup` downloads a static FFmpeg binary to `~/.snipclip/bin/`
3. **Docker** — `Dockerfile` provided for containerized usage, zero host dependencies

---

## Cross-Platform & Hardware Support

### CPU Mode (always works)
- `faster-whisper` with CTranslate2 runs on CPU, reasonably fast
- FFmpeg uses CPU encoding (libx264)
- Target: any machine with Python 3.10+

### GPU Mode (auto-detected)
- NVIDIA CUDA: faster-whisper auto-uses cuBLAS; FFmpeg uses `h264_nvenc`
- Apple Silicon: faster-whisper supports CoreML/ANE; FFmpeg uses `h264_videotoolbox`
- AMD: FFmpeg uses `h264_amf`
- Detection is automatic, no user config needed

### Platform Support
- Windows: primary dev target, full support
- macOS: supported
- Linux: supported
- All platforms use the same codebase

---

## Non-Goals (explicitly out of scope)

- GUI / web interface (CLI only; Claude Code is the "interface")
- Real-time / streaming video processing
- Video effects, transitions, color grading
- Multi-track timeline editing
- Cloud service / SaaS
- Built-in LLM API calls (engine makes NO LLM calls)

---

## Success Criteria

1. A user can `pip install snipclip` on Windows/macOS/Linux, CPU-only
2. `snipclip transcribe video.mp4` produces accurate, timestamped transcript
3. `snipclip cut video.mp4 --keep segments.json` produces valid, playable output
4. The Claude Code skill can orchestrate a full "raw footage → finished edit" pipeline
5. All intermediate artifacts are cacheable and replayable
6. The project is cleanly structured for open-source contribution
