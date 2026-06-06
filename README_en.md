# SnipClip 🎬

> AI-powered automatic video editing — engine does the hands, Claude does the brain.

SnipClip is a two-tier automatic video editing framework:

- **SnipClip Engine**: A Python CLI package that provides atomic video processing — audio extraction, Whisper transcription, cutting/concatenation, and subtitle generation. Zero intelligence, pure execution.
- **SnipClip Skill**: A Claude Code skill that acts as the "AI editing director." It reads transcripts, understands your intent, and makes all editing decisions.

## Quick Start

### 1. Install

```bash
pip install snipclip
snipclip setup          # auto-download FFmpeg if needed
```

### 2. Use with Claude Code

Copy the skill to your Claude Code skills directory:

```bash
cp skill/snipclip.md ~/.claude/skills/snipclip.md
```

Then in Claude Code:

```
/snipclip
Trim this meeting video to 3 minutes, keep only conclusions and decisions.
```

Claude will:
1. Probe the video
2. Transcribe speech to text
3. Analyze content and decide what to keep
4. Execute cuts via the engine
5. Deliver the finished video

### 3. Use Standalone CLI

```bash
# Get video info
snipclip probe video.mp4

# Transcribe speech
snipclip transcribe video.mp4 -o transcript.json

# Cut by segments (keep mode)
snipclip cut video.mp4 --keep segments.json -o output.mp4

# Generate subtitles
snipclip subtitle video.mp4 transcript.json
```

## Editing Personas

The Claude Code skill includes preset editing strategies:

| Persona | Best for | Behavior |
|---------|----------|----------|
| 🎓 Tutor | Tutorials, courses | Keep knowledge points, remove chatter |
| 💼 Meeting | Meetings, interviews | Keep conclusions and decisions |
| ⚡ Shorts | TikTok, Reels | Fast jump cuts, ≤30s per segment |
| 🎙️ Podcast | Vlog, podcast | Remove silence, preserve narrative |
| 🎯 Custom | Anything | Follow your exact instructions |

## Architecture

```
Claude Code (AI Director)        SnipClip Engine (Executor)
┌──────────────────────┐        ┌──────────────────────┐
│ • Understand intent  │───────▶│ probe   → video info │
│ • Analyze transcript │        │ extract → audio/wav  │
│ • Decide keep/cut    │        │ transcribe → text    │
│ • Present plan       │        │ cut     → output.mp4 │
│ • Iterate on feedback│        │ subtitle → .srt      │
└──────────────────────┘        └──────────────────────┘
```

## Requirements

- Python 3.10+
- FFmpeg (auto-download with `snipclip setup`)
- Claude Code (for AI-powered editing)
- CPU: works on any machine
- GPU: auto-detected (CUDA) for faster transcription

## Development

```bash
git clone https://github.com/buyicoder/snipclip.git
cd snipclip
pip install -e ".[dev]"
python scripts/make_fixtures.py   # generate test videos
pytest                              # run tests
```

## License

MIT
