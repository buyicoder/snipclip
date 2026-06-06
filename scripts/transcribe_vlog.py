#!/usr/bin/env python3
"""Transcribe merged vlog audio to text."""
import os, sys, json, time
from pathlib import Path

# Fix HF on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from faster_whisper import WhisperModel

audio = Path('D:/占占不学编程/2025五一/merged_audio.wav')
output = Path('D:/占占不学编程/2025五一/transcript.json')
log = Path('D:/占占不学编程/2025五一/transcribe.log')

# Use tiny model - fast on CPU (~2-3 min for 47 min audio)
MODEL_PATH = 'tiny'

def log_msg(msg):
    print(msg, flush=True)
    with open(log, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

log_msg(f'[{time.strftime("%H:%M:%S")}] Using model: {MODEL_PATH}')
log_msg(f'[{time.strftime("%H:%M:%S")}] Starting transcription of 47min audio...')
t0 = time.time()

try:
    model = WhisperModel(MODEL_PATH, device='cpu', compute_type='int8')
    segments_out = []
    raw_segments, info = model.transcribe(str(audio), beam_size=5, word_timestamps=True)

    for seg in raw_segments:
        if seg.no_speech_prob > 0.5:
            continue
        text = seg.text.strip()
        if not text:
            continue
        segments_out.append({
            'start': seg.start,
            'end': seg.end,
            'text': text,
            'confidence': 1.0 - seg.no_speech_prob,
        })
    elapsed = time.time() - t0

    output.write_text(json.dumps(segments_out, indent=2, ensure_ascii=False), encoding='utf-8')

    total_text = ' '.join(s['text'] for s in segments_out)
    log_msg(f'[{time.strftime("%H:%M:%S")}] DONE in {elapsed/60:.1f} min!')
    log_msg(f'Segments: {len(segments_out)}, Characters: {len(total_text)}')
    log_msg(f'Saved: {output}')
except Exception as e:
    log_msg(f'ERROR: {e}')
    import traceback
    log_msg(traceback.format_exc())
    sys.exit(1)
