from __future__ import annotations

import os
import re
import shutil
import whisper
from typing import List, Dict


# --------------------------------------------------
# FFmpeg setup (required by Whisper)
# --------------------------------------------------
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    os.environ["FFMPEG_BINARY"] = ffmpeg_path


# --------------------------------------------------
# Regex for Malayalam Unicode detection
# --------------------------------------------------
MALAYALAM_UNICODE_RE = re.compile(r"[\u0D00-\u0D7F]")


# --------------------------------------------------
# Validate input path
# --------------------------------------------------
def _validate_audio_path(audio_path: str) -> None:
    if not audio_path or not isinstance(audio_path, str):
        raise ValueError("audio_path must be a non-empty string")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    if os.path.isdir(audio_path):
        raise IsADirectoryError(f"Expected a file, got directory: {audio_path}")


# --------------------------------------------------
# Time formatting helper
# --------------------------------------------------
def _format_ts(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


# --------------------------------------------------
# Malayalam-aware cleanup
# --------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans Whisper output while preserving Manglish and Malayalam.
    """
    text = text.strip()

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    # Remove excessive filler noise
    text = re.sub(r"\b(uh|um|erm|ah)\b", "", text, flags=re.IGNORECASE)

    # Do NOT lowercase Malayalam text
    if not MALAYALAM_UNICODE_RE.search(text):
        text = text.strip()

    return text


# --------------------------------------------------
# Lightweight speaker segmentation (CPU-safe)
# --------------------------------------------------
def assign_speakers(segments: List[Dict]) -> List[Dict]:
    """
    Simple heuristic-based speaker tagging.
    Speaker changes assumed when pause > 1.2s
    """

    speaker_id = 1
    last_end = 0.0

    for seg in segments:
        if seg["start"] - last_end > 1.2:
            speaker_id += 1

        seg["speaker"] = f"Speaker {speaker_id}"
        last_end = seg["end"]

    return segments


# --------------------------------------------------
# Main transcription function
# --------------------------------------------------
def transcribe_audio(audio_path: str) -> Dict:
    """
    Full-featured transcription for MOM generation.
    Handles long audio, Manglish, timestamps, and speakers.
    """

    print(">>> USING FULL MOM TRANSCRIBE PIPELINE <<<")
    _validate_audio_path(audio_path)

    audio_path = os.path.abspath(audio_path)
    print("ABSOLUTE PATH:", audio_path)

    # Load Whisper (CPU-optimized)
    model = whisper.load_model("base")

    try:
        result = model.transcribe(
            audio_path,
            task="transcribe",   # IMPORTANT for Manglish
            fp16=False,
            verbose=False
        )
    except Exception as e:
        raise RuntimeError(f"Whisper failed: {e}")

    segments = result.get("segments", [])
    if not segments:
        raise RuntimeError("No segments returned by Whisper")

    # Assign speakers
    segments = assign_speakers(segments)

    # Build structured transcript
    structured_segments = []
    full_text_parts = []

    for seg in segments:
        start = _format_ts(seg["start"])
        end = _format_ts(seg["end"])
        speaker = seg["speaker"]
        text = clean_text(seg["text"])

        if not text:
            continue

        structured_segments.append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "text": text
        })

        full_text_parts.append(f"{speaker}: {text}")

    full_text = "\n".join(full_text_parts)

    return full_text



# --------------------------------------------------
# MOM-ready formatter (OPTIONAL helper)
# --------------------------------------------------
def format_for_mom(transcription: Dict) -> str:
    """
    Converts structured transcript into MOM-friendly text
    """

    lines = []
    for seg in transcription["segments"]:
        line = (
            f"[{seg['start']} - {seg['end']}] "
            f"{seg['speaker']}: {seg['text']}"
        )
        lines.append(line)

    return "\n".join(lines)

