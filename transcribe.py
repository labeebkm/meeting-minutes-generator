from __future__ import annotations

import os
import whisper
import numpy as np


import shutil

ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    os.environ["FFMPEG_BINARY"] = ffmpeg_path


def _validate_audio_path(audio_path: str) -> None:
    if not audio_path or not isinstance(audio_path, str):
        raise ValueError("audio_path must be a non-empty string")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if os.path.isdir(audio_path):
        raise IsADirectoryError(f"Expected a file, got directory: {audio_path}")


def transcribe_audio(audio_path: str) -> str:
    print(">>> USING NEW TRANSCRIBE.PY <<<", audio_path)

    _validate_audio_path(audio_path)

    audio_path = os.path.abspath(audio_path)
    print("ABSOLUTE AUDIO PATH:", audio_path)

    model = whisper.load_model("base")

    try:
        audio = whisper.load_audio(audio_path)
        print("AUDIO LOADED, SHAPE:", audio.shape, "MAX AMP:", np.max(np.abs(audio)))
    except Exception as e:
        raise RuntimeError(f"whisper.load_audio FAILED: {e}")

    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(
        task="transcribe",
        language=None,
        fp16=False,
    )

    result = whisper.decode(model, mel, options)

    text = (result.text or "").strip()
    if not text:
        raise RuntimeError("Whisper returned empty transcript")

    return text

