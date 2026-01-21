from __future__ import annotations

"""
One-time model download helper.

Run this ONCE while online to populate your local cache, then the app can run offline.

Models (fixed by interview requirements):
- Whisper: base
- Translation: Helsinki-NLP/opus-mt-ml-en
- Summarization: t5-small
"""

import os


def _temporarily_enable_online_downloads() -> None:
    # Our app sets offline env vars by default for safety.
    # This script explicitly enables online mode for downloading.
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)


def download_all() -> None:
    _temporarily_enable_online_downloads()

    # Whisper (OpenAI whisper package downloads weights on first use)
    import whisper

    print('Downloading Whisper model: "base" ...')
    whisper.load_model("base")

    # HuggingFace models (Transformers cache)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    for model_name in ("Helsinki-NLP/opus-mt-ml-en", "t5-small"):
        print(f'Downloading HuggingFace model: "{model_name}" ...')
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print("Done. Models are cached locally.")


if __name__ == "__main__":
    download_all()


