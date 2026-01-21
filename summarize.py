from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from utils import clean_text, normalize_bullets


def _lazy_load_summarizer(model_name: str = "t5-small"):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import transformers. Install dependencies from requirements.txt.\n"
            f"Import error: {e}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    model.eval()
    return tokenizer, model


def _chunk_text_by_t5_tokens(tokenizer, text: str, max_input_tokens: int = 512, overlap_tokens: int = 64) -> List[str]:
    """
    Chunk to a hard cap of `max_input_tokens` (including the T5 prefix tokens).
    This follows the requirement: "Chunk input to max 512 tokens".
    """
    text = clean_text(text)
    if not text:
        return []

    prefix = "summarize: "
    # Compute prefix token cost once.
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    budget = max(16, max_input_tokens - len(prefix_ids))

    # Tokenize full text once (no specials) then window it.
    ids: List[int] = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(ids):
        end = min(start + budget, len(ids))
        chunk_ids = ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(ids):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def _summarize_chunk_t5(tokenizer, model, chunk: str) -> str:
    # T5 expects a task prefix.
    inp = "summarize: " + chunk.strip()
    enc = tokenizer(inp, return_tensors="pt", truncation=True, max_length=512)
    with __import__("torch").no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=140,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def _to_bullets(text: str) -> str:
    # Heuristic: split by sentence-like boundaries and turn into bullets.
    text = clean_text(text)
    if not text:
        return ""

    # If already bullet-like, normalize.
    if any(ln.strip().startswith(("-", "•", "*")) for ln in text.splitlines()):
        return normalize_bullets(text)

    # Offline-only: do NOT download NLTK data. Use a robust regex splitter.
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)

    parts = [p.strip() for p in parts if p and p.strip()]
    parts = parts[:12]  # keep MoM concise
    return normalize_bullets("\n".join(parts))


def generate_summary(text: str) -> str:
    """
    Generate Minutes of Meeting (MoM) in English using abstractive summarization (T5-small).
    Handles long transcripts via chunking + second-pass summarization.
    """
    text = clean_text(text or "")
    if not text:
        return ""

    tokenizer, model = _lazy_load_summarizer("t5-small")

    # Chunk transcript to keep each chunk inside model constraints (512 tokens).
    chunks = _chunk_text_by_t5_tokens(tokenizer, text, max_input_tokens=512, overlap_tokens=64)
    if not chunks:
        return ""

    # First pass summaries per chunk.
    partials: List[str] = []
    for ch in chunks:
        partials.append(_summarize_chunk_t5(tokenizer, model, ch))

    merged = clean_text(" ".join(partials))

    # Second pass: summarize summaries into a compact MoM.
    final_summary = _summarize_chunk_t5(tokenizer, model, merged) if merged else ""
    bullets = _to_bullets(final_summary)
    return bullets or normalize_bullets(final_summary)


