from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List


def set_deterministic(seed: int = 0) -> None:
    """
    Best-effort deterministic settings for CPU-only inference.

    Notes:
    - Some ops may still be nondeterministic depending on backend/build.
    - This project uses CPU only; we explicitly avoid CUDA paths.
    """
    import os
    import random

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # HuggingFace offline flags (we also pass local_files_only=True when loading)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    except Exception:
        pass

MALAYALAM_UNICODE_RE = re.compile(r"[\u0D00-\u0D7F]")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding=encoding) as f:
        f.write(text)


def clean_text(text: str) -> str:
    # Normalize whitespace and strip obvious transcription artifacts.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_malayalam(text: str) -> bool:
    return bool(MALAYALAM_UNICODE_RE.search(text))


def normalize_bullets(text: str) -> str:
    # Ensure each bullet starts with "- ".
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        ln = re.sub(r"^[•\-\*\u2022]+\s*", "", ln)
        out.append(f"- {ln}")
    return "\n".join(out).strip()


def chunk_text_by_words(text: str, max_words: int = 350, overlap_words: int = 40) -> List[str]:
    """
    Simple, robust chunking that doesn't require tokenizers.
    Works well enough for CPU-only workflows.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


@dataclass(frozen=True)
class ProtectedSpans:
    protected_text: str
    placeholder_to_value: dict


_LATIN_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_\-./]*\b")


def protect_latin_terms(text: str) -> ProtectedSpans:
    """
    Protect English/technical words (Latin script) from MT by replacing with placeholders,
    then restoring post-translation.
    """
    placeholder_to_value = {}
    counter = 0

    def repl(match: re.Match) -> str:
        nonlocal counter
        val = match.group(0)
        # Skip very short common words; keep acronyms/product names/paths.
        if len(val) <= 2:
            return val
        counter += 1
        key = f"__TERM_{counter}__"
        placeholder_to_value[key] = val
        return key

    protected = _LATIN_WORD_RE.sub(repl, text)
    return ProtectedSpans(protected_text=protected, placeholder_to_value=placeholder_to_value)


def restore_protected_terms(text: str, placeholder_to_value: dict) -> str:
    out = text
    # Restore longer keys first to avoid partial overlaps (very unlikely but safe).
    for k in sorted(placeholder_to_value.keys(), key=len, reverse=True):
        out = out.replace(k, placeholder_to_value[k])
    return out




