from __future__ import annotations

from utils import clean_text, looks_like_malayalam, protect_latin_terms, restore_protected_terms, chunk_text_by_words


def _lazy_load_malayalam_mt():
    """
    Offline Malayalam->English MT using a lightweight MarianMT model.
    Model: Helsinki-NLP/opus-mt-ml-en
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import transformers. Install dependencies from requirements.txt.\n"
            f"Import error: {e}"
        )

    model_name = "Helsinki-NLP/opus-mt-ml-en"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            "Translation model not found in local HuggingFace cache (offline-only mode).\n"
            f"Missing: {model_name}\n"
            "Download once while online, then rerun offline.\n"
            f"Underlying error: {e}"
        ) from e
    model.eval()
    return tokenizer, model


def translate_to_english(text: str) -> str:
    """
    Translate Malayalam content to English while preserving English technical words.

    Strategy:
    - If no Malayalam script is detected, return cleaned input (already English / Romanized Manglish).
    - Otherwise run offline MT (MarianMT) on chunks and restore protected English terms.
    """
    text = clean_text(text or "")
    if not text:
        return ""

    if not looks_like_malayalam(text):
        # Many Manglish transcripts are already Latin script; MT won't help.
        return text

    protected = protect_latin_terms(text)
    protected_text = protected.protected_text

    tokenizer, model = _lazy_load_malayalam_mt()

    # Marian models handle moderate lengths; chunk conservatively for CPU stability.
    chunks = chunk_text_by_words(protected_text, max_words=200, overlap_words=20)
    if not chunks:
        return restore_protected_terms(protected_text, protected.placeholder_to_value)

    translated_parts = []
    for ch in chunks:
        inputs = tokenizer(ch, return_tensors="pt", truncation=True, max_length=512)
        with __import__("torch").no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        translated_parts.append(out_text)

    translated = clean_text(" ".join(translated_parts))
    translated = restore_protected_terms(translated, protected.placeholder_to_value)
    return translated




