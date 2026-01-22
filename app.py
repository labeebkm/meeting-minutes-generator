from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from summarize import generate_summary
from transcribe import transcribe_audio
from translate import translate_to_english
from utils import clean_text, ensure_dir, write_text, set_deterministic


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"


# def _save_uploaded_file(uploaded_file) -> str:
#     ensure_dir(str(APP_DIR / "tmp"))
#     suffix = Path(uploaded_file.name).suffix.lower()
#     ts = int(time.time())
#     out_path = APP_DIR / "tmp" / f"upload_{ts}{suffix}"
#     with open(out_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return str(out_path)

def _save_uploaded_file(uploaded_file) -> str:
    tmp_dir = APP_DIR / "tmp"
    ensure_dir(str(tmp_dir))

    # FORCE stable filename (Streamlit rerun safe)
    suffix = Path(uploaded_file.name).suffix.lower()
    out_path = tmp_dir / f"upload{suffix}"

    uploaded_file.seek(0)   # CRITICAL
    data = uploaded_file.read()

    if not data:
        raise RuntimeError("Uploaded file is empty")

    with open(out_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    return str(out_path)


def main() -> None:
    st.set_page_config(page_title="Meeting Minutes Generator", layout="wide")
    st.title("Meeting Minutes Generator (Offline)")
    st.caption(
        "Whisper (base, multilingual) → Text cleaning → Malayalam→English (MarianMT) → "
        "Summarization (t5-small) → Save to output/"
    )

    uploaded = st.file_uploader("Upload meeting audio (.wav or .mp3)", type=["wav", "mp3"])

    col1, col2 = st.columns(2)
    with col1:
        generate_btn = st.button("Generate Minutes", type="primary", use_container_width=True, disabled=uploaded is None)
    with col2:
        st.write("")

    if generate_btn and uploaded is not None:
        # Deterministic execution (best-effort on CPU).
        set_deterministic(0)

        audio_path = _save_uploaded_file(uploaded)

        with st.status("Running pipeline...", expanded=True) as status:
            try:
                st.write('1) Audio → Text (Whisper "base", multilingual)…')
                raw_transcript = transcribe_audio(audio_path)

                st.write("2) Text cleaning and normalization…")
                cleaned_transcript = clean_text(raw_transcript)

                st.write("3) Malayalam-to-English translation (offline MarianMT)…")
                #english_transcript = clean_text(translate_to_english(cleaned_transcript))
                english_transcript = cleaned_transcript

                st.write("4) English text summarization (t5-small)…")
                minutes = clean_text(generate_summary(english_transcript))

                ensure_dir(str(OUTPUT_DIR))
                write_text(str(OUTPUT_DIR / "transcript.txt"), english_transcript + "\n")
                write_text(str(OUTPUT_DIR / "meeting_minutes.txt"), minutes + "\n")

                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Failed", state="error", expanded=True)
                st.error(str(e))
                return

        st.subheader("Full English Transcript")
        st.text_area("Transcript", value=english_transcript, height=320)

        st.subheader("Minutes of Meeting (MoM)")
        # Render as bullets when possible.
        st.markdown(minutes if minutes.strip().startswith("-") else f"- {minutes}")
        st.text_area("MoM (raw text)", value=minutes, height=220)

        st.success(f"Saved: `{OUTPUT_DIR / 'transcript.txt'}` and `{OUTPUT_DIR / 'meeting_minutes.txt'}`")


if __name__ == "__main__":
    main()




