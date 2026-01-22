from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from transcribe import transcribe_audio
from utils import (
    clean_text,
    ensure_dir,
    write_text,
    set_deterministic,
    reconstruct_sentences,
)
from mom_builder import build_mom


# --------------------------------------------------
# Paths
# --------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"


# --------------------------------------------------
# File upload helper
# --------------------------------------------------
def _save_uploaded_file(uploaded_file) -> str:
    tmp_dir = APP_DIR / "tmp"
    ensure_dir(str(tmp_dir))

    # Stable filename (important for Streamlit reruns)
    suffix = Path(uploaded_file.name).suffix.lower()
    out_path = tmp_dir / f"upload{suffix}"

    uploaded_file.seek(0)
    data = uploaded_file.read()

    if not data:
        raise RuntimeError("Uploaded file is empty")

    with open(out_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    return str(out_path)


# --------------------------------------------------
# MoM serialization helper (for file output)
# --------------------------------------------------
def mom_to_text(mom: dict) -> str:
    lines = []

    if mom.get("discussion"):
        lines.append("Key Discussion Points:")
        lines.extend(f"- {d}" for d in mom["discussion"])
        lines.append("")

    if mom.get("decisions"):
        lines.append("Decisions:")
        lines.extend(f"- {d}" for d in mom["decisions"])
        lines.append("")

    if mom.get("actions"):
        lines.append("Action Items:")
        lines.extend(f"- {a}" for a in mom["actions"])
        lines.append("")

    return "\n".join(lines).strip()


# --------------------------------------------------
# Main Streamlit app
# --------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Meeting Minutes Generator", layout="wide")
    st.title("Meeting Minutes Generator (Offline)")
    st.caption(
        "Whisper (base, multilingual) → Sentence reconstruction → "
        "Manglish cleaning → Structured MoM (Discussion / Decisions / Actions)"
    )

    uploaded = st.file_uploader(
        "Upload meeting audio (.wav or .mp3)",
        type=["wav", "mp3"],
    )

    col1, col2 = st.columns(2)
    with col1:
        generate_btn = st.button(
            "Generate Minutes",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )
    with col2:
        st.write("")

    if generate_btn and uploaded is not None:
        # Best-effort deterministic execution
        set_deterministic(0)

        audio_path = _save_uploaded_file(uploaded)

        with st.status("Running pipeline...", expanded=True) as status:
            try:
                # --------------------------------------------------
                # 1) Audio → Text
                # --------------------------------------------------
                st.write('1) Audio → Text (Whisper "base", multilingual)…')
                raw_transcript = transcribe_audio(audio_path)

                # --------------------------------------------------
                # 2) Sentence reconstruction + cleaning
                # --------------------------------------------------
                st.write("2) Text cleaning and sentence reconstruction…")
                cleaned_transcript = clean_text(
                    reconstruct_sentences(raw_transcript)
                )

                # --------------------------------------------------
                # 3) Build structured MoM (NO summarizer)
                # --------------------------------------------------
                st.write("3) Building structured Minutes of Meeting…")
                mom = build_mom(cleaned_transcript)

                # --------------------------------------------------
                # 4) Save outputs
                # --------------------------------------------------
                ensure_dir(str(OUTPUT_DIR))
                write_text(
                    str(OUTPUT_DIR / "transcript.txt"),
                    cleaned_transcript + "\n",
                )

                mom_text = mom_to_text(mom)
                write_text(
                    str(OUTPUT_DIR / "meeting_minutes.txt"),
                    mom_text + "\n",
                )

                status.update(
                    label="Done",
                    state="complete",
                    expanded=False,
                )

            except Exception as e:
                status.update(
                    label="Failed",
                    state="error",
                    expanded=True,
                )
                st.error(str(e))
                return

        # --------------------------------------------------
        # UI Output
        # --------------------------------------------------
        st.subheader("Full Transcript (Cleaned)")
        st.text_area(
            "Transcript",
            value=cleaned_transcript,
            height=320,
        )

        st.subheader("Minutes of Meeting (MoM)")

        if mom.get("discussion"):
            st.markdown("### Key Discussion Points")
            for d in mom["discussion"]:
                st.markdown(f"- {d}")

        if mom.get("decisions"):
            st.markdown("### Decisions")
            for d in mom["decisions"]:
                st.markdown(f"- {d}")

        if mom.get("actions"):
            st.markdown("### Action Items")
            for a in mom["actions"]:
                st.markdown(f"- {a}")

        st.success(
            f"Saved:\n"
            f"- `{OUTPUT_DIR / 'transcript.txt'}`\n"
            f"- `{OUTPUT_DIR / 'meeting_minutes.txt'}`"
        )


if __name__ == "__main__":
    main()





