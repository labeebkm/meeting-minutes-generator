# meeting-minutes-generator

Offline (CPU-only) Meeting Minutes generator for Manglish (Malayalam + English mixed speech).

## What it does

- Accepts a recorded meeting audio file (`.wav` / `.mp3`)
- Transcribes using **OpenAI Whisper (open-source, multilingual)**
- Translates Malayalam script to English **offline** using HuggingFace Transformers
- Generates structured **Minutes of Meeting (MoM)** (English) using **T5-small** summarization
- Provides a simple **Streamlit** UI for demo
- Saves outputs as plain text files in `output/`

## Architecture (end-to-end)

1. `transcribe.py`: **Audio → Text** using Whisper (**base**, task="transcribe", multilingual)
2. `utils.py`: **Text cleaning and normalization**
3. `translate.py`: Malayalam-to-English **TEXT translation** using MarianMT (`Helsinki-NLP/opus-mt-ml-en`) preserving English technical terms
4. `summarize.py`: English text summarization using **t5-small** (CPU-friendly), chunked to **max 512 tokens**
5. `output/`: Save outputs (`transcript.txt`, `meeting_minutes.txt`)
6. `app.py`: Streamlit UI (upload → button → display + autosave)

## Repo layout

```
meeting-minutes-generator/
├── app.py
├── transcribe.py
├── translate.py
├── summarize.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
├── sample_audio/
│   └── sample_meeting.wav
└── output/
    ├── transcript.txt
    └── meeting_minutes.txt
```

## Installation (Windows / CPU-only)

### 1) Create and activate a virtual environment

```bash
cd "meeting-minutes-generator"
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) (One-time) Download models while online

Run this once on an internet-connected machine to populate local caches:

```bash
python download_models.py
```

After this step, you can run the app **offline**.

### 3) Install FFmpeg (required by Whisper)

Whisper needs `ffmpeg` available in your PATH.

- Install via `winget`:

```bash
winget install Gyan.FFmpeg
```

Then open a new terminal and verify:

```bash
ffmpeg -version
```

## Run the demo UI

```bash
streamlit run app.py
```

## Sample workflow

1. Launch the UI (`streamlit run app.py`)
2. Upload a `.wav` or `.mp3` meeting recording
3. Click **Generate Minutes**
4. The app shows:
   - **Full English transcript**
   - **Bullet-point Minutes of Meeting**
5. Outputs are saved to:
   - `output/transcript.txt`
   - `output/meeting_minutes.txt`

## Notes on Manglish handling

- Whisper is used for multilingual transcription and handles code-switching well.
- For translation:
  - If Malayalam **Unicode script** is detected, it is translated offline using `Helsinki-NLP/opus-mt-ml-en`.
  - English/technical words (Latin script) are protected with placeholders and restored after translation.
  - If the transcript is already Latin script (Romanized Manglish), translation is skipped (the text is treated as already English-like).

## Offline-only requirement (model download)

This project runs **offline-only** at inference time. You must download the models once on a machine with internet, then reuse the local cache offline:

- Whisper: `base`
- Translation: `Helsinki-NLP/opus-mt-ml-en`
- Summarization: `t5-small`

When offline, the code loads HuggingFace models with `local_files_only=True` and will error clearly if the cache is missing.

## Limitations

- **Romanized Malayalam** (Manglish written in Latin script) is not reliably translatable without a specialized transliteration/MT model.
- CPU-only summarization/translation is slower; long meetings can take minutes.
- T5-small summaries are useful but not “perfect” MoM; accuracy depends on audio quality and meeting structure.

## Future improvements

- Add a diarization step (speaker separation) using offline tools (e.g., pyannote alternatives that are CPU-friendly).
- Replace the translation heuristic with a better Malayalam/English mixed-segment detector and per-segment translation.
- Add a “key decisions / action items / owners / due dates” extraction pass using lightweight rules + NER.
- Add caching for loaded models to speed up repeated runs (Streamlit `st.cache_resource`).




