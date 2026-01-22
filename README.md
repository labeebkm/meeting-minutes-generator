# Meeting Minutes Generator (Offline, CPU-only)

An offline, CPU-only Meeting Minutes (MoM) generator for multilingual meetings, with strong support for Manglish (Malayalam + English mixed speech).

This project converts raw meeting audio into a clean transcript and structured Minutes of Meeting, without requiring internet access at runtime.

---

## What this project does

- Accepts recorded meeting audio (.wav / .mp3)
- Transcribes speech using OpenAI Whisper (open-source, offline)
- Reconstructs fragmented ASR output into sentence-level text
- Cleans and normalizes Manglish (Malayalam + English mixed speech)
- Generates structured Minutes of Meeting (MoM), including:
  - Key Discussion Points
  - Decisions
  - Action Items
- Runs fully offline after a one-time model download
- Provides a Streamlit UI for demo and evaluation
- Saves outputs as plain text files

---

## Key design decision (important)

Instead of relying purely on abstractive summarization, this project uses a hybrid ASR + rule-based MoM builder.

### Why?

ASR output is often fragmented, noisy, and not sentence-aligned. Direct summarization of raw ASR text produces poor MoM quality.

### Solution used here

- Reconstruct sentence-like units from Whisper output
- Remove noise and micro-fragments
- Explicitly extract discussion points, decisions, and action items
- Produce deterministic, explainable MoM output

This approach significantly improves MoM quality while remaining CPU-only and fully offline.

---

## End-to-end architecture

Audio (.wav / .mp3)
    ↓
Whisper ASR (base, multilingual, offline)
    ↓
Sentence reconstruction (ASR fragment merging)
    ↓
Manglish-aware text cleaning
    ↓
Structured MoM builder
(Discussion / Decisions / Actions)
    ↓
Minutes of Meeting
    ↓
Saved as text + displayed in Streamlit UI

---

## Repository layout

meeting-minutes-generator/
├── app.py                 # Streamlit UI and pipeline orchestration
├── transcribe.py          # Whisper ASR (long-audio safe)
├── mom_builder.py         # Structured MoM extraction logic
├── utils.py               # Cleaning, sentence reconstruction, helpers
├── download_models.py     # One-time offline model downloader
├── requirements.txt
├── README.md
├── .gitignore
├── sample_audio/
│   └── sample_meeting.wav
└── output/
    ├── transcript.txt
    └── meeting_minutes.txt

---

## Installation (Windows, CPU-only)

### 1) Create and activate a Conda environment (recommended)

cd meeting-minutes-generator
conda create -p ./momenv python=3.10 -y
conda activate ./momenv

Note:
This project was developed and tested using Conda on Windows.
Conda avoids common issues with torch, whisper, and native dependencies.

---

### 2) Install Python dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

---

### 3) Install FFmpeg (required by Whisper)

Whisper requires FFmpeg for audio decoding.

Recommended (Windows):

1. Download FFmpeg from:
   https://www.gyan.dev/ffmpeg/builds/
2. Download ffmpeg-release-essentials.zip
3. Extract and place it at:

   C:\ffmpeg\bin\ffmpeg.exe

4. Register FFmpeg inside the Conda environment:

conda env config vars set FFMPEG_BINARY=C:\ffmpeg\bin\ffmpeg.exe
conda deactivate
conda activate ./momenv

---

## One-time model download (internet required once)

Run this once on a machine with internet access:

python download_models.py

This downloads and caches locally:
- Whisper model: base

After this step, the application runs fully offline.

---

## Run the demo UI

python -m streamlit run app.py

A browser window will open automatically.

---

## Sample workflow

1. Launch the UI
2. Upload a .wav or .mp3 meeting recording
3. Click "Generate Minutes"
4. The app displays:
   - Cleaned transcript
   - Structured MoM:
     - Key Discussion Points
     - Decisions
     - Action Items
5. Outputs are saved to:
   - output/transcript.txt
   - output/meeting_minutes.txt

---

## Manglish handling

- Whisper detects spoken language automatically
- Malayalam + English mixed speech is preserved
- Non-relevant Unicode noise is removed
- No online translation is required
- Designed specifically for Indian multilingual speech patterns

---

## Offline-only guarantee

- All models are loaded from local cache
- No API calls
- No cloud services
- Deterministic CPU-only execution

Once models are downloaded, the system runs completely offline.

---

## Limitations

- CPU-only inference is slower for long recordings
- Whisper accuracy depends on audio quality
- No speaker diarization yet
- Rule-based MoM extraction may miss subtle decisions

---

## Future improvements

- Speaker diarization (offline-friendly)
- Owner and deadline extraction
- PDF / DOCX export
- Confidence scoring for MoM items
- Optional GPU acceleration

---


## Run command (PowerShell)

& "$env:CONDA_PREFIX\python.exe" -m streamlit run app.py
