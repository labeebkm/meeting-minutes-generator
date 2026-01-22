# Meeting Minutes Generator (Offline, CPU-only)

An **offline, CPU-only Meeting Minutes (MoM) generator** for multilingual meetings, with strong support for **Manglish (Malayalam + English mixed speech)** and other Indian language mixtures.

The system converts **any spoken language → English transcript → structured Minutes of Meeting**, without requiring internet access at runtime.

---

## What this project does

- Accepts a recorded meeting audio file (`.wav` / `.mp3`)
- Transcribes and **normalizes all languages into English** using **OpenAI Whisper (open-source)**
- Generates structured **Minutes of Meeting (MoM)** using **T5-small summarization**
- Runs **fully offline** after a one-time model download
- Uses a simple **Streamlit UI** for demo and evaluation
- Saves outputs as plain text files

---

## Key design decision (important)

Instead of transcribing speech into the original language and translating later, this project uses:

**Whisper’s `task="translate"` mode**  
to convert *any spoken language directly into English at the ASR stage*.

This significantly improves robustness for:
- Manglish (Malayalam + English)
- Tanglish (Tamil + English)
- Hinglish (Hindi + English)
- Mixed multilingual meetings

It also simplifies downstream NLP tasks like summarization.

---

## End-to-end architecture

Audio (.wav / .mp3)
        ↓
Whisper ASR (base, task="translate")
        ↓
English transcript
        ↓
Text cleaning & normalization
        ↓
T5-small summarization (CPU-only, chunked)
        ↓
Minutes of Meeting (MoM)
        ↓
Saved as text + displayed in Streamlit UI

---

## Repository layout

meeting-minutes-generator/
├── app.py                 # Streamlit UI
├── transcribe.py          # Whisper ASR (audio → English)
├── summarize.py           # MoM generation using T5-small
├── utils.py               # Text cleaning, chunking utilities
├── download_models.py     # One-time model download helper
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

cd "meeting-minutes-generator"
conda create -p ./momenv python=3.10 -y
conda activate ./momenv

Note:
This project was developed and tested using **Conda on Windows**.
Conda avoids common issues with `torch`, `whisper`, and native dependencies.

---

### 2) Install Python dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

---

### 3) Install FFmpeg (required by Whisper)

Whisper requires **FFmpeg** for audio decoding.

Recommended (Windows):

1. Download FFmpeg from:
   https://www.gyan.dev/ffmpeg/builds/
2. Download **ffmpeg-release-essentials.zip**
3. Extract and place it at:

C:\ffmpeg\bin\ffmpeg.exe

4. Register FFmpeg explicitly inside the Conda environment:

conda env config vars set FFMPEG_BINARY=C:\ffmpeg\bin\ffmpeg.exe
conda deactivate
conda activate ./momenv

This avoids PATH length issues on Windows and works reliably with Conda.

---

## One-time model download (internet required once)

Run this **once** on a machine with internet access:

python download_models.py

This downloads and caches locally:
- Whisper model: base
- Summarization model: t5-small

After this step, the application runs **fully offline**.

---

## Run the demo UI

python -m streamlit run app.py

A browser window will open automatically.

---

## Sample workflow

1. Launch the UI
2. Upload a `.wav` or `.mp3` meeting recording
3. Click **Generate Minutes**
4. The app displays:
   - Full English transcript
   - Bullet-point Minutes of Meeting
5. Outputs are saved to:
   - output/transcript.txt
   - output/meeting_minutes.txt

---

## Multilingual & Manglish handling

- Whisper automatically detects the spoken language
- Using `task="translate"`, **all speech is normalized into English**
- This handles:
  - Manglish (Malayalam + English)
  - Tamil / Tanglish
  - Hindi / Hinglish
  - Mixed multilingual conversations
- No separate translation model is required at runtime

This design greatly improves summarization accuracy.

---

## Offline-only guarantee

- All models are loaded from **local cache only**
- No API calls
- No cloud dependencies
- Clear error messages if models are missing

Once models are downloaded, the system runs **completely offline**.

---

## Limitations

- CPU-only inference is slower for long meetings
- Whisper accuracy depends on audio quality
- T5-small summaries are useful but not perfect MoM replacements
- No speaker diarization yet

---

## Future improvements

- Speaker diarization (offline-friendly)
- Action item / decision extraction
- Named entity recognition (owners, dates)
- Streamlit model caching for faster reruns
- Optional GPU acceleration

---

## Why this project is interview-ready

- Offline-first design
- No paid APIs
- Robust multilingual handling
- Clear architectural trade-offs
- Realistic constraints (CPU-only, Windows)
- Clean separation of ASR and NLP stages



