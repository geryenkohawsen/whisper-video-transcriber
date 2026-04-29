# Video Transcriber

Local video transcription on Apple Silicon. Uses MLX Whisper (Metal GPU) with
optional speaker diarization via pyannote.audio.

## Hardware target

MacBook Air M4, 36GB unified memory. Default model `large-v3` runs at ~1x realtime.

## Setup

### 1. ffmpeg

```bash
brew install ffmpeg
```

### 2. Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`mlx-whisper` is auto-selected on `arm64 darwin`. PyTorch fallback handles non-Apple machines.

### 3. Diarization (optional, needed for dialogue format)

a. Get HF token: <https://huggingface.co/settings/tokens>

b. Accept terms on all gated models:

- <https://huggingface.co/pyannote/speaker-diarization-3.1>
- <https://huggingface.co/pyannote/segmentation-3.0>
- <https://huggingface.co/pyannote/speaker-diarization-community-1>

c. Provide token via `.env` file (preferred — persists across restarts):

```bash
cp .env.example .env
# edit .env, paste token after HF_TOKEN=
```

Or export inline (one-shot):

```bash
export HF_TOKEN=hf_xxx...
```

`.env` is git-ignored. `python-dotenv` loads it on app start.

Without `HF_TOKEN`, app still runs — dialogue tab shows a hint instead.

### 4. Run

```bash
python app.py
```

Open <http://127.0.0.1:5000>.

Override default model:

```bash
WHISPER_MODEL=large-v3 python app.py
```

## Models

| Name       | Size    | Speed (M4)   | Use when                           |
| ---------- | ------- | ------------ | ---------------------------------- |
| `large-v3` | ~3.5 GB | ~1x realtime | accuracy first (default)           |
| `turbo`    | ~1.6 GB | ~3x realtime | speed matters, accuracy still high |
| `medium`   | ~1.5 GB | ~2x realtime | fallback / older devices           |

Models auto-download from HuggingFace on first use to `~/.cache/huggingface`.
Whisper PyTorch fallback caches to `~/.cache/whisper`.

## Output formats

- **Plain** — paragraph of transcript text.
- **Timestamped** — `[mm:ss → mm:ss] text` per segment.
- **Dialogue** — `Speaker 1: "..."` (requires diarization).

## Bulk mode

Top of UI has **Single | Bulk** switcher. Bulk mode:

1. Drop or select multiple videos.
2. Pick output format for ZIP (`dialogue.md` / `timestamped.md` / `plain.md`).
3. Click **Transcribe all** — files process sequentially (avoids OOM at large-v3).
4. Click **Download .zip** — bundles `<filename>.<format>.md` per file plus `summary.md`.

Diarization toggle, model, language apply to all files. If diarization fails for a file in `dialogue` mode, it falls back to timestamped output for that file.

## Architecture

```
app.py          Flask routes, request handling
transcriber.py  MLX/Torch backend abstraction with fallback chain
diarizer.py     pyannote.audio pipeline wrapper
formatters.py   plain / timestamped / dialogue rendering + speaker alignment
templates/index.html  upload UI with tabbed output
```

## Notes

- Long videos: app keeps temp file on disk, processes once, no chunking. M4 with
  36GB handles multi-hour audio without OOM at large-v3.
- Diarization runs on MPS (Apple GPU) automatically.
- Fallback chain: requested model → `large-v3` → `turbo` → `medium`. App never
  hard-fails on model load issues.
