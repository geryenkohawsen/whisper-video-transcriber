import json
import os
import queue
import tempfile
import threading
import time
import warnings

from dotenv import load_dotenv

# Load .env before importing modules that read env vars (e.g. HF_TOKEN in diarizer).
load_dotenv()

# Silent / very short clips trigger benign numpy stats warnings inside whisper VAD
# and pyannote embeddings ("Mean of empty slice", "invalid value encountered in divide").
# Suppress to keep server log readable; clip output is empty either way.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

from flask import Flask, Response, jsonify, render_template, request

import diarizer
import formatters
import progress
import transcriber

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB

DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
AVAILABLE_MODELS = ["large-v3", "turbo", "medium"]


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_model=DEFAULT_MODEL,
        models=AVAILABLE_MODELS,
        backend=transcriber.detect_backend(),
        diarization_available=diarizer.is_available(),
    )


@app.route("/transcribe", methods=["POST"])
def transcribe_route():
    """
    Streams NDJSON events while processing. Each line is a JSON object:
      {"event": "progress", "stage": "...", "message": "..."}
      {"event": "result", ...full result...}
      {"event": "error", "error": "..."}
    """
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded = request.files["video"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Parse params synchronously before streaming.
    model_name = request.form.get("model", DEFAULT_MODEL)
    if model_name not in AVAILABLE_MODELS:
        model_name = DEFAULT_MODEL

    language = request.form.get("language") or None
    if language == "auto":
        language = None

    enable_diarization = request.form.get("diarize") == "true"
    num_speakers = request.form.get("num_speakers")
    try:
        num_speakers = int(num_speakers) if num_speakers else None
    except ValueError:
        num_speakers = None

    # Save upload to temp file (must complete before streaming starts).
    suffix = os.path.splitext(uploaded.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded.save(tmp.name)
        tmp_path = tmp.name

    # Worker thread posts events to queue; generator yields them as NDJSON.
    q: "queue.Queue[dict]" = queue.Queue()
    SENTINEL = {"__done__": True}

    def post(event_type: str, **fields):
        q.put({"event": event_type, **fields})

    def worker():
        try:
            post("progress", stage="upload", message="File saved. Preparing...")

            post("progress", stage="loading_model", message=f"Loading {model_name} ({transcriber.detect_backend()})...")
            t0 = time.time()

            # Emit at most one event per integer percent change to avoid flooding the queue.
            last_pct = [-1]
            def _on_tqdm(current: int, total: int):
                if not total:
                    return
                pct = int(current * 100 / total)
                if pct == last_pct[0]:
                    return
                last_pct[0] = pct
                post(
                    "progress",
                    stage="transcribing",
                    message=f"Transcribing: {pct}% ({current}/{total} frames)",
                    percent=pct,
                    current=current,
                    total=total,
                )

            with progress.capture_tqdm(_on_tqdm):
                result = transcriber.transcribe(tmp_path, model_name=model_name, language=language)
            t_transcribe = time.time() - t0
            post("progress", stage="transcribed", message=f"Transcription done in {t_transcribe:.1f}s ({len(result['segments'])} segments).", percent=100)

            segments = result["segments"]
            diar_segments = None
            diar_warning = None

            if enable_diarization:
                post("progress", stage="diarizing", message="Running speaker diarization (this can take a while)...", percent=0)
                try:
                    t0 = time.time()
                    diar_last_pct = [-1]
                    def _on_diar(current: int, total: int, step: str):
                        if not total:
                            return
                        pct = int(current * 100 / total)
                        if pct == diar_last_pct[0]:
                            return
                        diar_last_pct[0] = pct
                        post(
                            "progress",
                            stage="diarizing",
                            message=f"Diarization [{step}]: {pct}%",
                            percent=pct,
                            current=current,
                            total=total,
                            step=step,
                        )
                    diar_hook = progress.PyannoteHook(_on_diar)
                    diar_segments = diarizer.diarize(tmp_path, num_speakers=num_speakers, hook=diar_hook)
                    t_diar = time.time() - t0
                    n_speakers = len({s["speaker"] for s in diar_segments})
                    post("progress", stage="diarized", message=f"Diarization done in {t_diar:.1f}s ({n_speakers} speakers).", percent=100)
                except diarizer.DiarizationUnavailable as e:
                    diar_warning = str(e)
                    post("progress", stage="diar_skipped", message=f"Diarization unavailable: {e}")
                except Exception as e:
                    diar_warning = f"Diarization failed: {type(e).__name__}: {e}"
                    post("progress", stage="diar_failed", message=diar_warning)

            post("progress", stage="formatting", message="Formatting output...")
            # Build turns once for client-side rename / re-rendering.
            turns = (
                formatters.build_turns(segments, diar_segments)
                if diar_segments
                else []
            )
            response = {
                "language": result["language"],
                "model": result["model"],
                "backend": result["backend"],
                "plain": formatters.plain(segments),
                "timestamped": formatters.timestamped(segments),
                "dialogue": formatters.dialogue(segments, diar_segments),
                "plain_md": formatters.plain_md(segments),
                "timestamped_md": formatters.timestamped_md(segments),
                "dialogue_md": formatters.dialogue_md(segments, diar_segments),
                "segments": segments,
                "dialogue_turns": turns,
                "diarization_used": diar_segments is not None,
                "diarization_warning": diar_warning,
            }
            post("result", **response)

        except Exception as e:
            post("error", error=f"{type(e).__name__}: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            q.put(SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            event = q.get()
            if event is SENTINEL:
                return
            yield json.dumps(event) + "\n"

    return Response(stream(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    backend = transcriber.detect_backend()
    print(f"Backend: {backend} (mlx = Apple Silicon GPU; torch = CPU fallback)")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Diarization: {'available' if diarizer.is_available() else 'unavailable (no HF_TOKEN or pyannote missing)'}")
    print("Server: http://127.0.0.1:5000  (or http://localhost:5000)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
