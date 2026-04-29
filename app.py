import json
import os
import queue
import tempfile
import threading
import time

from dotenv import load_dotenv

# Load .env before importing modules that read env vars (e.g. HF_TOKEN in diarizer).
load_dotenv()

from flask import Flask, Response, jsonify, render_template, request

import diarizer
import formatters
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
            result = transcriber.transcribe(tmp_path, model_name=model_name, language=language)
            t_transcribe = time.time() - t0
            post("progress", stage="transcribed", message=f"Transcription done in {t_transcribe:.1f}s ({len(result['segments'])} segments).")

            segments = result["segments"]
            diar_segments = None
            diar_warning = None

            if enable_diarization:
                post("progress", stage="diarizing", message="Running speaker diarization (this can take a while)...")
                try:
                    t0 = time.time()
                    diar_segments = diarizer.diarize(tmp_path, num_speakers=num_speakers)
                    t_diar = time.time() - t0
                    n_speakers = len({s["speaker"] for s in diar_segments})
                    post("progress", stage="diarized", message=f"Diarization done in {t_diar:.1f}s ({n_speakers} speakers).")
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
