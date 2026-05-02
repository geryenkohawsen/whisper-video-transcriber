import json
import os
import queue
import tempfile
import threading
import time
import uuid
import warnings

from dotenv import load_dotenv

# Load .env before importing modules that read env vars (e.g. HF_TOKEN in diarizer).
load_dotenv()

# Silent / very short clips trigger benign numpy stats warnings inside whisper VAD
# and pyannote embeddings ("Mean of empty slice", "invalid value encountered in divide").
# Suppress to keep server log readable; clip output is empty either way.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

from flask import Flask, Response, jsonify, render_template, request, url_for

import diarizer
import extractor
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
        active_page="transcribe",
    )


# ============================================================
# Audio extraction (separate page)
# ============================================================

# Token → (output_path, download_filename, mime_type). Files removed after first
# download or on next server restart (temp dir).
_EXTRACT_RESULTS: "dict[str, tuple[str, str, str]]" = {}
_EXTRACT_LOCK = threading.Lock()


@app.route("/extract", methods=["GET"])
def extract_index():
    return render_template(
        "extract.html",
        active_page="extract",
        ffmpeg_available=extractor.check_ffmpeg(),
    )


@app.route("/extract", methods=["POST"])
def extract_action():
    """Stream NDJSON progress while ffmpeg runs. Final event holds download token."""
    if not extractor.check_ffmpeg():
        return jsonify({"error": "ffmpeg/ffprobe not installed (brew install ffmpeg)"}), 500
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded = request.files["video"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    fmt = request.form.get("format", "mp3")
    if fmt not in extractor.supported_formats():
        return jsonify({"error": f"Unsupported format: {fmt}"}), 400
    bitrate = request.form.get("bitrate") or None

    info = extractor.format_info(fmt)
    base = os.path.splitext(uploaded.filename)[0] or "audio"
    out_ext = info["ext"]
    download_filename = f"{base}.{out_ext}"

    # Save upload to temp.
    in_suffix = os.path.splitext(uploaded.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=in_suffix) as tmp_in:
        uploaded.save(tmp_in.name)
        in_path = tmp_in.name

    # Reserve output path.
    out_fd, out_path = tempfile.mkstemp(suffix=f".{out_ext}", prefix="audio_extract_")
    os.close(out_fd)

    q: "queue.Queue[dict]" = queue.Queue()
    SENTINEL = {"__done__": True}

    def post(event_type: str, **fields):
        q.put({"event": event_type, **fields})

    def worker():
        try:
            t0 = time.time()
            duration = extractor.get_duration(in_path)
            post(
                "progress",
                stage="probed",
                message=(
                    f"Source duration: {duration:.1f}s. Extracting to {fmt}…"
                    if duration > 0
                    else "Source duration unknown. Extracting…"
                ),
                duration=duration,
            )

            def _on_pct(pct: int, current_sec: float, total_sec: float):
                elapsed = f"{int(current_sec)}s"
                if total_sec > 0:
                    elapsed = f"{int(current_sec)}s / {int(total_sec)}s"
                post(
                    "progress",
                    stage="extracting",
                    message=f"Extracting: {pct}% ({elapsed})",
                    percent=pct,
                    elapsed=elapsed,
                )

            extractor.extract(in_path, out_path, fmt=fmt, bitrate=bitrate, on_progress=_on_pct)

            elapsed_total = time.time() - t0
            size = os.path.getsize(out_path)

            token = uuid.uuid4().hex
            with _EXTRACT_LOCK:
                _EXTRACT_RESULTS[token] = (out_path, download_filename, info["mime"])

            post("progress", stage="done", message=f"Done in {elapsed_total:.1f}s ({size/1024/1024:.1f} MB).", percent=100)
            post(
                "result",
                download_url=url_for("extract_download", token=token),
                filename=download_filename,
                format=fmt,
                size=size,
            )

        except Exception as e:
            post("error", error=f"{type(e).__name__}: {e}")
            try:
                os.remove(out_path)
            except OSError:
                pass
        finally:
            try:
                os.remove(in_path)
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


@app.route("/extract/download/<token>", methods=["GET"])
def extract_download(token):
    """Serve extracted audio. File deleted after stream completes (single-use token)."""
    with _EXTRACT_LOCK:
        entry = _EXTRACT_RESULTS.pop(token, None)
    if not entry:
        return jsonify({"error": "Invalid or expired download token"}), 404

    path, filename, mime = entry

    def stream():
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Length": str(os.path.getsize(path)),
    }
    return Response(stream(), mimetype=mime, headers=headers)


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
    print(f"Audio extraction: {'available' if extractor.check_ffmpeg() else 'unavailable (install ffmpeg)'}")
    print("Server: http://127.0.0.1:5000  (or http://localhost:5000)")
    print("  /         → Transcribe")
    print("  /extract  → Extract Audio")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
