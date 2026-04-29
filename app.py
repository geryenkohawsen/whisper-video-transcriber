import os
import tempfile

from flask import Flask, jsonify, render_template, request

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
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded = request.files["video"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

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

    suffix = os.path.splitext(uploaded.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = transcriber.transcribe(tmp_path, model_name=model_name, language=language)
        segments = result["segments"]

        diar_segments = None
        diar_warning = None
        if enable_diarization:
            try:
                diar_segments = diarizer.diarize(tmp_path, num_speakers=num_speakers)
            except diarizer.DiarizationUnavailable as e:
                diar_warning = str(e)
            except Exception as e:
                diar_warning = f"Diarization failed: {type(e).__name__}: {e}"

        response = {
            "language": result["language"],
            "model": result["model"],
            "backend": result["backend"],
            "plain": formatters.plain(segments),
            "timestamped": formatters.timestamped(segments),
            "dialogue": formatters.dialogue(segments, diar_segments),
            "segments": segments,
            "diarization_used": diar_segments is not None,
            "diarization_warning": diar_warning,
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    backend = transcriber.detect_backend()
    print(f"Backend: {backend} (mlx = Apple Silicon GPU; torch = CPU fallback)")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Diarization: {'available' if diarizer.is_available() else 'unavailable (no HF_TOKEN or pyannote missing)'}")
    print("Server: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
