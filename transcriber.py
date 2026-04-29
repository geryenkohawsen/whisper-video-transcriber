"""
Transcription backend abstraction.

Tradeoffs (MacBook Air M4, 36GB unified memory):

  Backend          Model         RAM peak   Speed*     Accuracy
  ---------------------------------------------------------------
  mlx-whisper      large-v3      ~3.5 GB    ~1.0x RT   highest
  mlx-whisper      turbo         ~1.6 GB    ~3.0x RT   very high
  mlx-whisper      medium        ~1.5 GB    ~2.0x RT   high
  openai-whisper   large-v3      ~10 GB     ~0.3x RT   highest
  openai-whisper   medium        ~5 GB      ~0.6x RT   high

  * realtime factor; >1.0 = faster than realtime.

MLX uses Apple Silicon GPU (Metal). PyTorch backend is CPU-only here
because openai-whisper does not auto-route to MPS for all ops.
MLX is preferred. PyTorch is a fallback so the app never hard-fails.
"""

from __future__ import annotations

import threading
from typing import Optional

# Map UI-friendly names to MLX repo IDs (HF mlx-community).
MLX_REPOS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "medium": "mlx-community/whisper-medium-mlx",
}

# Fallback model order when requested model fails to load.
FALLBACK_ORDER = ["large-v3", "turbo", "medium"]

_BACKEND = None  # "mlx" | "torch"
_loaded_model = None
_loaded_name = None
_lock = threading.Lock()


def _try_import_mlx():
    try:
        import mlx_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def detect_backend() -> str:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = "mlx" if _try_import_mlx() else "torch"
    return _BACKEND


def _load_torch(name: str):
    import whisper

    # openai-whisper accepts these names directly.
    return whisper.load_model(name)


def transcribe(
    audio_path: str,
    model_name: str = "large-v3",
    language: Optional[str] = None,
) -> dict:
    """
    Run transcription. Returns dict with: text, language, segments[start,end,text].

    Tries requested model first, then falls back through FALLBACK_ORDER on
    OOM / load errors. Raises RuntimeError only if every fallback fails.
    """
    backend = detect_backend()
    candidates = [model_name] + [m for m in FALLBACK_ORDER if m != model_name]
    last_err: Optional[Exception] = None

    for candidate in candidates:
        try:
            if backend == "mlx":
                return _transcribe_mlx(audio_path, candidate, language)
            return _transcribe_torch(audio_path, candidate, language)
        except (MemoryError, RuntimeError, ValueError, OSError) as e:
            last_err = e
            # Reset cached torch model so we can try smaller one.
            _reset_torch_cache()
            continue

    raise RuntimeError(f"All transcription models failed. Last error: {last_err}")


def _transcribe_mlx(audio_path: str, model_name: str, language: Optional[str]) -> dict:
    import mlx_whisper

    repo = MLX_REPOS.get(model_name, MLX_REPOS["large-v3"])
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=repo,
        language=language,
        word_timestamps=False,
        verbose=False,
    )
    return _normalize(result, used_model=model_name, used_backend="mlx")


def _transcribe_torch(audio_path: str, model_name: str, language: Optional[str]) -> dict:
    global _loaded_model, _loaded_name
    with _lock:
        if _loaded_model is None or _loaded_name != model_name:
            _loaded_model = _load_torch(model_name)
            _loaded_name = model_name
        model = _loaded_model

    result = model.transcribe(audio_path, language=language, fp16=False, verbose=False)
    return _normalize(result, used_model=model_name, used_backend="torch")


def _reset_torch_cache():
    global _loaded_model, _loaded_name
    with _lock:
        _loaded_model = None
        _loaded_name = None


def _normalize(result: dict, used_model: str, used_backend: str) -> dict:
    return {
        "text": (result.get("text") or "").strip(),
        "language": result.get("language"),
        "segments": [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": s["text"].strip(),
            }
            for s in result.get("segments", [])
        ],
        "model": used_model,
        "backend": used_backend,
    }
