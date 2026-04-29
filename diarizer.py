"""
Speaker diarization via pyannote.audio 3.1.

Requires:
  - HF_TOKEN env var (https://huggingface.co/settings/tokens)
  - User accepted terms at:
        https://huggingface.co/pyannote/speaker-diarization-3.1
        https://huggingface.co/pyannote/segmentation-3.0

Pipeline runs on Apple Silicon GPU (MPS) when available; falls back to CPU.
Diarization on a 30-min file takes ~1-3 min on M4.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

_pipeline = None
_lock = threading.Lock()


class DiarizationUnavailable(RuntimeError):
    """Raised when diarization backend cannot be loaded (no token, missing deps)."""


def is_available() -> bool:
    if not os.environ.get("HF_TOKEN"):
        return False
    try:
        import pyannote.audio  # noqa: F401
        return True
    except ImportError:
        return False


def _load_pipeline():
    global _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise DiarizationUnavailable(
                "HF_TOKEN not set. Get token at https://huggingface.co/settings/tokens "
                "and accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise DiarizationUnavailable(f"pyannote.audio not installed: {e}")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token,
        )

        # Use Apple GPU if available.
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
        elif torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        _pipeline = pipeline
        return _pipeline


def diarize(audio_path: str, num_speakers: Optional[int] = None) -> List[dict]:
    """
    Returns list of {start, end, speaker} dicts sorted by start time.
    `speaker` is "SPEAKER_00", "SPEAKER_01", ... from pyannote.
    """
    pipeline = _load_pipeline()

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(audio_path, **kwargs)

    # community-1 returns DiarizeOutput; access .speaker_diarization for iteration.
    # Fallback to direct iteration for older pyannote Annotation objects.
    iterable = getattr(diarization, "speaker_diarization", diarization)

    segments = []
    for turn, speaker in iterable:
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
            }
        )
    segments.sort(key=lambda s: s["start"])
    return segments
