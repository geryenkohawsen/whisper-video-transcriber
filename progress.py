"""
Progress capture utilities.

Both `mlx_whisper.transcribe` and `whisper.transcribe` do `import tqdm` then
construct `tqdm.tqdm(total=content_frames, ...)` and call `.update(n)` per chunk.
We swap `tqdm.tqdm` for a subclass that fires a callback on each update, so the
Flask streaming layer can forward live percentages to the browser.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Optional

import tqdm as _tqdm_mod


ProgressCallback = Callable[[int, int], None]  # (current, total)


class _CallbackTqdm(_tqdm_mod.tqdm):
    """tqdm subclass that fires a callback after each update."""
    _callback: Optional[ProgressCallback] = None

    def update(self, n: int = 1):  # type: ignore[override]
        ret = super().update(n)
        cb = _CallbackTqdm._callback
        if cb and self.total:
            try:
                cb(int(self.n), int(self.total))
            except Exception:
                # Never let progress reporting break transcription.
                pass
        return ret


@contextlib.contextmanager
def capture_tqdm(callback: ProgressCallback):
    """
    Context that swaps the global `tqdm.tqdm` class with a callback-firing
    subclass. Any code inside that does `tqdm.tqdm(...)` will report progress
    via `callback(current, total)`.
    """
    original = _tqdm_mod.tqdm
    _CallbackTqdm._callback = callback
    _tqdm_mod.tqdm = _CallbackTqdm  # type: ignore[assignment]
    try:
        yield
    finally:
        _tqdm_mod.tqdm = original  # type: ignore[assignment]
        _CallbackTqdm._callback = None


class PyannoteHook:
    """
    Hook passed to pyannote.audio pipelines. Pipeline calls
    `hook(step_name, step_artifact, file=..., total=..., completed=...)`
    multiple times per step. We forward (completed, total, step_name) to a
    callback whenever both totals are present.
    """

    def __init__(self, callback: Callable[[int, int, str], None]):
        self._cb = callback

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if total is not None and completed is not None:
            try:
                self._cb(int(completed), int(total), str(step_name))
            except Exception:
                pass

    # pyannote sometimes uses hook as a context manager.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
