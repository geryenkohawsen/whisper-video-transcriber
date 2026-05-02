"""
Audio extraction using ffmpeg (subprocess).

Streams ffmpeg's `-progress pipe:1` output line-by-line to compute live percent
against the file's total duration (queried via ffprobe).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable, List, Optional


class ExtractionError(RuntimeError):
    pass


# Map output format → (extension, codec args, mime).
# `args` is appended to the ffmpeg command after `-vn`.
_FORMATS = {
    "mp3":  {"ext": "mp3",  "codec": ["-c:a", "libmp3lame"], "mime": "audio/mpeg",   "supports_bitrate": True},
    "m4a":  {"ext": "m4a",  "codec": ["-c:a", "aac"],         "mime": "audio/mp4",    "supports_bitrate": True},
    "wav":  {"ext": "wav",  "codec": ["-c:a", "pcm_s16le"],   "mime": "audio/wav",    "supports_bitrate": False},
    "flac": {"ext": "flac", "codec": ["-c:a", "flac"],        "mime": "audio/flac",   "supports_bitrate": False},
    "ogg":  {"ext": "ogg",  "codec": ["-c:a", "libvorbis"],   "mime": "audio/ogg",    "supports_bitrate": True},
    # `copy` skips re-encoding — fastest, but extension depends on source codec.
    # Default to .m4a wrapper (works for AAC). User gets warning if mismatch.
    "copy": {"ext": "m4a",  "codec": ["-c:a", "copy"],        "mime": "audio/mp4",    "supports_bitrate": False},
}


def supported_formats() -> List[str]:
    return list(_FORMATS.keys())


def format_info(fmt: str) -> dict:
    if fmt not in _FORMATS:
        raise ExtractionError(f"Unsupported format: {fmt}")
    return _FORMATS[fmt]


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def get_duration(audio_or_video_path: str) -> float:
    """Return media duration in seconds via ffprobe. 0.0 if unknown."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_or_video_path,
            ],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return float(out.stdout.strip() or 0.0)
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def _build_args(in_path: str, out_path: str, fmt: str, bitrate: Optional[str]) -> List[str]:
    info = format_info(fmt)
    args = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-vn",  # no video
        "-progress", "pipe:1",
        "-nostats",
        "-loglevel", "error",
    ]
    args += info["codec"]
    if bitrate and info["supports_bitrate"]:
        args += ["-b:a", bitrate]
    args += [out_path]
    return args


def extract(
    in_path: str,
    out_path: str,
    fmt: str = "mp3",
    bitrate: Optional[str] = "192k",
    on_progress: Optional[Callable[[int, float, float], None]] = None,
) -> None:
    """
    Run ffmpeg to extract audio. Calls `on_progress(percent, current_seconds, total_seconds)`
    repeatedly during processing. Raises ExtractionError on failure.
    """
    if not check_ffmpeg():
        raise ExtractionError("ffmpeg or ffprobe not found in PATH")

    duration = get_duration(in_path)  # may be 0.0 (unknown)
    args = _build_args(in_path, out_path, fmt, bitrate)

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered so we get progress in real time
    )

    last_pct = -1
    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue

            # ffmpeg -progress emits key=value, blocks ending in `progress=continue|end`.
            if "=" not in line:
                continue
            key, value = line.split("=", 1)

            current_seconds = None
            if key == "out_time_us" or key == "out_time_ms":
                # Both are microseconds in modern ffmpeg (despite "_ms" naming).
                try:
                    current_seconds = int(value) / 1_000_000
                except ValueError:
                    current_seconds = None
            elif key == "out_time":
                current_seconds = _parse_hhmmss(value)
            elif key == "progress" and value == "end":
                if on_progress:
                    try:
                        on_progress(100, duration, duration)
                    except Exception:
                        pass
                continue

            if current_seconds is None or duration <= 0:
                continue

            pct = int(min(99, current_seconds * 100 / duration))
            if pct == last_pct:
                continue
            last_pct = pct
            if on_progress:
                try:
                    on_progress(pct, current_seconds, duration)
                except Exception:
                    pass

        proc.wait()
        if proc.returncode != 0:
            err = (proc.stderr.read() if proc.stderr else "").strip()
            raise ExtractionError(f"ffmpeg failed (exit {proc.returncode}): {err or 'unknown error'}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _parse_hhmmss(s: str) -> Optional[float]:
    # "00:01:23.456000"
    try:
        parts = s.split(":")
        if len(parts) != 3:
            return None
        h, m = int(parts[0]), int(parts[1])
        sec = float(parts[2])
        return h * 3600 + m * 60 + sec
    except (ValueError, IndexError):
        return None
