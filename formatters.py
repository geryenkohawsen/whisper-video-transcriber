"""
Output formatters: plain, timestamped, dialogue (speaker-labeled).
Each format has plaintext + markdown variants.

Alignment strategy for dialogue:
  For each Whisper transcript segment, pick the diarization speaker
  whose time range has the largest overlap with that segment.
  Consecutive segments by same speaker are merged into one turn.
"""

from __future__ import annotations

from typing import List, Optional


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


# ---- plain ----

def plain(segments: List[dict]) -> str:
    return " ".join(s["text"].strip() for s in segments if s["text"].strip())


def plain_md(segments: List[dict]) -> str:
    text = plain(segments)
    return f"# Transcript\n\n{text}\n"


# ---- timestamped ----

def timestamped(segments: List[dict]) -> str:
    lines = []
    for s in segments:
        if not s["text"].strip():
            continue
        ts = f"[{format_timestamp(s['start'])} → {format_timestamp(s['end'])}]"
        lines.append(f"{ts} {s['text'].strip()}")
    return "\n".join(lines)


def timestamped_md(segments: List[dict]) -> str:
    lines = ["# Transcript (Timestamped)\n"]
    for s in segments:
        if not s["text"].strip():
            continue
        ts = f"{format_timestamp(s['start'])} → {format_timestamp(s['end'])}"
        lines.append(f"- **`{ts}`** {s['text'].strip()}")
    return "\n".join(lines) + "\n"


# ---- dialogue (speaker-labeled) ----

def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(
    transcript_segments: List[dict],
    diarization_segments: List[dict],
) -> List[dict]:
    """
    Returns transcript segments with `speaker` field added.
    Speaker is the diarization label with max time overlap, or None.
    """
    out = []
    for seg in transcript_segments:
        best_label = None
        best_overlap = 0.0
        for d in diarization_segments:
            ov = _overlap(seg["start"], seg["end"], d["start"], d["end"])
            if ov > best_overlap:
                best_overlap = ov
                best_label = d["speaker"]
        out.append({**seg, "speaker": best_label})
    return out


def _relabel(segments: List[dict]) -> List[dict]:
    """Map raw pyannote labels (SPEAKER_00...) to Speaker 1, Speaker 2, ... in order of appearance."""
    mapping = {}
    counter = 1
    out = []
    for s in segments:
        raw = s.get("speaker")
        if raw and raw not in mapping:
            mapping[raw] = f"Speaker {counter}"
            counter += 1
        label = mapping.get(raw, "Unknown")
        out.append({**s, "speaker": label})
    return out


def build_turns(
    transcript_segments: List[dict],
    diarization_segments: List[dict],
) -> List[dict]:
    """Returns list of {speaker, text, start, end}."""
    tagged = assign_speakers(transcript_segments, diarization_segments)
    tagged = _relabel(tagged)

    turns: List[dict] = []
    for seg in tagged:
        text = seg["text"].strip()
        if not text:
            continue
        if turns and turns[-1]["speaker"] == seg["speaker"]:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = seg["end"]
        else:
            turns.append(
                {
                    "speaker": seg["speaker"],
                    "text": text,
                    "start": seg["start"],
                    "end": seg["end"],
                }
            )
    return turns


def dialogue(
    transcript_segments: List[dict],
    diarization_segments: Optional[List[dict]],
) -> str:
    """
    [mm:ss] Speaker 1: "..."

    [mm:ss] Speaker 2: "..."
    """
    if not diarization_segments:
        return ""

    turns = build_turns(transcript_segments, diarization_segments)
    return "\n\n".join(
        f'[{format_timestamp(t["start"])}] {t["speaker"]}: "{t["text"]}"'
        for t in turns
    )


def dialogue_md(
    transcript_segments: List[dict],
    diarization_segments: Optional[List[dict]],
) -> str:
    if not diarization_segments:
        return ""

    turns = build_turns(transcript_segments, diarization_segments)
    lines = ["# Dialogue Transcript\n"]
    for t in turns:
        ts = format_timestamp(t["start"])
        lines.append(f"**{t['speaker']}** _`{ts}`_:\n\n> {t['text']}\n")
    return "\n".join(lines)
