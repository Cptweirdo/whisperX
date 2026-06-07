"""Render a whisperx result dict into a Markdown transcript.

Speaker turns come from :func:`app.edits.group_turns`; the SPA renders the live
HTML transcript client-side from the JSON API, so only the Markdown export
remains server-side.
"""

from __future__ import annotations

import re
from typing import Optional

from app.edits import group_turns


def _fmt_ts(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _speaker_label(raw: Optional[str]) -> str:
    """SPEAKER_00 -> 'Speaker 1'; pass through anything else."""
    if not raw:
        return "Speaker"
    m = re.fullmatch(r"SPEAKER_(\d+)", str(raw))
    return f"Speaker {int(m.group(1)) + 1}" if m else str(raw)


def resolve_label(raw: Optional[str], names: Optional[dict] = None) -> str:
    """User-assigned name for a speaker key if set, else the default label."""
    if names and raw and names.get(raw):
        return names[raw]
    return _speaker_label(raw)


def render_markdown(result: dict, names: Optional[dict] = None,
                    title: Optional[str] = None) -> str:
    """Render a result as a Markdown transcript: a title heading, then one block
    per speaker turn — a bold speaker tag with the turn's ``[start - end]`` span,
    followed by the turn text. ``names`` maps raw speaker keys to display names.

    Mirrors :func:`render_transcript`'s turn grouping so the export matches the
    on-screen transcript (including edits, since callers pass the overlaid segments).
    """
    lines: list[str] = [f"# {title.strip()}" if title and title.strip() else "# Transcript", ""]
    segments = result.get("segments", [])
    if not segments:
        lines.append("_No speech detected._")
        return "\n".join(lines) + "\n"

    for t in group_turns(segments):
        text = t.text.strip()
        if not text:
            continue
        label = resolve_label(t.speaker, names)
        span = f"{_fmt_ts(t.start)} – {_fmt_ts(t.end)}"
        lines.append(f"**{label}** [{span}]")
        lines.append("")
        lines.append(text)
        lines.append("")
    return "\n".join(lines) + "\n"
