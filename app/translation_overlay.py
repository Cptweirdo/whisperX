"""Structure-locked translation overlay: pure, Flask-free text join.

A translation is **not** a frozen copy of the transcript. It stores only the
translated *strings*, keyed by each source segment's ``start`` time, alongside the
source text they were translated from. Segment boundaries, speaker assignment and
turn grouping always come from the *current* original at render time — so a speaker
reassignment (or rename) on the original propagates to every language for free, and
a translation can never drift to a different turn count than the original.

The join (:func:`apply_overlay`) walks the current original segments and, for each,
substitutes its translated string when one exists *and* the source text still
matches. When the source text changed (the user edited that turn) or no translation
exists yet, the segment is marked ``stale`` and falls back to the original text so
the rest of the translation stays usable.

Overlay payload shapes:

- **v2** (current): ``{"version": 2, ..., "entries": {start_key: {"src", "tr"}}}``
- **v1** (legacy): ``{..., "segments": [{"start", "end", "speaker", "text"}]}`` — read
  by deriving entries from its segments; with no stored source text, freshness falls
  back to an exact start match only.
"""

from __future__ import annotations

from typing import Optional


def start_key(start) -> Optional[str]:
    """Stable per-segment key: the start time to millisecond precision, or ``None``
    for a segment with no start (which therefore can carry no translation)."""
    if start is None:
        return None
    return f"{float(start):.3f}"


def build_entries(segments: list[dict], translated_texts: list[str]) -> dict:
    """Map ``start_key -> {"src": original_text, "tr": translated_text}``.

    Pairs each source segment with its translation (positional, like the translator's
    own input/output ordering). Start-less segments are skipped — they have no key.
    ``src`` is stored so a later text edit on the original can be detected as stale.
    """
    entries: dict[str, dict] = {}
    for seg, tr in zip(segments, translated_texts):
        key = start_key(seg.get("start"))
        if key is None:
            continue
        entries[key] = {"src": seg.get("text") or "", "tr": tr or ""}
    return entries


def _entries_of(overlay: dict) -> tuple[dict, bool]:
    """Return ``(entries, has_src)`` for either overlay shape.

    ``has_src`` is False for a legacy v1 overlay (no stored source text), where
    freshness can only be decided by an exact start match.
    """
    entries = overlay.get("entries")
    if entries is not None:
        return entries, True
    # Legacy v1: derive entries from the frozen segments. No source text to compare,
    # so every present key is treated as fresh on an exact start match.
    derived: dict[str, dict] = {}
    for seg in overlay.get("segments", []):
        key = start_key(seg.get("start"))
        if key is not None:
            derived[key] = {"tr": seg.get("text") or ""}
    return derived, False


def apply_overlay(orig_segments: list[dict], overlay: dict) -> list[dict]:
    """Join a translation overlay onto the current original segments.

    Returns one view segment per original segment, carrying the original's
    ``start``/``end``/``speaker`` (the speaker is *always* the live original's, which
    is how reassignment propagates) and either the translated text (``stale=False``)
    or the original text as a fallback (``stale=True``) when no current translation
    applies.
    """
    entries, has_src = _entries_of(overlay or {})
    out: list[dict] = []
    for seg in orig_segments:
        src_text = seg.get("text") or ""
        entry = entries.get(start_key(seg.get("start")))
        if entry is not None and (not has_src or entry.get("src", "") == src_text):
            text, stale = entry.get("tr") or "", False
        else:
            text, stale = src_text, True
        out.append({
            "start": seg.get("start"),
            "end": seg.get("end"),
            "speaker": seg.get("speaker"),
            "text": text,
            "stale": stale,
        })
    return out
