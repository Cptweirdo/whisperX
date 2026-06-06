"""Unit tests for the structure-locked translation join (``app.translation_overlay``).

The overlay stores only translated strings keyed by source-segment start time;
``apply_overlay`` joins them onto the *current* original so speaker/structure always
track the live transcript. These pin the join's freshness + fallback rules (O1-O10
in the plan).
"""

from __future__ import annotations

from app.translation_overlay import apply_overlay, build_entries, start_key


# O1 -------------------------------------------------------------------------
def test_start_key_formats_and_handles_none():
    assert start_key(0) == "0.000"
    assert start_key(1.2345) == "1.234" or start_key(1.2345) == "1.235"  # rounding
    assert start_key(None) is None


# O2 -------------------------------------------------------------------------
def test_build_entries_stores_src_and_tr_skips_startless():
    segs = [
        {"start": 0.0, "text": "hello"},
        {"start": 1.0, "text": "world"},
        {"start": None, "text": "ghost"},  # no key -> skipped
    ]
    entries = build_entries(segs, ["hola", "mundo", "fantasma"])
    assert entries == {
        "0.000": {"src": "hello", "tr": "hola"},
        "1.000": {"src": "world", "tr": "mundo"},
    }


def _overlay(entries):
    return {"version": 2, "entries": entries}


# O3 -------------------------------------------------------------------------
def test_apply_overlay_fresh_on_exact_match():
    orig = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hello"}]
    overlay = _overlay({"0.000": {"src": "hello", "tr": "hola"}})
    out = apply_overlay(orig, overlay)
    assert out == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00",
                    "text": "hola", "stale": False}]


# O4 -------------------------------------------------------------------------
def test_apply_overlay_stale_on_src_mismatch():
    orig = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hello EDITED"}]
    overlay = _overlay({"0.000": {"src": "hello", "tr": "hola"}})
    out = apply_overlay(orig, overlay)
    assert out[0]["text"] == "hello EDITED"  # falls back to current original text
    assert out[0]["stale"] is True


# O5 -------------------------------------------------------------------------
def test_apply_overlay_stale_on_missing_key():
    orig = [{"start": 5.0, "end": 6.0, "speaker": "SPEAKER_00", "text": "new turn"}]
    overlay = _overlay({"0.000": {"src": "hello", "tr": "hola"}})
    out = apply_overlay(orig, overlay)
    assert out[0]["text"] == "new turn" and out[0]["stale"] is True


# O6 -------------------------------------------------------------------------
def test_apply_overlay_speaker_always_from_current_original():
    overlay = _overlay({"0.000": {"src": "hello", "tr": "hola"}})
    # Same overlay, different current speaker -> the join reflects the reassignment.
    a = apply_overlay([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hello"}], overlay)
    b = apply_overlay([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_09", "text": "hello"}], overlay)
    assert a[0]["speaker"] == "SPEAKER_00"
    assert b[0]["speaker"] == "SPEAKER_09"
    assert a[0]["text"] == b[0]["text"] == "hola"  # text unaffected by reassignment


# O7 -------------------------------------------------------------------------
def test_apply_overlay_startless_segment_is_stale():
    orig = [{"start": None, "end": None, "speaker": "SPEAKER_00", "text": "untimed"}]
    out = apply_overlay(orig, _overlay({}))
    assert out[0]["text"] == "untimed" and out[0]["stale"] is True


# O8 -------------------------------------------------------------------------
def test_apply_overlay_empty_translation_string():
    orig = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hello"}]
    overlay = _overlay({"0.000": {"src": "hello", "tr": ""}})
    out = apply_overlay(orig, overlay)
    assert out[0]["text"] == "" and out[0]["stale"] is False  # fresh, just empty


# O9 -------------------------------------------------------------------------
def test_apply_overlay_v1_back_compat():
    # Legacy overlay carries frozen `segments` (no `entries`, no stored src).
    overlay = {"version": 1, "segments": [{"start": 0.0, "end": 1.0, "text": "hola"}]}
    orig = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hello"}]
    out = apply_overlay(orig, overlay)
    # Exact start match -> fresh (can't detect text drift without src).
    assert out[0]["text"] == "hola" and out[0]["stale"] is False
    # A start with no v1 segment -> stale fallback.
    out2 = apply_overlay([{"start": 9.0, "end": 9.5, "speaker": "SPEAKER_00", "text": "x"}], overlay)
    assert out2[0]["stale"] is True


# O10 ------------------------------------------------------------------------
def test_build_entries_duplicate_start_last_wins():
    segs = [{"start": 0.0, "text": "a"}, {"start": 0.0, "text": "b"}]
    entries = build_entries(segs, ["AA", "BB"])
    assert entries == {"0.000": {"src": "b", "tr": "BB"}}  # deterministic: last wins
