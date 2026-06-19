"""Phase-1 parity oracle: the C++ ``whisperx_core`` edits algorithms vs the
pure-Python ``app.edits._py_*`` functions they replace (the ``edits`` stage).

Two things are proven here:

1. **Per-function parity** — group_turns / distinct_speakers / next_speaker_key /
   coalesce_segments / realign_words / apply_turn_edit / apply_turn_reassign /
   undo_last produce identical results on each side, including **exact ==** on the
   interpolated word timestamps (the borrow arithmetic is float-order sensitive;
   edits.cpp is built ``-ffp-contract=off`` to match CPython bit-for-bit). The
   volatile delta ``ts`` is dropped before comparison.
2. **difflib fidelity** — the C++ ``matching_blocks`` primitive realign_words is
   built on matches ``difflib.SequenceMatcher(autojunk=False).get_matching_blocks()``
   on adversarial token lists (repeats, ties, a >200-element list that would
   trip difflib's autojunk pruning if it were on, and unicode).

Skipped unless the C++ module has been built (``cmake --build build``); run with
the build dir on the path (``PYTHONPATH=build``).
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

for cand in (ROOT / "build", ROOT / "build" / "Release", ROOT / "build" / "Debug"):
    if cand.is_dir():
        sys.path.insert(0, str(cand))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

whisperx_core = pytest.importorskip(
    "whisperx_core", reason="build the C++ module first: cmake --build build")

from app import edits as ed  # noqa: E402

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")


def _word(token, start=None, end=None):
    w = {"word": token}
    if start is not None:
        w["start"], w["end"] = start, end
    return w


def _seg(start, end, text, speaker=None, words=None):
    s = {"start": start, "end": end, "text": text}
    if speaker is not None:
        s["speaker"] = speaker
    if words is not None:
        s["words"] = words
    return s


def _strip_ts(delta: dict) -> dict:
    """Drop the volatile ISO timestamp after asserting its format."""
    assert _ISO.match(delta["ts"]), f"ts={delta['ts']!r} not ISO-8601 UTC seconds"
    return {k: v for k, v in delta.items() if k != "ts"}


@pytest.fixture
def diarized():
    return [
        _seg(0.0, 1.0, "Hello there.", "SPEAKER_00",
             [_word("Hello", 0.0, 0.5), _word("there.", 0.5, 1.0)]),
        _seg(1.0, 2.0, "How are you?", "SPEAKER_00",
             [_word("How", 1.0, 1.4), _word("are", 1.4, 1.7), _word("you?", 1.7, 2.0)]),
        _seg(2.0, 3.0, "Fine thanks.", "SPEAKER_01",
             [_word("Fine", 2.0, 2.5), _word("thanks.", 2.5, 3.0)]),
        _seg(3.0, 4.0, "Good.", "SPEAKER_00", [_word("Good.", 3.0, 4.0)]),
    ]


# --- per-function parity -----------------------------------------------------

def test_group_turns_parity(diarized):
    py = ed._py_group_turns(diarized)
    cpp = [ed.Turn(**d) for d in whisperx_core.group_turns(diarized)]
    assert py == cpp


def test_distinct_speakers_parity(diarized):
    assert (ed._py_distinct_speakers(diarized) ==
            whisperx_core.distinct_speakers(diarized))


@pytest.mark.parametrize("keys", [
    [], ["SPEAKER_00", "SPEAKER_01"], ["SPEAKER_00", "SPEAKER_05", "Alice"],
    ["Alice", "Bob"], ["SPEAKER_09", "SPEAKER_10"],
])
def test_next_speaker_key_parity(keys):
    assert ed._py_next_speaker_key(keys) == whisperx_core.next_speaker_key(keys)


def test_coalesce_segments_parity():
    segs = [_seg(0.0, 0.1, "uh", "SPEAKER_00", [_word("uh", 0.0, 0.1)]),
            _seg(0.1, 0.5, "hello there", "SPEAKER_00",
                 [_word("hello", 0.1, 0.3), _word("there", 0.3, 0.5)]),
            _seg(0.5, 1.5, "fine", "SPEAKER_01", [_word("fine", 0.5, 1.5)]),
            _seg(1.5, 1.55, "x", "SPEAKER_01", [])]
    assert (ed._py_coalesce_segments(segs) ==
            whisperx_core.coalesce_segments(segs, ed.SEGMENT_MIN_DURATION))


@pytest.mark.parametrize("new_text,start,end", [
    ("Hello there. How are you? Right.", 0.0, 2.0),    # trailing borrow
    ("Hello there. How are really you?", 0.0, 2.0),    # midway borrow (zero gap)
    ("Hello there. you?", 0.0, 2.0),                   # deletion
    ("pre core post", 0.0, 3.0),                       # leading/trailing anchors
    ("A new B", 0.0, 3.0),                             # interpolate across a gap
    ("totally different words", 0.0, 2.0),             # full rewrite -> []
])
def test_realign_words_parity_exact_floats(new_text, start, end):
    old = [_word("Hello", 0.0, 0.5), _word("there.", 0.5, 1.0),
           _word("How", 1.0, 1.4), _word("are", 1.4, 1.7), _word("you?", 1.7, 2.0)]
    py = ed._py_realign_words(old, new_text, start, end)
    cpp = whisperx_core.realign_words(old, new_text, start, end)
    assert py == cpp  # exact equality, including the interpolated floats


def test_realign_zero_gap_borrow_is_bit_exact():
    # The float-order-sensitive case the -ffp-contract=off flag protects.
    old = [_word("A", 0.0, 0.5), _word("B", 0.5, 1.0)]
    py = ed._py_realign_words(old, "A mid B", 0.0, 1.0)
    cpp = whisperx_core.realign_words(old, "A mid B", 0.0, 1.0)
    assert py == cpp
    assert [w.get("start") for w in cpp] == [w.get("start") for w in py]


def test_apply_turn_edit_parity(diarized):
    py_segs, py_delta = ed._py_apply_turn_edit(diarized, 0, "Hello there. plus")
    cpp_segs, cpp_delta = whisperx_core.apply_turn_edit(
        diarized, 0, "Hello there. plus")
    assert py_segs == cpp_segs
    assert _strip_ts(py_delta) == _strip_ts(cpp_delta)


def test_apply_turn_reassign_parity(diarized):
    py_segs, py_delta = ed._py_apply_turn_reassign(diarized, 1, "SPEAKER_02")
    cpp_segs, cpp_delta = whisperx_core.apply_turn_reassign(
        diarized, 1, "SPEAKER_02")
    assert py_segs == cpp_segs
    assert _strip_ts(py_delta) == _strip_ts(cpp_delta)


def test_apply_turn_reassign_nochange_parity(diarized):
    with pytest.raises(ed.NoChange):
        ed._py_apply_turn_reassign(diarized, 1, "SPEAKER_01")
    with pytest.raises(whisperx_core.NoChange):
        whisperx_core.apply_turn_reassign(diarized, 1, "SPEAKER_01")


@pytest.mark.parametrize("ti", [-1, 99])
def test_apply_turn_edit_out_of_range_parity(diarized, ti):
    with pytest.raises(IndexError):
        ed._py_apply_turn_edit(diarized, ti, "x")
    with pytest.raises(IndexError):
        whisperx_core.apply_turn_edit(diarized, ti, "x")


def test_undo_last_parity(diarized):
    _, delta = ed._py_apply_turn_edit(diarized, 0, "edited")
    py_segs, py_hist = ed._py_undo_last(diarized, [delta])
    cpp_segs, cpp_hist = whisperx_core.undo_last(diarized, [delta])
    assert py_segs == cpp_segs
    assert py_hist == cpp_hist


# --- difflib fidelity (the matching-block primitive) -------------------------

def _difflib_blocks(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(m.a, m.b, m.size) for m in sm.get_matching_blocks()]


@pytest.mark.parametrize("a,b", [
    (["a", "b", "c"], ["a", "b", "c"]),
    (["a", "b", "c"], ["a", "c"]),                       # deletion
    (["a", "c"], ["a", "b", "c"]),                       # insertion
    (["a", "x", "a"], ["a", "y", "a"]),                  # repeats + tie-break
    (["the", "the", "the"], ["the", "the"]),
    (list("abcabcabc"), list("abcXabc")),
    (["héllo", "wörld"], ["héllo", "wörld", "!"]),       # unicode
    (["x"] * 300, ["x"] * 300),                          # > 200: no autojunk drift
    (["q"] * 250 + ["z"], ["q"] * 250),
])
def test_matching_blocks_match_difflib(a, b):
    cpp = [tuple(t) for t in whisperx_core.matching_blocks(a, b)]
    assert cpp == _difflib_blocks(a, b)
