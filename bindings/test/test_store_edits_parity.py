"""Phase-1 parity oracle: the file-backed sidecars (edits/undo overlay +
translation files) through the C++ ``whisperx_core.SessionStore`` vs the
pure-Python ``app.store._PyStore``.

Two things are proven here:

1. **Behavioural parity** — the same edit/reassign/undo sequence through each
   store yields identical ``current_segments``, ``transcript.edits.json``
   overlays, history lengths, and undo round-trips (including the
   overlay-dropped-when-pristine behaviour) and identical translation-file I/O.
2. **On-disk compatibility** — a ``transcript.edits.json`` written by one store
   reads back identically through the other, in both directions.

Skipped unless the C++ module has been built (``cmake --build build``); run with
the build dir on the path (``PYTHONPATH=build``).
"""

from __future__ import annotations

import json
import os
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

from app.store import _PyStore  # noqa: E402


def _word(token, start, end):
    return {"word": token, "start": start, "end": end}


def _seg(start, end, text, speaker, words):
    return {"start": start, "end": end, "text": text, "speaker": speaker,
            "words": words}


DIARIZED = [
    _seg(0.0, 1.0, "Hello there.", "SPEAKER_00",
         [_word("Hello", 0.0, 0.5), _word("there.", 0.5, 1.0)]),
    _seg(1.0, 2.0, "How are you?", "SPEAKER_00",
         [_word("How", 1.0, 1.4), _word("are", 1.4, 1.7), _word("you?", 1.7, 2.0)]),
    _seg(2.0, 3.0, "Fine thanks.", "SPEAKER_01",
         [_word("Fine", 2.0, 2.5), _word("thanks.", 2.5, 3.0)]),
    _seg(3.0, 4.0, "Good.", "SPEAKER_00", [_word("Good.", 3.0, 4.0)]),
]

SID = "sess1"


def _write_result(data_dir: Path, sid: str, segments: list) -> None:
    sdir = data_dir / "sessions" / sid
    os.makedirs(sdir, exist_ok=True)
    with open(sdir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f)


def _edits_on_disk(data_dir: Path, sid: str):
    path = data_dir / "sessions" / sid / "transcript.edits.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def stores(tmp_path):
    """A Python and a C++ store over two separate data dirs, both seeded with the
    same transcript.json."""
    py_dir, cpp_dir = tmp_path / "py", tmp_path / "cpp"
    py_dir.mkdir()
    cpp_dir.mkdir()
    _write_result(py_dir, SID, DIARIZED)
    _write_result(cpp_dir, SID, DIARIZED)
    py = _PyStore(str(py_dir))
    cpp = whisperx_core.SessionStore(str(cpp_dir))
    return (py, py_dir), (cpp, cpp_dir)


def test_load_result_and_baseline_parity(stores):
    (py, _), (cpp, _) = stores
    assert py.load_result(SID) == cpp.load_result(SID)
    # No overlay yet -> coalesced original (DIARIZED is already coalesced).
    assert py.current_segments(SID, DIARIZED) == cpp.current_segments(SID, DIARIZED)
    assert py.load_edits(SID) is None and cpp.load_edits(SID) is None


def test_edit_sequence_parity(stores):
    (py, py_dir), (cpp, cpp_dir) = stores

    def both(fn):
        return fn(py), fn(cpp)

    # 1) collapse the opening multi-segment turn (interpolated word timing).
    p, c = both(lambda s: s.save_turn_edit(SID, 0, "Hello there. How are you? Right."))
    assert p == c
    # 2) reassign the SPEAKER_01 turn (now turn 1).
    p, c = both(lambda s: s.save_turn_reassign(SID, 1, "SPEAKER_02"))
    assert p == c

    # overlays match on disk (semantic JSON; key order may differ) ...
    assert _edits_on_disk(py_dir, SID) == _edits_on_disk(cpp_dir, SID)
    # ... and through the API.
    assert py.load_edits(SID) == cpp.load_edits(SID)
    assert py.current_segments(SID, DIARIZED) == cpp.current_segments(SID, DIARIZED)
    assert py.edit_history_len(SID) == cpp.edit_history_len(SID) == 2

    # 3) undo both edits; the overlay is dropped once fully reverted to baseline.
    p, c = both(lambda s: s.undo_turn_edit(SID))
    assert p == c
    p, c = both(lambda s: s.undo_turn_edit(SID))
    assert p == c
    assert py.load_edits(SID) is None and cpp.load_edits(SID) is None
    assert _edits_on_disk(py_dir, SID) is None
    assert _edits_on_disk(cpp_dir, SID) is None


def test_reassign_noop_writes_nothing_parity(stores):
    (py, py_dir), (cpp, cpp_dir) = stores
    p = py.save_turn_reassign(SID, 1, "SPEAKER_01")    # already SPEAKER_01
    c = cpp.save_turn_reassign(SID, 1, "SPEAKER_01")
    assert p == c
    assert _edits_on_disk(py_dir, SID) is None
    assert _edits_on_disk(cpp_dir, SID) is None


def test_history_cap_parity(tmp_path):
    py_dir, cpp_dir = tmp_path / "py", tmp_path / "cpp"
    segs = [_seg(0.0, 1.0, "start", "SPEAKER_00", [])]
    for d in (py_dir, cpp_dir):
        d.mkdir()
        _write_result(d, SID, segs)
    py = _PyStore(str(py_dir))
    cpp = whisperx_core.SessionStore(str(cpp_dir))
    for n in range(101):  # exceed HISTORY_LIMIT (100)
        assert py.save_turn_edit(SID, 0, f"edit {n}") == cpp.save_turn_edit(SID, 0, f"edit {n}")
    assert py.edit_history_len(SID) == cpp.edit_history_len(SID) == 100
    assert py.load_edits(SID) == cpp.load_edits(SID)


def test_translation_io_parity(stores):
    (py, py_dir), (cpp, cpp_dir) = stores
    payload = {"version": 2, "target_language": "es", "service": "deepl",
               "entries": {"0.000": {"src": "Hello", "tr": "Hola 😀"}}}
    py.save_translation(SID, "es", payload)
    cpp.save_translation(SID, "es", payload)
    assert py.load_translation(SID, "es") == cpp.load_translation(SID, "es") == payload
    assert py.load_translation(SID, "fr") is None
    assert cpp.load_translation(SID, "fr") is None


# --- on-disk compatibility (overlay written by one store, read by the other) -

def test_py_edits_read_by_cpp(tmp_path):
    d = tmp_path / "shared"
    d.mkdir()
    _write_result(d, SID, DIARIZED)
    py = _PyStore(str(d))
    py.save_turn_edit(SID, 0, "Hello there. extra")
    py.save_turn_reassign(SID, 1, "SPEAKER_02")

    cpp = whisperx_core.SessionStore(str(d))
    assert cpp.load_edits(SID) == py.load_edits(SID)
    assert cpp.current_segments(SID, DIARIZED) == py.current_segments(SID, DIARIZED)


def test_cpp_edits_read_by_py(tmp_path):
    d = tmp_path / "shared"
    d.mkdir()
    _write_result(d, SID, DIARIZED)
    cpp = whisperx_core.SessionStore(str(d))
    cpp.save_turn_edit(SID, 0, "Hello there. extra")
    cpp.save_turn_reassign(SID, 1, "SPEAKER_02")

    py = _PyStore(str(d))
    assert py.load_edits(SID) == cpp.load_edits(SID)
    assert py.current_segments(SID, DIARIZED) == cpp.current_segments(SID, DIARIZED)
