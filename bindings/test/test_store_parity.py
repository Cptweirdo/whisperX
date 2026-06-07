"""Phase-1 parity oracle: the C++ ``whisperx_core.SessionStore`` vs the Python
``app.store._PyDbStore`` it replaces.

Two things are proven here:

1. **Behavioural parity** — the same op sequence on each store yields the same
   row dicts (field-for-field, semantic JSON), the same list ordering, settings,
   speaker-name and translation-status results. Volatile ``created_at`` /
   ``updated_at`` are checked for *format* (ISO-8601 UTC, seconds) but not equality
   (they're "now" at call time).
2. **On-disk compatibility (the in-place-upgrade contract, plan §2)** — a
   ``sessions.db`` written by one implementation is read back identically by the
   other, in both directions, with no schema drift.

Skipped unless the C++ module has been built (``cmake --build build``); run with
the build dir on the path (``PYTHONPATH=build``).
"""

from __future__ import annotations

import re
import sys
import tempfile
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

from app.store import _PyDbStore  # noqa: E402

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")


def _py_store(tmp_path) -> _PyDbStore:
    return _PyDbStore(str(tmp_path))


def _cpp_store(tmp_path):
    return whisperx_core.SessionStore(str(tmp_path))


def _strip_ts(row: dict | None) -> dict | None:
    """Drop the two volatile timestamp columns after asserting their format."""
    if row is None:
        return None
    for k in ("created_at", "updated_at"):
        assert _ISO.match(row[k]), f"{k}={row[k]!r} not ISO-8601 UTC seconds"
    return {k: v for k, v in row.items() if k not in ("created_at", "updated_at")}


@pytest.fixture
def stores(tmp_path):
    """A Python and a C++ store over two separate fresh data dirs."""
    return _py_store(tmp_path / "py"), _cpp_store(tmp_path / "cpp")


def _run_both(stores, fn):
    """Apply ``fn`` to each store; return (py_result, cpp_result)."""
    return fn(stores[0]), fn(stores[1])


def test_create_and_get_row_shape_parity(stores):
    opts = {"diarize": True, "model": "large-v2", "beam": 5}
    _run_both(stores, lambda s: s.create("s1", "Rec.mp3", "audio.mp3", opts, "large-v2"))
    py, cpp = _run_both(stores, lambda s: s.get("s1"))
    assert _strip_ts(py) == _strip_ts(cpp)
    # spot-check the typed columns survived the seam
    assert cpp["status"] == "queued"
    assert cpp["options"] == opts
    assert cpp["diarized"] is None
    assert cpp["created_at"] == cpp["updated_at"]
    # a missing id is None on both
    assert _run_both(stores, lambda s: s.get("nope")) == (None, None)


def test_lifecycle_parity(stores):
    _run_both(stores, lambda s: s.create("s1", "f", "a.wav", {}, None))
    _run_both(stores, lambda s: s.mark_running("s1"))
    _run_both(stores, lambda s: s.mark_stage("s1", "aligning"))
    _run_both(stores, lambda s: s.mark_duration("s1", 12.5))
    py, cpp = _run_both(stores, lambda s: s.get("s1"))
    assert _strip_ts(py) == _strip_ts(cpp)

    _run_both(stores, lambda s: s.mark_done(
        "s1", language="en", diarized=True, model="large-v2",
        num_segments=42, duration=12.5))
    py, cpp = _run_both(stores, lambda s: s.get("s1"))
    assert _strip_ts(py) == _strip_ts(cpp)
    assert cpp["diarized"] is True and cpp["num_segments"] == 42

    _run_both(stores, lambda s: s.mark_error("s1", "boom"))
    py, cpp = _run_both(stores, lambda s: s.get("s1"))
    assert _strip_ts(py) == _strip_ts(cpp)


def test_list_ordering_parity(stores):
    for sid in ("a", "c", "b"):
        _run_both(stores, lambda s, sid=sid: s.create(sid, "f", "a.wav", {}, None))
    py, cpp = _run_both(stores, lambda s: s.list())
    assert [r["id"] for r in py] == [r["id"] for r in cpp]
    assert [_strip_ts(r) for r in py] == [_strip_ts(r) for r in cpp]


def test_settings_parity(stores):
    py, cpp = _run_both(stores, lambda s: s.get_setting("active_model", "dflt"))
    assert py == cpp == "dflt"
    _run_both(stores, lambda s: s.set_setting("active_model", "large-v2"))
    _run_both(stores, lambda s: s.set_setting("active_model", "tiny"))  # upsert
    py, cpp = _run_both(stores, lambda s: s.get_setting("active_model"))
    assert py == cpp == "tiny"


def test_speaker_names_parity(stores):
    _run_both(stores, lambda s: s.create("s1", "f", "a.wav", {}, None))
    _run_both(stores, lambda s: s.set_speaker_name("s1", "SPEAKER_00", "Alice"))
    _run_both(stores, lambda s: s.set_speaker_name("s1", "SPEAKER_01", "Bob"))
    _run_both(stores, lambda s: s.set_speaker_name("s1", "SPEAKER_00", "  Alyssa "))
    _run_both(stores, lambda s: s.set_speaker_name("s1", "SPEAKER_01", "  "))  # clear
    py, cpp = _run_both(stores, lambda s: s.get_speaker_names("s1"))
    assert py == cpp == {"SPEAKER_00": "Alyssa"}


def test_translation_status_parity(stores):
    _run_both(stores, lambda s: s.create("s1", "f", "a.wav", {}, None))
    py, cpp = _run_both(
        stores, lambda s: s.set_translation_status("s1", "es", "running", service="deepl"))
    assert py == cpp
    _run_both(stores, lambda s: s.set_translation_status("s1", "es", "error", error="rl"))
    py, cpp = _run_both(stores, lambda s: s.get_translations("s1"))
    assert py == cpp and cpp["es"]["error"] == "rl"
    # success drops the error key on both
    py, cpp = _run_both(stores, lambda s: s.set_translation_status("s1", "es", "done"))
    assert py == cpp and "error" not in cpp["es"]
    assert _run_both(stores, lambda s: s.get_translations("nope")) == ({}, {})


def test_reconcile_and_active_jobs_parity(stores):
    _run_both(stores, lambda s: s.create("s1", "f", "a.wav", {}, None))
    _run_both(stores, lambda s: s.create("s2", "f", "a.wav", {}, None))
    _run_both(stores, lambda s: s.mark_done(
        "s2", language="en", diarized=False, model="tiny", num_segments=1, duration=1.0))
    py, cpp = _run_both(stores, lambda s: s.has_active_jobs())
    assert py == cpp is True
    _run_both(stores, lambda s: s.mark_running("s1"))
    py, cpp = _run_both(stores, lambda s: sorted(s.reconcile_startup()))
    assert py == cpp == ["s1"]
    py, cpp = _run_both(stores, lambda s: s.get("s1")["status"])
    assert py == cpp == "queued"


def test_delete_parity(stores, tmp_path):
    _run_both(stores, lambda s: s.create("s1", "f", "a.wav", {}, None))
    py, cpp = _run_both(stores, lambda s: s.delete("s1"))
    assert py == cpp is True
    assert _run_both(stores, lambda s: s.get("s1")) == (None, None)
    assert _run_both(stores, lambda s: s.delete("s1")) == (False, False)  # already gone


# --- on-disk compatibility (the in-place-upgrade contract) -----------------

def test_python_db_read_by_cpp(tmp_path):
    """A DB written by the Python store reads back identically through C++."""
    d = tmp_path / "shared"
    py = _PyDbStore(str(d))
    py.create("s1", "Rec.mp3", "audio.mp3", {"diarize": True}, "large-v2")
    py.mark_done("s1", language="en", diarized=True, model="large-v2",
                 num_segments=7, duration=3.5)
    py.set_speaker_name("s1", "SPEAKER_00", "Alice")
    py.set_translation_status("s1", "es", "done", service="deepl")
    py.close()  # checkpoint WAL before the other connection opens

    cpp = whisperx_core.SessionStore(str(d))
    reopened = _PyDbStore(str(d))  # the Python view of the same on-disk file
    assert _strip_ts(cpp.get("s1")) == _strip_ts(reopened.get("s1"))
    assert cpp.get_speaker_names("s1") == {"SPEAKER_00": "Alice"}
    assert cpp.get_translations("s1")["es"]["status"] == "done"


def test_cpp_db_read_by_python(tmp_path):
    """A DB written by the C++ store reads back identically through Python."""
    d = tmp_path / "shared"
    cpp = whisperx_core.SessionStore(str(d))
    cpp.create("s1", "Rec.mp3", "audio.mp3", {"diarize": False}, "tiny")
    cpp.mark_error("s1", "kaboom")
    cpp.set_translation_status("s1", "fr", "running", service="google")
    cpp.close()

    py = _PyDbStore(str(d))
    row = py.get("s1")
    assert row["status"] == "error" and row["error"] == "kaboom"
    assert row["options"] == {"diarize": False}
    assert py.get_translations("s1")["fr"]["service"] == "google"
