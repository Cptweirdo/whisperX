"""Pipeline-stats persistence on the session row.

Covers the durable storage of per-step timings/config/machine (the ``stats``
column) and graceful handling of rows that predate the column.
"""

from __future__ import annotations

import sqlite3

from app.store import SessionStore


def _store(tmp_path) -> SessionStore:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return SessionStore(str(data_dir))


def _stats() -> dict:
    return {
        "timings": {"decoding": 0.5, "transcribing": 42.3, "total": 90.0},
        "config": {"model": "small", "device": "cpu", "diarize": False},
        "machine": {"platform": "Linux-x86_64", "gpu": None, "torch": "2.0"},
        "audio_duration": 191.0,
        "finished_at": "2026-06-06T12:00:00+00:00",
    }


def test_stats_round_trip(tmp_path):
    """mark_done(stats=…) persists the blob; get() returns it as a dict."""
    store = _store(tmp_path)
    store.create("s1", filename="a.wav", audio_filename="audio.bin",
                 options={}, model="small")
    stats = _stats()
    store.mark_done("s1", language="en", diarized=False, model="small",
                    num_segments=3, duration=191.0, stats=stats)

    row = store.get("s1")
    assert row["stats"] == stats
    assert row["stats"]["timings"]["transcribing"] == 42.3


def test_stats_absent_when_not_supplied(tmp_path):
    """A done session with no stats reads back falsy (button stays hidden)."""
    store = _store(tmp_path)
    store.create("s2", filename="a.wav", audio_filename="audio.bin",
                 options={}, model="small")
    store.mark_done("s2", language="en", diarized=False, model="small",
                    num_segments=0, duration=0.0)

    row = store.get("s2")
    assert not row.get("stats")


def test_old_db_without_stats_column_is_migrated(tmp_path):
    """A DB created before the stats column gains it on open; old rows read falsy."""
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    db_path = db_dir / "sessions.db"

    # Simulate a legacy DB: sessions table with no `stats` column + one row.
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, filename TEXT, audio_filename TEXT,
            status TEXT NOT NULL, error TEXT, options TEXT, language TEXT,
            diarized INTEGER, model TEXT, num_segments INTEGER, duration REAL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        """
    )
    legacy.execute(
        "INSERT INTO sessions (id, status, created_at, updated_at) "
        "VALUES ('old', 'done', '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')"
    )
    legacy.commit()
    legacy.close()

    # Opening through SessionStore migrates the schema in place.
    store = SessionStore(str(db_dir))
    cols = {r["name"] for r in store._db.execute("PRAGMA table_info(sessions)")}
    assert "stats" in cols

    row = store.get("old")
    assert not row.get("stats")  # NULL → falsy, no crash

    # And the migrated DB can still store stats on a fresh session.
    store.create("new", filename="a.wav", audio_filename="audio.bin",
                 options={}, model="small")
    store.mark_done("new", language="en", diarized=False, model="small",
                    num_segments=1, duration=10.0, stats=_stats())
    assert store.get("new")["stats"] == _stats()
