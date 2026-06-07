"""Contract tests for the JSON API (/api/*) the Svelte SPA consumes.

These import app.server (which is heavy-but-lazy: torch is only pulled when a
model actually loads). WHISPERX_NO_WARM=1 keeps the import from kicking off a
model download, and a temp data dir isolates the SQLite store.
"""

import json
import os
import tempfile

import pytest

os.environ["WHISPERX_NO_WARM"] = "1"
os.environ.setdefault("WHISPERX_DATA_DIR", tempfile.mkdtemp(prefix="wx-api-test-"))

# The web app depends on Flask + keyring (app/requirements.txt), which the base
# `uv sync` env doesn't install. Skip the whole module rather than error there.
try:
    from app import server  # noqa: E402
except Exception as exc:  # noqa: BLE001 - missing web deps -> skip, don't fail CI
    pytest.skip(f"web app deps unavailable ({exc})", allow_module_level=True)


@pytest.fixture()
def client():
    return server.app.test_client()


def _make_session(session_id: str, segments: list, *, filename="Rec") -> None:
    """Fabricate a finished session: a DB row + a transcript.json result file,
    bypassing the (model-bound) upload pipeline."""
    store = server._sessions
    os.makedirs(store.session_dir(session_id), exist_ok=True)
    store.create(session_id, filename=filename, audio_filename="audio.wav",
                 options={}, model="small")
    store.mark_done(session_id, language="en", diarized=True, model="small",
                    num_segments=len(segments), duration=segments[-1]["end"])
    with open(store.result_path(session_id), "w", encoding="utf-8") as f:
        json.dump({"segments": segments, "language": "en",
                   "num_segments": len(segments), "duration": segments[-1]["end"]}, f)


def _segments():
    return [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello there",
         "words": [{"word": "Hello", "start": 0.0, "end": 0.5},
                   {"word": "there", "start": 0.5, "end": 1.0}]},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "friend",
         "words": [{"word": "friend", "start": 1.0, "end": 2.0}]},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01", "text": "Hi",
         "words": [{"word": "Hi", "start": 2.0, "end": 3.0}]},
    ]


def test_session_detail_turns(client):
    _make_session("s_detail", _segments())
    data = client.get("/api/sessions/s_detail").get_json()
    turns = data["turns"]
    assert [t["label"] for t in turns] == ["Speaker 1", "Speaker 2"]
    # Turn 0 groups the two consecutive SPEAKER_00 segments into one bubble.
    assert turns[0]["index"] == 0
    assert [w["text"] for w in turns[0]["words"]] == ["Hello", "there", "friend"]
    assert turns[1]["index"] == 1
    assert data["speaker_names"] == {}
    assert data["can_undo"] is False


def test_turn_indices_match_group_turns(client):
    """The SPA edits by turn.index; it must match edits.group_turns exactly."""
    from app.edits import group_turns

    segs = _segments()
    _make_session("s_parity", segs)
    turns = client.get("/api/sessions/s_parity").get_json()["turns"]
    grouped = group_turns(segs)
    for t in turns:
        assert grouped[t["index"]].speaker == t["speaker"]


def test_edit_undo_roundtrip(client):
    _make_session("s_edit", _segments())
    res = client.post("/api/sessions/s_edit/turns/0", json={"text": "Hey friend"})
    body = res.get_json()
    assert body["can_undo"] is True
    assert body["turns"][0]["text"] == "Hey friend"

    undo = client.post("/api/sessions/s_edit/undo").get_json()
    assert undo["can_undo"] is False
    assert undo["turns"][0]["text"] == "Hello there friend"


def test_reassign_enrolls_new_speaker(client):
    _make_session("s_reassign", _segments())
    res = client.post("/api/sessions/s_reassign/turns/1/speaker", json={"name": "Alice"})
    assert res.status_code == 200
    labels = [t["label"] for t in res.get_json()["turns"]]
    assert "Alice" in labels


def test_reassign_merges_adjacent_turns_and_undo_restores(client):
    """The exact flow the browser e2e exercises: with three alternating turns
    [SP00, SP01, SP00], reassigning the middle turn to its neighbours' speaker
    collapses them into a single turn (enabling Undo); undo restores all three.
    Mirrors the Playwright e2e so the data contract is locked even when the browser
    suite can't run."""
    alternating = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Welcome everyone",
         "words": [{"word": "Welcome", "start": 0.0, "end": 0.5},
                   {"word": "everyone", "start": 0.5, "end": 1.0}]},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "Glad to be here",
         "words": [{"word": "Glad", "start": 1.0, "end": 2.0}]},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "Let us begin",
         "words": [{"word": "begin", "start": 2.0, "end": 3.0}]},
    ]
    _make_session("s_merge", alternating)
    before = client.get("/api/sessions/s_merge").get_json()["turns"]
    assert [t["label"] for t in before] == ["Speaker 1", "Speaker 2", "Speaker 1"]

    res = client.post("/api/sessions/s_merge/turns/1/speaker", json={"speaker": "SPEAKER_00"})
    merged = res.get_json()
    assert [t["label"] for t in merged["turns"]] == ["Speaker 1"]
    assert merged["can_undo"] is True

    undo = client.post("/api/sessions/s_merge/undo").get_json()
    assert [t["label"] for t in undo["turns"]] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert undo["can_undo"] is False


def test_reassign_duplicate_name_409(client):
    _make_session("s_dup", _segments())
    # "Speaker 1" already exists as the default label for SPEAKER_00.
    res = client.post("/api/sessions/s_dup/turns/1/speaker", json={"name": "Speaker 1"})
    assert res.status_code == 409


def test_speakers_list(client):
    _make_session("s_spk", _segments())
    data = client.get("/api/sessions/s_spk/speakers").get_json()
    assert {s["key"] for s in data} == {"SPEAKER_00", "SPEAKER_01"}


def test_rename_and_delete(client):
    _make_session("s_rd", _segments())
    r = client.post("/api/sessions/s_rd/rename", json={"name": "New title"})
    assert r.get_json()["filename"] == "New title"
    assert client.post("/api/sessions/s_rd/delete").get_json() == {"deleted": True}
    assert client.get("/api/sessions/s_rd").status_code == 404


def test_list_and_summary(client):
    _make_session("s_list", _segments())
    data = client.get("/api/sessions").get_json()
    assert data["summary"]["count"] >= 1
    assert any(s["id"] == "s_list" for s in data["sessions"])


def test_upload_requires_models_ready(client):
    # No model is warmed in tests, so uploads are rejected with a JSON 503.
    res = client.post("/api/sessions")
    assert res.status_code == 503
    assert res.get_json() == {"error": "loading_models"}


def test_settings_and_onboarding_shape(client):
    s = client.get("/api/settings").get_json()
    assert {"default_language", "languages", "models", "backup", "onboarded"} <= s.keys()
    o = client.get("/api/onboarding").get_json()
    assert {"sizes", "selected_size", "models", "backup"} <= o.keys()
