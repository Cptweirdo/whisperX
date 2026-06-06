"""HTTP surface for on-demand transcript translation.

Covers the endpoints added in ``app.server``:

    POST /sessions/<id>/translate                 -> queue a translation
    GET  /sessions/<id>/translation/<lang>        -> rendered / JSON result
    GET  /sessions/<id>/translation/<lang>/download/<fmt>
    POST /settings/translation-service            -> persist the provider pref

The pure pieces (the :class:`Translator` abstraction, the keyring storage, the
store overlay) are pinned in test_translation.py; here we assert the request
glue and the background runner's effect, with the translator and the API key
mocked so nothing hits the network.

Like the other server tests, ``app.server`` has import-time side effects (a
model warm-up thread, a SessionStore), so we point the data dir at a tmp dir and
neutralize model loading before importing it.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid

import pytest

pytest.importorskip("flask", reason="app dep; see app/requirements.txt")


@pytest.fixture(scope="module")
def server_mod():
    from app import pipeline
    pipeline.ModelManager.load_asr = lambda self, name: object()
    pipeline.ModelManager.ensure_diarize = lambda self: None

    tmp = tempfile.mkdtemp(prefix="translate-server-test-")
    prev = os.environ.get("WHISPERX_DATA_DIR")
    os.environ["WHISPERX_DATA_DIR"] = tmp
    try:
        from app import server
        yield server
    finally:
        if prev is None:
            os.environ.pop("WHISPERX_DATA_DIR", None)
        else:
            os.environ["WHISPERX_DATA_DIR"] = prev


def _done_session(server_mod):
    store = server_mod._sessions
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    os.makedirs(store.session_dir(sid), exist_ok=True)
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello there.", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "How are you?", "speaker": "SPEAKER_00"},
    ]
    store.create(sid, "rec.wav", "rec.wav", {}, model="tiny")
    with open(store.result_path(sid), "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f)
    store.mark_done(sid, language="en", diarized=True, model="tiny",
                    num_segments=2, duration=2.0)
    return sid


def _two_speaker_session(server_mod):
    """A finished session with two consecutive turns by different speakers."""
    store = server_mod._sessions
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    os.makedirs(store.session_dir(sid), exist_ok=True)
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello there.", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "How are you?", "speaker": "SPEAKER_01"},
    ]
    store.create(sid, "rec.wav", "rec.wav", {}, model="tiny")
    with open(store.result_path(sid), "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f)
    store.mark_done(sid, language="en", diarized=True, model="tiny",
                    num_segments=2, duration=2.0)
    return sid


class _FakeTranslator:
    def translate(self, texts, target_lang, source_lang=None):
        return [f"[{target_lang}] {t}" for t in texts]


def _enable(server_mod, monkeypatch):
    """Mock the API key + translator so translation runs offline."""
    monkeypatch.setattr(server_mod.secret_store, "resolve_google_api_key", lambda: "k")
    monkeypatch.setattr("app.translate_job.get_translator",
                        lambda service, api_key: _FakeTranslator())


def _translate(server_mod, client, sid, lang):
    client.post(f"/sessions/{sid}/translate", data={"target_language": lang})
    return _wait_done(server_mod._sessions, sid, lang)


def _turn_ids(html):
    """Ordered list of the `.turn` div indices in a rendered body."""
    return re.findall(r'class="turn" data-turn="(\d+)"', html)


def _wait_done(store, sid, lang, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = store.get_translations(sid).get(lang, {}).get("status")
        if st in ("done", "error"):
            return st
        time.sleep(0.02)
    return None


# --- guards ------------------------------------------------------------------

def test_translate_requires_key(server_mod, monkeypatch):
    monkeypatch.setattr(server_mod.secret_store, "resolve_google_api_key", lambda: None)
    sid = _done_session(server_mod)
    resp = server_mod.app.test_client().post(
        f"/sessions/{sid}/translate", data={"target_language": "es"})
    assert resp.status_code == 400


def test_translate_rejects_bad_language(server_mod, monkeypatch):
    monkeypatch.setattr(server_mod.secret_store, "resolve_google_api_key", lambda: "k")
    sid = _done_session(server_mod)
    resp = server_mod.app.test_client().post(
        f"/sessions/{sid}/translate", data={"target_language": "../etc"})
    assert resp.status_code == 400


def test_translate_404_for_unknown_session(server_mod, monkeypatch):
    monkeypatch.setattr(server_mod.secret_store, "resolve_google_api_key", lambda: "k")
    resp = server_mod.app.test_client().post(
        "/sessions/nope/translate", data={"target_language": "es"})
    assert resp.status_code == 404


# --- happy path (translator mocked) ------------------------------------------

def test_translate_writes_overlay_and_serves_it(server_mod, monkeypatch):
    monkeypatch.setattr(server_mod.secret_store, "resolve_google_api_key", lambda: "k")

    class _FakeTranslator:
        def translate(self, texts, target_lang, source_lang=None):
            return [f"[{target_lang}] {t}" for t in texts]

    # Patch the factory used by the background runner.
    monkeypatch.setattr("app.translate_job.get_translator",
                        lambda service, api_key: _FakeTranslator())

    sid = _done_session(server_mod)
    client = server_mod.app.test_client()
    resp = client.post(f"/sessions/{sid}/translate", data={"target_language": "es"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "running"

    status = _wait_done(server_mod._sessions, sid, "es")
    assert status == "done"

    # v2 overlay: text-only entries keyed by start time (J1).
    payload = server_mod._sessions.load_translation(sid, "es")
    assert payload["version"] == 2
    assert payload["entries"]["0.000"] == {"src": "Hello there.", "tr": "[es] Hello there."}
    assert payload["entries"]["1.000"]["tr"] == "[es] How are you?"
    assert "segments" not in payload
    # No static srt/vtt/txt written at translate time (exports are on-demand, J1).
    sess_dir = server_mod._sessions.session_dir(sid)
    assert not os.path.exists(os.path.join(sess_dir, "transcript.translation.es.srt"))

    # JSON view joins onto the current original
    j = client.get(f"/sessions/{sid}/translation/es?format=json").get_json()
    assert j["target_language"] == "es"
    assert [s["text"] for s in j["segments"]] == ["[es] Hello there.", "[es] How are you?"]
    # Rendered view returns HTML containing the translated text
    html = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert "[es]" in html
    # srt artifact is generated on demand
    srt = client.get(f"/sessions/{sid}/translation/es/download/srt")
    assert srt.status_code == 200
    assert "[es]" in srt.get_data(as_text=True)

    # original transcript untouched
    assert server_mod._sessions.load_result(sid)["segments"][0]["text"] == "Hello there."


def test_translation_failure_records_error(server_mod, monkeypatch):
    monkeypatch.setattr(server_mod.secret_store, "resolve_google_api_key", lambda: "k")

    def _boom(service, api_key):
        class _T:
            def translate(self, *a, **k):
                raise RuntimeError("api down")
        return _T()

    monkeypatch.setattr("app.translate_job.get_translator", _boom)
    sid = _done_session(server_mod)
    server_mod.app.test_client().post(
        f"/sessions/{sid}/translate", data={"target_language": "fr"})
    assert _wait_done(server_mod._sessions, sid, "fr") == "error"
    assert "api down" in server_mod._sessions.get_translations(sid)["fr"]["error"]


# --- settings ----------------------------------------------------------------

def test_set_translation_service(server_mod):
    client = server_mod.app.test_client()
    resp = client.post("/settings/translation-service",
                       data={"translation_service": "google"})
    assert resp.status_code == 200
    assert server_mod._sessions.get_setting("translation_service") == "google"


def test_set_translation_service_rejects_unknown(server_mod):
    resp = server_mod.app.test_client().post(
        "/settings/translation-service", data={"translation_service": "bogus"})
    assert resp.status_code == 400


# --- Group 1: speaker change / add propagation -------------------------------

def test_reassign_propagates_to_all_languages(server_mod, monkeypatch):
    """S1/S6/S12 + E7: enroll a new speaker on the original; the new label shows in
    every translation, the translated text stays fresh, and exports reflect the key."""
    _enable(server_mod, monkeypatch)
    sid = _done_session(server_mod)          # one turn, SPEAKER_00
    client = server_mod.app.test_client()
    assert _translate(server_mod, client, sid, "es") == "done"
    assert _translate(server_mod, client, sid, "fr") == "done"

    # Reassign turn 0 to a brand-new enrolled speaker.
    r = client.post(f"/sessions/{sid}/turns/0/speaker", data={"name": "Bob"})
    assert r.status_code == 200
    assert "Bob" in r.get_data(as_text=True)         # original re-render shows it

    for lang in ("es", "fr"):
        html = client.get(f"/sessions/{sid}/translation/{lang}").get_data(as_text=True)
        assert "Bob" in html                          # S1: new label everywhere
        assert f"[{lang}]" in html                     # S12: text still translated
        assert "seg--untranslated" not in html         # S12: reassign != stale
    # E7: srt export carries the newly minted speaker key prefix.
    srt = client.get(f"/sessions/{sid}/translation/es/download/srt").get_data(as_text=True)
    assert "[SPEAKER_01]:" in srt


def test_reassign_existing_key_merges_turns_consistently(server_mod, monkeypatch):
    """S3/S4: a translation view's turn indices match the original; reassigning a turn
    to its neighbour's speaker merges turns identically in both views."""
    _enable(server_mod, monkeypatch)
    sid = _two_speaker_session(server_mod)
    client = server_mod.app.test_client()
    assert _translate(server_mod, client, sid, "es") == "done"

    page = client.get(f"/sessions/{sid}/view").get_data(as_text=True)
    tr = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert _turn_ids(page) == _turn_ids(tr) == ["0", "1"]   # S3 parity, 2 turns

    # Merge: reassign turn 1 (SPEAKER_01) onto SPEAKER_00.
    r = client.post(f"/sessions/{sid}/turns/1/speaker", data={"speaker": "SPEAKER_00"})
    assert _turn_ids(r.get_data(as_text=True)) == ["0"]      # original merged to one turn
    tr2 = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert _turn_ids(tr2) == ["0"]                            # S4: translation merged too


def test_reassign_to_same_speaker_is_noop(server_mod, monkeypatch):
    """S5: reassigning to the current speaker changes nothing."""
    _enable(server_mod, monkeypatch)
    sid = _done_session(server_mod)
    client = server_mod.app.test_client()
    assert _translate(server_mod, client, sid, "es") == "done"
    before = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    client.post(f"/sessions/{sid}/turns/0/speaker", data={"speaker": "SPEAKER_00"})
    after = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert before == after


def test_rename_propagates_to_translations(server_mod, monkeypatch):
    """S9: a session-global speaker rename shows in every translation view."""
    _enable(server_mod, monkeypatch)
    sid = _done_session(server_mod)
    client = server_mod.app.test_client()
    assert _translate(server_mod, client, sid, "es") == "done"
    server_mod._sessions.set_speaker_name(sid, "SPEAKER_00", "Alice")
    html = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert "Alice" in html


def test_undiarized_session_translates(server_mod, monkeypatch):
    """S10: a session with no speakers still translates, views and downloads."""
    _enable(server_mod, monkeypatch)
    store = server_mod._sessions
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    os.makedirs(store.session_dir(sid), exist_ok=True)
    store.create(sid, "rec.wav", "rec.wav", {}, model="tiny")
    with open(store.result_path(sid), "w", encoding="utf-8") as f:
        json.dump({"segments": [{"start": 0.0, "end": 1.0, "text": "Hello there."}]}, f)
    store.mark_done(sid, language="en", diarized=False, model="tiny",
                    num_segments=1, duration=1.0)
    client = server_mod.app.test_client()
    assert _translate(server_mod, client, sid, "es") == "done"
    html = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert "[es]" in html and "turn__swap" not in html
    assert client.get(f"/sessions/{sid}/translation/es/download/txt").status_code == 200


def test_translate_after_reassign_inherits(server_mod, monkeypatch):
    """S11: a translation created *after* a reassignment inherits the current speaker."""
    _enable(server_mod, monkeypatch)
    sid = _done_session(server_mod)
    client = server_mod.app.test_client()
    client.post(f"/sessions/{sid}/turns/0/speaker", data={"name": "Bob"})
    assert _translate(server_mod, client, sid, "es") == "done"
    html = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert "Bob" in html and "seg--untranslated" not in html


# --- Group 2: segment / text edits (staleness) -------------------------------

def test_text_edit_marks_segment_stale(server_mod, monkeypatch):
    """E1/E7: editing a turn's text makes only that segment stale (original text +
    marker); other turns stay translated, and the export shows the fallback text."""
    _enable(server_mod, monkeypatch)
    sid = _two_speaker_session(server_mod)
    client = server_mod.app.test_client()
    assert _translate(server_mod, client, sid, "es") == "done"

    r = client.post(f"/sessions/{sid}/turns/0", data={"text": "Goodbye"})
    assert r.status_code == 200
    html = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert "seg--untranslated" in html and "Goodbye" in html       # turn 0 stale
    assert "[es] How are you?" in html                              # turn 1 still fresh
    # E7: export carries the original-text fallback for the stale segment.
    srt = client.get(f"/sessions/{sid}/translation/es/download/srt").get_data(as_text=True)
    assert "Goodbye" in srt


def test_undo_restores_fresh_translation(server_mod, monkeypatch):
    """E2: undoing the text edit re-matches the source, so the translation is fresh again."""
    _enable(server_mod, monkeypatch)
    sid = _two_speaker_session(server_mod)
    client = server_mod.app.test_client()
    assert _translate(server_mod, client, sid, "es") == "done"
    client.post(f"/sessions/{sid}/turns/0", data={"text": "Goodbye"})
    client.post(f"/sessions/{sid}/undo")
    html = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert "seg--untranslated" not in html and "[es] Hello there." in html


def test_delete_turn_drops_it_from_translation(server_mod, monkeypatch):
    """E3: deleting a turn (empty edit) removes it from the translation view."""
    _enable(server_mod, monkeypatch)
    sid = _two_speaker_session(server_mod)
    client = server_mod.app.test_client()
    assert _translate(server_mod, client, sid, "es") == "done"
    client.post(f"/sessions/{sid}/turns/0", data={"text": ""})     # delete turn 0
    html = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert _turn_ids(html) == ["0"]                                 # only one turn left
    assert "Hello" not in html and "[es] How are you?" in html


def test_stale_and_reassigned_turn(server_mod, monkeypatch):
    """E4: a turn edited *and* reassigned shows the original text + marker AND the new
    speaker simultaneously."""
    _enable(server_mod, monkeypatch)
    sid = _two_speaker_session(server_mod)
    client = server_mod.app.test_client()
    assert _translate(server_mod, client, sid, "es") == "done"
    client.post(f"/sessions/{sid}/turns/0", data={"text": "Goodbye"})
    client.post(f"/sessions/{sid}/turns/0/speaker", data={"name": "Zed"})
    html = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert "seg--untranslated" in html and "Goodbye" in html and "Zed" in html


def test_mixed_fresh_stale_within_one_turn(server_mod, monkeypatch):
    """E5: a same-speaker turn spanning two segments where only one segment's source
    changed marks only that span — the other stays translated."""
    _enable(server_mod, monkeypatch)
    sid = _done_session(server_mod)            # two SPEAKER_00 segments -> one turn
    client = server_mod.app.test_client()
    assert _translate(server_mod, client, sid, "es") == "done"

    # Diverge only the second segment's source text directly in the stored result
    # (simulating drift within a turn without collapsing it via the turn editor).
    store = server_mod._sessions
    result = store.load_result(sid)
    result["segments"][1]["text"] = "CHANGED"
    with open(store.result_path(sid), "w", encoding="utf-8") as f:
        json.dump(result, f)

    html = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert _turn_ids(html) == ["0"]                       # still one turn
    assert "[es] Hello there." in html                     # seg 0 fresh
    assert "seg--untranslated" in html and "CHANGED" in html  # seg 1 stale
    assert "[es] How are you?" not in html


def test_retranslate_clears_staleness(server_mod, monkeypatch):
    """E6: re-running translate after an edit rebuilds entries from current source."""
    _enable(server_mod, monkeypatch)
    sid = _two_speaker_session(server_mod)
    client = server_mod.app.test_client()
    assert _translate(server_mod, client, sid, "es") == "done"
    client.post(f"/sessions/{sid}/turns/0", data={"text": "Goodbye"})
    assert "seg--untranslated" in client.get(
        f"/sessions/{sid}/translation/es").get_data(as_text=True)

    # Re-translate. Status may still read the previous "done" momentarily, so wait on
    # the overlay actually being rebuilt from the edited source text.
    client.post(f"/sessions/{sid}/translate", data={"target_language": "es"})
    deadline = time.time() + 5.0
    while time.time() < deadline:
        ov = server_mod._sessions.load_translation(sid, "es") or {}
        if ov.get("entries", {}).get("0.000", {}).get("src") == "Goodbye":
            break
        time.sleep(0.02)
    html = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert "seg--untranslated" not in html and "[es] Goodbye" in html
