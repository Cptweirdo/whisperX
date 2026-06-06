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

    payload = server_mod._sessions.load_translation(sid, "es")
    assert [s["text"] for s in payload["segments"]] == [
        "[es] Hello there.", "[es] How are you?"]

    # JSON view
    j = client.get(f"/sessions/{sid}/translation/es?format=json").get_json()
    assert j["target_language"] == "es"
    # Rendered view returns HTML containing the translated text
    html = client.get(f"/sessions/{sid}/translation/es").get_data(as_text=True)
    assert "Hello" in html and "[es]" in html
    # srt artifact is downloadable
    srt = client.get(f"/sessions/{sid}/translation/es/download/srt")
    assert srt.status_code == 200

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
