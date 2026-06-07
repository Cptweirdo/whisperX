"""Backend translation support: service abstraction, secret storage, persistence.

Translation is an on-demand, non-destructive operation on a finished transcript.
These tests pin the pluggable :class:`Translator` abstraction and its Google
backend (batching + order preservation, HTTP mocked), the keyring-backed API key
storage with env-var precedence, and the per-language overlay persistence +
status tracking on the session store (including the schema migration).
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from app import secret_store
from app.store import SessionStore
from app.translation import DEFAULT_SERVICE, SERVICES, get_translator
from app.translation.base import TranslationError
from app.translation.google import GoogleTranslator, _batch


# --- service abstraction -----------------------------------------------------

def test_registry_has_google_default():
    assert DEFAULT_SERVICE == "google"
    assert "google" in SERVICES


def test_get_translator_unknown_raises():
    with pytest.raises(TranslationError):
        get_translator("nope", "key")


def test_get_translator_returns_google():
    t = get_translator("google", "AIzaKEY")
    assert isinstance(t, GoogleTranslator)
    assert t.name == "google" and t.label == "Google Translate"


def test_google_requires_key():
    with pytest.raises(TranslationError):
        GoogleTranslator("")


# --- batching ----------------------------------------------------------------

def test_batch_splits_on_segment_cap():
    texts = [f"s{i}" for i in range(250)]
    batches = _batch(texts)
    assert all(len(b) <= 100 for b in batches)
    assert sum(len(b) for b in batches) == 250
    # order preserved when flattened
    assert [t for b in batches for t in b] == texts


def test_batch_splits_on_char_cap():
    big = "x" * 20_000
    batches = _batch([big, big])  # 40k chars > 25k cap -> two batches
    assert len(batches) == 2


# --- google translate (HTTP mocked) ------------------------------------------

class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_google_translate_preserves_order(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=0):
        # Decode the q= fields from the POST body to echo back translations.
        from urllib.parse import parse_qs
        body = parse_qs(req.data.decode())
        qs = body["q"]
        calls.append(qs)
        return _FakeResp(
            {"data": {"translations": [{"translatedText": q.upper()} for q in qs]}}
        )

    monkeypatch.setattr("app.translation.google.urlopen", fake_urlopen)
    out = GoogleTranslator("k").translate(["a", "b", "c"], "es")
    assert out == ["A", "B", "C"]
    assert calls == [["a", "b", "c"]]


def test_google_translate_empty_is_noop(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("should not hit the network for empty input")

    monkeypatch.setattr("app.translation.google.urlopen", boom)
    assert GoogleTranslator("k").translate([], "es") == []


def test_google_translate_count_mismatch_raises(monkeypatch):
    def fake_urlopen(req, timeout=0):
        return _FakeResp({"data": {"translations": [{"translatedText": "only one"}]}})

    monkeypatch.setattr("app.translation.google.urlopen", fake_urlopen)
    with pytest.raises(TranslationError):
        GoogleTranslator("k").translate(["a", "b"], "es")


# --- secret store: google API key --------------------------------------------

class _FakeKeyring:
    """Minimal in-memory keyring stand-in."""

    def __init__(self):
        self.store = {}

    def set_password(self, service, key, value):
        self.store[(service, key)] = value

    def get_password(self, service, key):
        return self.store.get((service, key))

    def delete_password(self, service, key):
        self.store.pop((service, key), None)


@pytest.fixture
def fake_keyring(monkeypatch):
    fk = _FakeKeyring()
    monkeypatch.setattr(secret_store, "keyring_available", lambda: True)
    import sys
    import types

    mod = types.ModuleType("keyring")
    mod.set_password = fk.set_password
    mod.get_password = fk.get_password
    mod.delete_password = fk.delete_password
    monkeypatch.setitem(sys.modules, "keyring", mod)
    monkeypatch.delenv("GOOGLE_TRANSLATE_API_KEY", raising=False)
    return fk


def test_google_key_roundtrip(fake_keyring):
    assert secret_store.get_stored_google_api_key() is None
    secret_store.set_google_api_key("AIzaSECRET")
    assert secret_store.get_stored_google_api_key() == "AIzaSECRET"
    assert secret_store.resolve_google_api_key() == "AIzaSECRET"
    secret_store.delete_google_api_key()
    assert secret_store.get_stored_google_api_key() is None


def test_google_key_env_takes_precedence(fake_keyring, monkeypatch):
    secret_store.set_google_api_key("from_keyring")
    monkeypatch.setenv("GOOGLE_TRANSLATE_API_KEY", "from_env")
    assert secret_store.resolve_google_api_key() == "from_env"


def test_set_google_key_empty_raises(fake_keyring):
    with pytest.raises(ValueError):
        secret_store.set_google_api_key("   ")


# --- store: migration + per-language overlay ---------------------------------

def _old_db(path):
    """Create a sessions DB lacking the translations (and stage) columns."""
    con = sqlite3.connect(os.path.join(path, "sessions.db"))
    con.executescript(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, filename TEXT, "
        "audio_filename TEXT, status TEXT NOT NULL, error TEXT, options TEXT, "
        "language TEXT, diarized INTEGER, model TEXT, num_segments INTEGER, "
        "duration REAL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
    )
    con.commit()
    con.close()


def test_migration_adds_translations_column(tmp_path):
    _old_db(str(tmp_path))
    store = SessionStore(str(tmp_path))
    # Opening the legacy DB migrated it; the translations column now works
    # end-to-end (write + read), which is only possible if it was added. Asserted
    # through the public API so it holds for either DB backend (Python or C++).
    store.create("m1", "a.wav", "audio.wav", {})
    store.set_translation_status("m1", "es", "done", service="deepl")
    assert store.get_translations("m1")["es"]["status"] == "done"
    assert store.get("m1")["translations"]["es"]["service"] == "deepl"


def _store_with_session(tmp_path):
    store = SessionStore(str(tmp_path))
    sid = "sess1"
    os.makedirs(store.session_dir(sid), exist_ok=True)
    store.create(sid, "a.wav", "audio.wav", {"language": "en"})
    with open(store.result_path(sid), "w", encoding="utf-8") as f:
        json.dump({"segments": [{"start": 0, "end": 1, "text": "hi"}]}, f)
    return store, sid


def test_translation_overlay_roundtrip(tmp_path):
    store, sid = _store_with_session(tmp_path)
    payload = {"version": 1, "target_language": "es",
               "segments": [{"start": 0, "end": 1, "text": "hola"}]}
    store.save_translation(sid, "es", payload)
    assert store.load_translation(sid, "es") == payload
    assert store.load_translation(sid, "fr") is None


def test_translation_status_transitions(tmp_path):
    store, sid = _store_with_session(tmp_path)
    store.set_translation_status(sid, "es", "running", service="google")
    assert store.get_translations(sid) == {"es": {"status": "running", "service": "google"}}
    store.set_translation_status(sid, "es", "done")
    assert store.get_translations(sid)["es"]["status"] == "done"
    # error carries a message; clearing to a non-error status drops it
    store.set_translation_status(sid, "fr", "error", service="google", error="boom")
    assert store.get_translations(sid)["fr"]["error"] == "boom"
    store.set_translation_status(sid, "fr", "done")
    assert "error" not in store.get_translations(sid)["fr"]


def test_translation_original_result_untouched(tmp_path):
    store, sid = _store_with_session(tmp_path)
    store.save_translation(sid, "es",
                           {"segments": [{"start": 0, "end": 1, "text": "hola"}]})
    assert store.load_result(sid)["segments"] == [{"start": 0, "end": 1, "text": "hi"}]
