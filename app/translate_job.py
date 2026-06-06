"""On-demand translation of a finished transcript into a target language.

Translation is network-bound (an external API call), so it runs on its own
small executor rather than the single-worker CPU :class:`~app.jobs.JobQueue`,
and won't block transcription. Each run is non-destructive: it reads the
session's *current* (possibly edited) segments and writes a per-language overlay
``transcript.translation.<lang>.json`` plus srt/vtt/txt artifacts; the original
transcript is never mutated. Status is recorded on the ``translations`` column
of the session row (durable, for reconnects) and published to SSE subscribers on
the ``translate:<session_id>`` channel.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from app import pipeline, secret_store
from app.translation import get_translator

logger = logging.getLogger(__name__)


def channel(session_id: str) -> str:
    """SSE channel name carrying a session's translation progress."""
    return f"translate:{session_id}"


def _event(lang: str, status: str, error: str | None = None) -> dict:
    event = {"lang": lang, "status": status}
    if error is not None:
        event["error"] = error
    return event


def run_translation(
    store,
    session_id: str,
    target_lang: str,
    *,
    service: str,
    api_key: str,
    broker=None,
) -> None:
    """Translate a session's transcript into ``target_lang`` and persist it."""

    def publish(event: dict) -> None:
        if broker is not None:
            broker.publish(channel(session_id), event)

    store.set_translation_status(session_id, target_lang, "running", service=service)
    publish(_event(target_lang, "running"))
    try:
        result = store.load_result(session_id) or {}
        segments = store.current_segments(session_id, result.get("segments", []))
        translator = get_translator(service, api_key)
        translated_text = translator.translate(
            [s.get("text", "") for s in segments], target_lang
        )

        out_segments = [
            {
                "start": s.get("start"),
                "end": s.get("end"),
                "speaker": s.get("speaker"),
                "text": text,
            }
            for s, text in zip(segments, translated_text)
        ]
        payload = {
            "version": 1,
            "target_language": target_lang,
            "service": service,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "segments": out_segments,
        }
        store.save_translation(session_id, target_lang, payload)
        _write_artifacts(store, session_id, target_lang, out_segments)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        logger.exception("Translation of %s -> %s failed", session_id, target_lang)
        store.set_translation_status(session_id, target_lang, "error", error=str(exc))
        publish(_event(target_lang, "error", str(exc)))
        return

    store.set_translation_status(session_id, target_lang, "done")
    publish(_event(target_lang, "done"))


def _write_artifacts(store, session_id: str, lang: str, segments: list) -> None:
    """Write srt/vtt/txt artifacts for the translation (json is the overlay)."""
    import os

    from whisperx.utils import get_writer

    output_dir = store.session_dir(session_id)
    # The writer derives the output name by stripping the last dotted suffix off
    # the basename we pass (os.path.splitext). The language tag itself looks like
    # that suffix, so add a throwaway ".x" for splitext to eat instead, leaving
    # the final name transcript.translation.<lang>.<fmt>.
    stem = os.path.join(output_dir, f"transcript.translation.{lang}.x")
    result = {"segments": segments, "language": lang}
    for fmt in ("srt", "vtt", "txt"):
        writer = get_writer(fmt, output_dir)
        writer(result, stem, pipeline.WRITER_OPTIONS)


class TranslationQueue:
    """Serialized background executor for translation jobs.

    Single worker so concurrent requests for the same (or different) sessions
    queue rather than hammering the translation API in parallel; it is separate
    from the transcription queue so the two never block each other.
    """

    def __init__(self, store, broker=None):
        self._store = store
        self._broker = broker
        self._executor = ThreadPoolExecutor(max_workers=1)

    def submit(self, session_id: str, target_lang: str, service: str) -> None:
        api_key = secret_store.resolve_google_api_key()
        self._executor.submit(
            run_translation,
            self._store,
            session_id,
            target_lang,
            service=service,
            api_key=api_key,
            broker=self._broker,
        )
