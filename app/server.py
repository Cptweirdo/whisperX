"""Flask JSON API for WhisperX, serving the Svelte SPA (app/web -> static/spa).

Run:  python -m app.server     (or  flask --app app.server run)
Config: put HF_TOKEN (and optional WHISPERX_* overrides) in app/.env.
        See app/.env.example. In Docker the same file is injected via
        docker-compose `env_file`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path


def _load_dotenv(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file. Real env vars take precedence."""
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value[:1] not in ('"', "'"):  # strip unquoted inline comment
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        value = value.strip('"').strip("'")
        if key and key not in os.environ:  # don't override an already-set var
            os.environ[key] = value


from app import paths  # noqa: E402 - stdlib-only; safe to import before the .env load

# Config precedence (highest first), since _load_dotenv never overrides an
# already-set key:
#   1. real environment variables
#   2. app/.env            — dev/back-compat; may itself set WHISPERX_DATA_DIR
#   3. <data_dir>/.env     — user/per-machine overrides for the packaged app
#   4. app/defaults.env    — ship-with-the-app defaults baked into the bundle by
#                            the macOS packager (e.g. OAuth client id/secret +
#                            default backup backend). Loaded LAST = lowest priority,
#                            so any of the above overrides it. Absent in dev/source.
_load_dotenv(Path(__file__).with_name(".env"))
_load_dotenv(paths.data_dir() / ".env")
_load_dotenv(Path(__file__).with_name("defaults.env"))

from datetime import datetime, timezone  # noqa: E402

from flask import (  # noqa: E402 - load .env first
    Flask,
    Response,
    abort,
    jsonify,
    request,
    send_file,
)
from werkzeug.utils import secure_filename  # noqa: E402

from app import backup as backup_pkg  # noqa: E402
from app import diarize_model  # noqa: E402
from app import pipeline  # noqa: E402
from app import secret_store  # noqa: E402
from app import translate_job  # noqa: E402
from app import translation_overlay  # noqa: E402
from app.translation import DEFAULT_SERVICE, SERVICES  # noqa: E402
from app.sse import Broker, sse_response  # noqa: E402
from app.jobs import JobQueue  # noqa: E402
from app.edits import distinct_speakers, group_turns, next_speaker_key  # noqa: E402
from app.render import render_markdown, resolve_label  # noqa: E402
from app.store import SessionStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

# Sized for ~4h of WAV: 48 kHz / 16-bit / stereo ≈ 0.7 GB/h → 4h ≈ 2.8 GB, and
# 24-bit stereo ≈ 4.2 GB. Default 5 GB covers those with headroom. Werkzeug
# spools the upload to a temp file (not RAM), so a large cap is safe on the dev
# server. Override via WHISPERX_MAX_UPLOAD_MB.
MAX_UPLOAD_MB = int(os.environ.get("WHISPERX_MAX_UPLOAD_MB", "5000"))
# Hard ceiling on input *duration* (hours). The native CPU sherpa pipeline holds the
# whole decoded waveform resident and runs the full chain on CPU, so multi-hour inputs
# mean multi-hour runtime and OOM risk — reject upfront with a readable error instead
# of failing mid-job. 0 disables. The advisory log warning (pipeline.LONG_AUDIO_WARN_S)
# covers the band below this.
MAX_AUDIO_HOURS = float(os.environ.get("WHISPERX_MAX_AUDIO_HOURS", "4"))
# paths.data_dir() already honors WHISPERX_DATA_DIR (see app/paths.py).
DATA_DIR = str(paths.data_dir())


def _probe_duration_seconds(path: str) -> float | None:
    """Audio duration in seconds from the container header — in-process via the
    native core (libav*, no decode, no subprocess). None if the core isn't built or
    the container has no usable duration, in which case callers skip the check."""
    try:
        import whisperx_core
    except ImportError:
        return None
    if not hasattr(whisperx_core, "probe_duration"):
        return None
    try:
        dur = whisperx_core.probe_duration(path)
    except Exception:
        return None
    return dur if dur and dur > 0 else None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(413)
def _too_large(_e):
    # Flask aborts the upload before the route runs; return JSON the SPA surfaces.
    return jsonify({"error": "File too large.",
                    "max_mb": MAX_UPLOAD_MB}), 413


# --- View formatting (dashboard cards + transcript header) -------------------
_STATUS_META = {
    "done": ("Done", "chip--ok", True),
    "running": ("Processing", "chip--run", False),
    "queued": ("Queued", "chip--run", False),
    "error": ("Error", "chip--err", False),
}


def _fmt_duration(sec) -> str:
    sec = int(sec or 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _fmt_clock(total_sec) -> str:
    total_sec = int(total_sec or 0)
    h, rem = divmod(total_sec, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %Y · %H:%M")
    except ValueError:
        return iso


def _card(row: dict) -> dict:
    label, chip_class, viewable = _STATUS_META.get(row["status"], ("Unprocessed", "", False))
    if row["status"] == "done":
        sub = f"{row.get('num_segments') or 0} segments transcribed"
        if row.get("language"):
            sub += f" · language {row['language']}"
        sub += "."
    elif row["status"] == "error":
        sub = f"Failed: {row.get('error') or 'unknown error'}"
    else:
        sub = "Awaiting transcription on CPU."
    return {
        "id": row["id"],
        "name": row.get("filename") or "Untitled recording",
        "chip_label": label,
        "chip_class": chip_class,
        "viewable": viewable,
        "dur": _fmt_duration(row.get("duration")),
        "date": _fmt_date(row.get("created_at")),
        "sub": sub,
        "model": row.get("model"),
        "language": row.get("language"),
        "diarized": bool(row.get("diarized")),
        "num_segments": row.get("num_segments") or 0,
        "status": row["status"],
        "stage": row.get("stage"),
        "error": row.get("error"),
        "translations": row.get("translations") or {},
    }


def _summary(rows: list[dict]) -> dict:
    done = [r for r in rows if r["status"] == "done"]
    total_audio = sum(r.get("duration") or 0 for r in rows)
    transcribed = sum(r.get("duration") or 0 for r in done)
    pct = round(len(done) / len(rows) * 100) if rows else 0
    return {
        "count": len(rows),
        "transcribed": _fmt_clock(transcribed),
        "total_audio": _fmt_clock(total_audio),
        "pct": pct,
    }

# --- Models: a manager caches multiple Whisper checkpoints; the active one is
#     warmed in a background thread so the server can boot before it is ready. ---
_sessions = SessionStore(DATA_DIR)

# SSE pub/sub. Per-session keys carry job progress; one reserved key carries
# global model-load state so the dashboard can react when the background warm
# finishes (flip the "loading models" toast to a success state, refresh dropdowns).
_broker = Broker()
MODELS_CHANNEL = "__models__"
BACKUP_CHANNEL = "__backup__"                 # one-shot OAuth consent flow
BACKUP_STATUS_CHANNEL = "__backup_status__"   # persistent sync-status stream


def _models_event(status: dict) -> dict:
    """SSE payload describing global model-load state.

    Drives the dashboard "loading models" → "ready" toast and refreshes the
    loaded/loading/failed badges on every model <sl-select>. ``models_ready``
    reflects the *active* model (the one a default upload would use)."""
    active = status.get("active")
    active_meta = next((m for m in status["models"] if m["name"] == active), None)
    return {
        "type": "models",
        "models_ready": bool(active_meta and active_meta["loaded"]),
        "active": active,
        "bundle_error": active_meta.get("error") if active_meta else None,
        "diarize_available": status.get("diarize_available"),
        "diarize_error": status.get("diarize_error"),
        "models": status.get("models", []),
    }


def _publish_models(status: dict) -> None:
    """ModelManager on_change hook: broadcast model state to SSE listeners."""
    _broker.publish(MODELS_CHANNEL, _models_event(status))


# Seed the active model + compute device from persisted settings (switches survive
# restart); fall back to the WHISPERX_* defaults if a stored value is no longer valid
# (an unavailable cuda device is downgraded to cpu inside ModelManager).
_manager = pipeline.ModelManager(
    active=_sessions.get_setting("active_model", pipeline.DEFAULT_MODEL),
    device=_sessions.get_setting("device", pipeline.DEFAULT_DEVICE),
    on_change=_publish_models,
)


def _warm_models() -> None:
    """Warm the active Whisper model + the shared diarizer in the background."""
    try:
        _manager.load_asr(_manager.active)
        diarize = _manager.ensure_diarize()
        logger.info("Models ready (active=%s, diarization %s).",
                    _manager.active, "ON" if diarize else "OFF")
    except Exception:  # noqa: BLE001 - the error is recorded in manager.status()
        logger.exception("Active model load failed")


def run_session(session_id: str, cancel_event=None) -> None:
    """Execute the pipeline for a session and persist results + metadata."""
    row = _sessions.get(session_id)
    if row is None:
        raise RuntimeError(f"session {session_id} disappeared")
    opts = row.get("options") or {}
    model = row.get("model") or _manager.active  # the model chosen at upload time
    audio_path = _sessions.audio_path(session_id)
    result = pipeline.run_job(
        _manager.bundle_for(model),  # loads on demand + caches; shares diarizer/align
        audio_path,
        _sessions.session_dir(session_id),
        artifact_basename="transcript",
        language=opts.get("language"),
        min_speakers=opts.get("min_speakers"),
        max_speakers=opts.get("max_speakers"),
        progress=lambda s: _on_stage(session_id, s),
        on_duration=lambda d: _sessions.mark_duration(session_id, d),
        cancel_event=cancel_event,
    )
    _sessions.mark_done(
        session_id,
        language=result.get("language"),
        diarized=bool(result.get("diarized")),
        model=model,
        num_segments=result.get("num_segments", len(result.get("segments", []))),
        duration=result.get("duration", 0.0),
    )


def _stage_event(stage: str, duration) -> dict:
    """SSE payload for a stage, with an ETA (seconds) when one can be estimated."""
    event = {"stage": stage}
    eta = pipeline.eta_seconds(stage, duration)
    if eta is not None:
        event["eta"] = round(eta)
    return event


def _on_stage(session_id: str, stage: str) -> None:
    """Persist the live stage (durable, for reconnects) and push it to SSE clients."""
    _sessions.mark_stage(session_id, stage)
    row = _sessions.get(session_id) or {}
    _broker.publish(session_id, _stage_event(stage, row.get("duration")))


_queue = JobQueue(_sessions, run_session, broker=_broker)

# Translation runs on its own (network-bound) executor so it never blocks the
# single-worker transcription queue. State is durable on the session row.
_translate_queue = translate_job.TranslationQueue(_sessions, broker=_broker)

_requeue_ids = _sessions.reconcile_startup()
if _requeue_ids:
    logger.info("Requeuing %d session(s) from before restart", len(_requeue_ids))
    for _sid in _requeue_ids:
        _queue.submit(_sid)

# --- Cloud backup: mirror the data dir (DB + artifacts) to a swappable backend.
#     Disabled unless WHISPERX_BACKUP_BACKEND is set. Snapshot-under-lock keeps
#     the DB copy consistent; periodic push runs only when local state changed. ---
_backup = backup_pkg.build_service(_sessions, on_change=lambda: _publish_backup())
if _backup.is_linked():
    _backup.start_periodic()


def models_ready() -> bool:
    return _manager.is_loaded(_manager.active)


@app.get("/healthz")
def healthz():
    """Liveness/readiness probe the launcher (or Tauri shell) polls before loading
    the UI: 200 as soon as Flask is serving; ``models_ready`` flips true once the
    active model has warmed."""
    return jsonify(status="ok", models_ready=models_ready())


# --- Backup helpers (thin: all logic lives in BackupService) -------------------
# Shared by the /api/backup/* routes and the backup SSE streams; build the JSON
# the SPA renders the Settings "Backup & Restore" card + onboarding step from.
_PROVIDER_LABELS = {"gdrive": "Google Drive", "local": "Local folder"}


def _human_size(n: int) -> str:
    size = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _human_ago(iso: str | None) -> str | None:
    """A loose "5m ago" / "2h ago" for the last-backup time; date for older."""
    if not iso:
        return None
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    secs = (datetime.now(timezone.utc) - when).total_seconds()
    if secs < 45:
        return "just now"
    if secs < 3600:
        return f"{round(secs / 60)}m ago"
    if secs < 86400:
        return f"{round(secs / 3600)}h ago"
    return when.astimezone().strftime("%b %-d, %H:%M")


def _remote_json(remote) -> dict | None:
    """JSON view of a RemoteState (restore/conflict prompts)."""
    if not remote or not getattr(remote, "exists", False):
        return None
    return {
        "exists": True,
        "entries": remote.entries,
        "total_size": remote.total_size,
        "size_human": _human_size(remote.total_size),
        "created_at": remote.created_at or None,
    }


def _backup_json(remote=None, notice: str | None = None, notice_ok: bool = True) -> dict:
    """JSON the SPA renders the backup card from (mirrors _backup_ctx)."""
    status = _backup.status()
    backend = status.get("backend")
    return {
        **status,
        "provider_label": _PROVIDER_LABELS.get(backend, backend or "Cloud backup"),
        "last_human": _human_ago(status.get("last_backup_at")),
        "folder": backup_pkg.gdrive_folder() if backend == "gdrive" else None,
        "remote": _remote_json(remote),
        "notice": notice,
        "notice_ok": notice_ok,
    }


# --- Live sync status (persistent SSE; mirrors the model-load stream) ----------
# Every BackupService state transition fires on_change -> _publish_backup, which
# re-renders the Settings card and pushes it to the persistent BACKUP_STATUS_CHANNEL
# stream. The browser swaps #backup-card in place, so auto/periodic/manual pushes,
# failures, and "last backup at X" all show live without a reload.
def _backup_status_event() -> dict:
    """Backup sync state for the persistent status stream (JSON; the SPA renders
    the card). In the CONFLICT state it re-probes the remote so the Load/Start-fresh
    prompt is preserved rather than overwritten by a status push."""
    state = _backup.status().get("state")
    remote = None
    if state == "conflict":
        try:
            rs = _backup.bootstrap()
            remote = rs if rs.exists else None
        except Exception:  # noqa: BLE001 - fall back to the plain card
            remote = None
    return {"type": "backup", "state": state, "status": _backup_json(remote=remote)}


def _publish_backup() -> None:
    """BackupService on_change hook: broadcast the re-rendered card to listeners."""
    try:
        _broker.publish(BACKUP_STATUS_CHANNEL, _backup_status_event())
    except Exception:  # noqa: BLE001 - a render hiccup must not break a backup op
        logger.exception("backup status publish failed")


def _run_backup_async() -> None:
    """Kick a push on a background thread so the request returns immediately and the
    UI shows live "Syncing…" -> "Up to date" via the status stream (on_change)."""
    def run():
        try:
            _backup.backup_now()
        except Exception:  # noqa: BLE001 - state machine records ERROR; surfaced via SSE
            logger.exception("manual backup failed")
    threading.Thread(target=run, name="backup-manual", daemon=True).start()


@app.get("/backup/status/events")
def backup_status_events():
    """Persistent Server-Sent Events stream of backup sync state.

    Emits the current card on connect (correct for late/reconnecting clients), then
    a freshly-rendered card on every state transition. Long-lived — backup state has
    no terminal, so the browser keeps it open until navigation."""
    return sse_response(_broker, BACKUP_STATUS_CHANNEL, initial=_backup_status_event)


# --- Non-blocking OAuth consent ------------------------------------------------
# The Google consent flow runs a loopback HTTP server and blocks until the user
# finishes in the browser, which can take a while. Rather than hold a Flask
# worker (and the originating fetch) open for the whole consent, we run it on a
# background thread and notify the page over SSE (BACKUP_CHANNEL) when it lands.
# The connect routes return a "connecting…" fragment immediately; the page opens
# an EventSource and swaps in the final card on the terminal event.
_backup_link_lock = threading.Lock()
_backup_link = {"active": False, "result": None}  # result: last terminal SSE event


def _start_backup_link() -> None:
    """Begin the OAuth consent flow in the background (idempotent while one runs).
    The terminal result lands on BACKUP_CHANNEL as JSON the SPA renders."""
    with _backup_link_lock:
        if _backup_link["active"]:
            return
        _backup_link["active"] = True
        _backup_link["result"] = None
    threading.Thread(target=_run_backup_link, daemon=True).start()


def _run_backup_link() -> None:
    from app.backup import LinkOutcome, oauth
    after = None  # "seed" | "restore" — a background op to kick AFTER publishing
    try:
        oauth.link_interactive()
        if _backup.interval and _backup.is_linked():
            _backup.start_periodic()
        a = _backup.assess_link()
        # Only a real conflict (remote has data AND local differs) prompts the user;
        # everything else acts automatically (see LinkOutcome). The adopt/overwrite
        # choice only renders when we pass a truthy `remote`.
        if a.outcome == LinkOutcome.FRESH:
            after = "seed"       # empty remote: push the first backup
        elif a.outcome == LinkOutcome.REMOTE_ONLY:
            after = "restore"    # empty local: pull the existing backup down
        remote = a.remote if a.outcome == LinkOutcome.DIVERGED else None
        event = {"status": "linked", "backup": _backup_json(remote=remote)}
    except Exception as exc:  # noqa: BLE001 - report to the page over SSE
        event = {"status": "error", "message": str(exc),
                 "backup": _backup_json(notice=str(exc), notice_ok=False)}
    with _backup_link_lock:
        _backup_link["active"] = False
        _backup_link["result"] = event
    _broker.publish(BACKUP_CHANNEL, event)
    # Kick the auto op AFTER publishing "connected" so the UI updates first.
    if after == "seed":
        _seed_initial_backup()
    elif after == "restore":
        _seed_initial_restore()


def _seed_initial_backup() -> None:
    """Push the first backup to a freshly-linked, empty remote — in a background
    thread so neither the link flow nor a request blocks on the upload. (Probing a
    fresh remote on link creates the folder but uploads nothing; without this the
    first push would wait for the periodic loop, up to one interval later.)"""
    def run():
        try:
            result = _backup.backup_now()
            logger.info("initial backup after link: uploaded=%d skipped=%d",
                        result.uploaded, result.skipped)
        except Exception:  # noqa: BLE001 - periodic loop will retry; surface in logs
            logger.exception("initial backup after link failed")
    threading.Thread(target=run, name="backup-initial", daemon=True).start()


def _seed_initial_restore() -> None:
    """Auto-pull a backup onto a freshly-linked, empty device (background thread).

    Only invoked when local has zero sessions (LinkOutcome.REMOTE_ONLY), so there is
    nothing local to lose. Runs off-thread so the connected card flips in first."""
    def run():
        try:
            n = _backup.adopt_remote()
            logger.info("auto-restore after link: restored=%d", n)
        except Exception:  # noqa: BLE001 - surface in logs; user can retry from Settings
            logger.exception("auto-restore after link failed")
    threading.Thread(target=run, name="backup-restore", daemon=True).start()


def _apply_backup_folder(name: str | None) -> None:
    """Persist a user-chosen Drive folder and re-target the live backend.

    Called on connect, before the consent flow runs, so the background bootstrap
    reads the right folder. Blank = keep whatever's stored (default if none).
    """
    name = (name or "").strip()
    if not name:
        return
    try:
        backup_pkg.set_gdrive_folder(name)
    except Exception:  # noqa: BLE001 - keyring missing: still target it this run;
        pass           #   the link itself will surface the real keyring error
    if _backup.backend is not None:
        _backup.backend.set_folder(backup_pkg.gdrive_folder())


@app.get("/backup/events")
def backup_events():
    """Server-Sent Events stream for the OAuth consent flow.

    Carries one terminal event (``linked`` / ``error``) with the rendered card
    HTML. A late subscriber (the flow finished before its EventSource opened)
    gets the stored result immediately; otherwise it relays the live event.
    """
    def pending():
        with _backup_link_lock:
            return _backup_link["result"]

    return sse_response(
        _broker,
        BACKUP_CHANNEL,
        pending=pending,
        terminal=lambda e: e.get("status") in ("linked", "error"),
    )


# --- Onboarding (first-run setup) -------------------------------------------
# The five model sizes shown in the onboarding "Engine" step (design subset of
# pipeline.WhisperModel; each id is a valid WhisperModel value). Advanced/.en/
# distil variants stay available in Settings.
ONBOARDING_SIZES = [
    {"id": "tiny", "name": "Tiny", "meta": "39M · 1GB",
     "note": "<b>Fastest, lowest accuracy.</b> Best for quick drafts and short, clean recordings."},
    {"id": "base", "name": "Base", "meta": "74M · 1GB",
     "note": "<b>Fast with decent accuracy.</b> A good default for clear single-speaker audio."},
    {"id": "small", "name": "Small", "meta": "244M · 2GB",
     "note": "<b>Balanced speed and accuracy.</b> Handles light background noise well."},
    {"id": "medium", "name": "Medium", "meta": "769M · 5GB",
     "note": "<b>Strong accuracy, slower.</b> Reliable for interviews and accented speech."},
    {"id": "large-v3", "name": "Large-v3", "meta": "1.5B · 10GB",
     "note": "<b>Best accuracy, multilingual.</b> Recommended for research-grade transcripts. Slowest."},
    {"id": "large-v3-turbo", "name": "Large Turbo", "meta": "809M · 6GB",
     "note": "<b>Near-large accuracy, much faster.</b> Recommended on Apple Silicon (whisper.cpp / Metal). Multilingual."},
]


def _onboarding_size(active: str) -> str:
    """The preselected size card: the active model if it's one of the cards, else
    the platform default (large-v3-turbo on Apple Silicon, small elsewhere)."""
    ids = {s["id"] for s in ONBOARDING_SIZES}
    if active in ids:
        return active
    return pipeline.DEFAULT_MODEL if pipeline.DEFAULT_MODEL in ids else "small"


@app.get("/sessions/<session_id>/events")
def session_events(session_id: str):
    """Server-Sent Events stream of stage/status changes for one session.

    Emits the current state immediately on connect (from the durable row), then
    live deltas via the broker, and closes on a terminal ``status`` event. The
    client reacts to ``done``/``error`` by fetching the final ``/status`` render.
    """
    if _sessions.get(session_id) is None:
        abort(404)

    def initial():
        # Read inside the stream (after subscribe) so we can't miss a terminal
        # event that fired in the gap between the existence check and subscribe.
        row = _sessions.get(session_id) or {}
        status = row.get("status")
        if status in ("done", "error"):
            return {"status": status}
        return (_stage_event(row["stage"], row.get("duration")) if row.get("stage")
                else {"status": status or "queued"})

    return sse_response(
        _broker,
        session_id,
        initial=initial,
        terminal=lambda e: e.get("status") in ("done", "error"),
    )


@app.get("/models/events")
def models_events():
    """Server-Sent Events stream of global model-load state.

    Emits the current state immediately on connect (so a late/reconnecting
    client is correct), then a fresh payload each time a model finishes loading
    or the active model / device changes. The client flips the "loading models"
    toast to a success state and refreshes model dropdowns. Long-lived: model
    state has no terminal, so the browser keeps the stream open until navigation.
    """
    return sse_response(_broker, MODELS_CHANNEL, initial=lambda: _models_event(_manager.status()))


@app.get("/sessions/<session_id>/audio")
def session_audio(session_id: str):
    path = _sessions.audio_path(session_id)
    if not path or not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=False)


@app.get("/sessions/<session_id>/download/<fmt>")
def download(session_id: str, fmt: str):
    if fmt not in pipeline.OUTPUT_FORMATS:
        abort(404)
    path = _sessions.artifact_path(session_id, fmt)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})?$")


def _valid_lang(lang: str) -> bool:
    """A safe BCP-47-ish language tag (also guards translation file paths)."""
    return bool(_LANG_RE.match(lang or ""))


# Curated target languages for the transcript translate picker. Each carries an
# English name and the language's own native name (shown in the dropdown / add
# dialog, scholastic-style). The list is intentionally short and common; Google
# Translate supports far more, but a tidy menu beats an exhaustive one.
TRANSLATION_LANGUAGES = [
    {"code": "en", "name": "English", "native": "English"},
    {"code": "es", "name": "Spanish", "native": "Español"},
    {"code": "fr", "name": "French", "native": "Français"},
    {"code": "de", "name": "German", "native": "Deutsch"},
    {"code": "it", "name": "Italian", "native": "Italiano"},
    {"code": "pt", "name": "Portuguese", "native": "Português"},
    {"code": "pt-BR", "name": "Portuguese (Brazil)", "native": "Português (BR)"},
    {"code": "nl", "name": "Dutch", "native": "Nederlands"},
    {"code": "ru", "name": "Russian", "native": "Русский"},
    {"code": "ja", "name": "Japanese", "native": "日本語"},
    {"code": "ko", "name": "Korean", "native": "한국어"},
    {"code": "zh", "name": "Chinese", "native": "中文"},
    {"code": "ar", "name": "Arabic", "native": "العربية"},
    {"code": "hi", "name": "Hindi", "native": "हिन्दी"},
]
_LANG_BY_CODE = {lang["code"]: lang for lang in TRANSLATION_LANGUAGES}


@app.get("/sessions/<session_id>/translate/events")
def translate_events(session_id: str):
    """SSE stream of translation progress for one session.

    Emits the current per-language status map on connect (durable, so a
    reconnecting client is correct), then live deltas; closes on a terminal
    ``done``/``error`` event for the language being watched.
    """
    if _sessions.get(session_id) is None:
        abort(404)

    def initial():
        return {"translations": _sessions.get_translations(session_id)}

    return sse_response(
        _broker,
        translate_job.channel(session_id),
        initial=initial,
        terminal=lambda e: e.get("status") in ("done", "error"),
    )


@app.get("/sessions/<session_id>/translation/<lang>/download/<fmt>")
def download_translation(session_id: str, lang: str, fmt: str):
    """Generate the translation export on demand from the joined segments, so it
    reflects the current speakers (reassignments) and the original-text fallback for
    any segment edited since translation."""
    if fmt not in pipeline.OUTPUT_FORMATS or not _valid_lang(lang):
        abort(404)
    overlay = _sessions.load_translation(session_id, lang)
    if overlay is None:
        abort(404)
    result = _sessions.load_result(session_id) or {}
    orig = _sessions.current_segments(session_id, result.get("segments", []))
    segs = translation_overlay.apply_overlay(orig, overlay)

    name = f"transcript.translation.{lang}.{fmt}"
    tmpdir = tempfile.mkdtemp(prefix="wx-tr-")
    out_path = os.path.join(tmpdir, name)
    if fmt == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"target_language": lang, "segments": segs}, f, ensure_ascii=False)
    else:
        from whisperx.utils import get_writer

        # get_writer strips the last dotted suffix off the stem (os.path.splitext);
        # the language tag looks like that suffix, so feed a throwaway ".x" to eat,
        # leaving the final name transcript.translation.<lang>.<fmt>.
        stem = os.path.join(tmpdir, f"transcript.translation.{lang}.x")
        writer = get_writer(fmt, tmpdir)
        writer({"segments": segs, "language": lang}, stem, pipeline.WRITER_OPTIONS)
    if not os.path.exists(out_path):
        abort(404)
    return send_file(out_path, as_attachment=True, download_name=name)


@app.get("/sessions/<session_id>/export.md")
def export_markdown(session_id: str):
    """Markdown transcript of the current (edit-overlaid) segments: title, then
    per-turn speaker tag + timestamp span + text."""
    row = _sessions.get(session_id)
    if row is None:
        abort(404)
    result = _sessions.load_result(session_id) or {}
    segments = _sessions.current_segments(session_id, result.get("segments", []))
    view = {**result, "segments": segments}
    title = row.get("filename") or "Transcript"
    md = render_markdown(view, _sessions.get_speaker_names(session_id), title=title)
    fname = f"{secure_filename(title) or 'transcript'}.md"
    return Response(
        md,
        mimetype="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# =============================================================================
# JSON API (/api/*) — the surface the Svelte SPA consumes. Binary, SSE, and the
# OAuth-callback routes stay at root; everything else is served the SPA shell by
# the catch-all at the bottom of this file.
# =============================================================================

# Transcription-language options (auto-detect + common), mirrors the old
# partials/_language_select.html list.
TRANSCRIBE_LANGUAGES = [
    {"code": "", "label": "Auto-detect"},
    {"code": "en", "label": "English"}, {"code": "es", "label": "Spanish"},
    {"code": "fr", "label": "French"}, {"code": "de", "label": "German"},
    {"code": "it", "label": "Italian"}, {"code": "pt", "label": "Portuguese"},
    {"code": "nl", "label": "Dutch"}, {"code": "ja", "label": "Japanese"},
    {"code": "zh", "label": "Chinese"}, {"code": "ru", "label": "Russian"},
]


def _body() -> dict:
    """Request payload as a dict, tolerant of JSON or form encoding."""
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    return {k: v for k, v in request.form.items()}


def _current_segments(session_id: str) -> tuple[dict, list]:
    """(result, edit-overlaid+coalesced segments) for a session."""
    result = _sessions.load_result(session_id) or {}
    segments = _sessions.current_segments(session_id, result.get("segments", []))
    return result, segments


def _turn_words(segments: list, seg_indices: list) -> list:
    """Flatten a turn's words to {text, start?, end?, stale?} entries — word-timed
    when the aligner produced words, else one segment-timed entry (mirrors
    render._word_spans, including the translation `stale` fallback flag)."""
    out: list[dict] = []
    for k in seg_indices:
        seg = segments[k]
        words = seg.get("words") or []
        if words:
            for w in words:
                token = (w.get("word") or "").strip()
                if not token:
                    continue
                wd: dict = {"text": token}
                if w.get("start") is not None and w.get("end") is not None:
                    wd["start"] = round(float(w["start"]), 3)
                    wd["end"] = round(float(w["end"]), 3)
                out.append(wd)
        else:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            wd = {"text": text}
            if seg.get("start") is not None and seg.get("end") is not None:
                wd["start"] = round(float(seg["start"]), 3)
                wd["end"] = round(float(seg["end"]), 3)
            if seg.get("stale"):
                wd["stale"] = True
            out.append(wd)
    return out


def _build_turns(segments: list, names: dict) -> list:
    """Speaker-grouped turns the SPA renders. Turn indices come from group_turns so
    they match the edit endpoints; empty-text turns are skipped from the output but
    never renumber the rest (mirrors render.render_transcript)."""
    turns = []
    for t in group_turns(segments):
        words = _turn_words(segments, t.seg_indices)
        if not any(w["text"] for w in words):
            continue
        turns.append({
            "index": t.index,
            "speaker": t.speaker,
            "label": resolve_label(t.speaker, names),
            "start": t.start,
            "end": t.end,
            "words": words,
            "text": t.text,
        })
    return turns


def _transcript_payload(session_id: str) -> dict:
    """{turns, segments, can_undo} from the current overlaid segments — the shared
    shape returned by the edit / undo / reassign endpoints."""
    _result, segments = _current_segments(session_id)
    names = _sessions.get_speaker_names(session_id)
    return {
        "turns": _build_turns(segments, names),
        "segments": segments,
        "can_undo": _sessions.edit_history_len(session_id) > 0,
    }


# --- Sessions ----------------------------------------------------------------
@app.get("/api/sessions")
def api_list_sessions():
    rows = _sessions.list()
    return jsonify({"sessions": [_card(r) for r in rows], "summary": _summary(rows)})


@app.post("/api/sessions")
def api_create_session():
    if not models_ready():
        return jsonify({"error": "loading_models"}), 503
    file = request.files.get("audio")
    if not file or not file.filename:
        return jsonify({"error": "No audio file uploaded."}), 400

    def _int(name):
        v = (request.form.get(name) or "").strip()
        return int(v) if v.isdigit() else None

    requested = (request.form.get("model") or "").strip()
    if requested:
        try:
            model = pipeline.WhisperModel(requested).value
        except ValueError:
            return jsonify({"error": f"Unknown model: {requested}"}), 400
    else:
        model = _manager.active

    session_id = uuid.uuid4().hex
    safe_name = secure_filename(file.filename) or "audio"
    ext = os.path.splitext(safe_name)[1] or ".bin"
    audio_filename = f"audio{ext}"
    os.makedirs(_sessions.session_dir(session_id), exist_ok=True)
    saved_path = os.path.join(_sessions.session_dir(session_id), audio_filename)
    file.save(saved_path)

    # Reject over-long audio upfront (header probe, no decode) so the user gets a
    # readable error now instead of a multi-hour run or an OOM mid-job.
    if MAX_AUDIO_HOURS > 0:
        dur = _probe_duration_seconds(saved_path)
        if dur is not None and dur > MAX_AUDIO_HOURS * 3600:
            shutil.rmtree(_sessions.session_dir(session_id), ignore_errors=True)
            return jsonify({
                "error": (f"Audio is {dur / 3600:.1f} h, over the "
                          f"{MAX_AUDIO_HOURS:g} h limit. Split it into shorter "
                          f"files and upload each."),
                "duration_hours": round(dur / 3600, 2),
                "max_hours": MAX_AUDIO_HOURS,
            }), 413

    display_name = (request.form.get("name") or "").strip() or file.filename
    _sessions.create(
        session_id,
        filename=display_name,
        audio_filename=audio_filename,
        options={
            "language": (request.form.get("language") or "").strip() or None,
            "min_speakers": _int("min_speakers"),
            "max_speakers": _int("max_speakers"),
        },
        model=model,
    )
    _queue.submit(session_id)
    return jsonify({"id": session_id, "status": "queued"}), 201


@app.get("/api/sessions/<session_id>")
def api_get_session(session_id: str):
    row = _sessions.get(session_id)
    if row is None:
        abort(404)
    row.pop("audio_filename", None)
    card = _card(row)
    names = _sessions.get_speaker_names(session_id)
    result = _sessions.load_result(session_id)
    if result is not None:
        result["segments"] = _sessions.current_segments(session_id, result.get("segments", []))
        card["turns"] = _build_turns(result["segments"], names)
    card["result"] = result
    card["speaker_names"] = names
    card["can_undo"] = _sessions.edit_history_len(session_id) > 0
    card["created_at"] = row.get("created_at")
    card["updated_at"] = row.get("updated_at")
    card["options"] = row.get("options") or {}
    card["formats"] = [f for f in pipeline.OUTPUT_FORMATS
                       if os.path.exists(_sessions.artifact_path(session_id, f))]
    return jsonify(card)


@app.post("/api/sessions/<session_id>/rename")
def api_rename_session(session_id: str):
    if _sessions.get(session_id) is None:
        abort(404)
    name = (_body().get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name cannot be empty."}), 400
    _sessions.rename(session_id, name)
    return jsonify({"id": session_id, "filename": name})


@app.post("/api/sessions/<session_id>/delete")
def api_delete_session(session_id: str):
    _queue.cancel(session_id)
    if not _sessions.delete(session_id):
        abort(404)
    return jsonify({"deleted": True})


@app.post("/api/sessions/<session_id>/turns/<int:turn_index>")
def api_edit_turn(session_id: str, turn_index: int):
    if _sessions.get(session_id) is None:
        abort(404)
    try:
        _sessions.save_turn_edit(session_id, turn_index, _body().get("text", ""))
    except IndexError:
        return jsonify({"error": "Unknown turn."}), 400
    return jsonify(_transcript_payload(session_id))


@app.post("/api/sessions/<session_id>/undo")
def api_undo_edit(session_id: str):
    if _sessions.get(session_id) is None:
        abort(404)
    _sessions.undo_turn_edit(session_id)
    return jsonify(_transcript_payload(session_id))


@app.get("/api/sessions/<session_id>/speakers")
def api_list_speakers(session_id: str):
    if _sessions.get(session_id) is None:
        abort(404)
    _result, segments = _current_segments(session_id)
    names = _sessions.get_speaker_names(session_id)
    return jsonify([
        {"key": key, "label": resolve_label(key, names)}
        for key in distinct_speakers(segments)
    ])


@app.post("/api/sessions/<session_id>/speakers")
def api_rename_speaker(session_id: str):
    if _sessions.get(session_id) is None:
        abort(404)
    data = _body()
    speaker = (data.get("speaker") or "").strip()
    if not speaker:
        return jsonify({"error": "Missing speaker key."}), 400
    name = (data.get("name") or "").strip()
    _sessions.set_speaker_name(session_id, speaker, name)
    return jsonify({"key": speaker, "label": resolve_label(speaker, {speaker: name} if name else None)})


@app.post("/api/sessions/<session_id>/turns/<int:turn_index>/speaker")
def api_reassign_turn(session_id: str, turn_index: int):
    if _sessions.get(session_id) is None:
        abort(404)
    data = _body()
    speaker = (data.get("speaker") or "").strip()
    name = (data.get("name") or "").strip()
    if not speaker:
        if not name:
            return jsonify({"error": "Provide a speaker key or a name for a new speaker."}), 400
        _result, segments = _current_segments(session_id)
        names = _sessions.get_speaker_names(session_id)
        taken = {resolve_label(k, names).casefold() for k in distinct_speakers(segments)}
        taken |= {v.casefold() for v in names.values()}
        if name.casefold() in taken:
            return jsonify({"error": f"A speaker named {name!r} already exists."}), 409
        existing = set(distinct_speakers(segments)) | set(names)
        speaker = next_speaker_key(existing)
    if name:
        _sessions.set_speaker_name(session_id, speaker, name)
    try:
        _sessions.save_turn_reassign(session_id, turn_index, speaker)
    except IndexError:
        return jsonify({"error": "Unknown turn."}), 400
    return jsonify(_transcript_payload(session_id))


# --- Models / device ---------------------------------------------------------
@app.get("/api/models")
def api_models():
    return jsonify(_manager.status())


@app.post("/api/models/active")
def api_switch_model():
    model = (_body().get("model") or "").strip()
    try:
        model = pipeline.WhisperModel(model).value
    except ValueError:
        return jsonify({"error": f"Unknown model: {model}"}), 400
    status = _manager.set_active(model)
    _sessions.set_setting("active_model", model)
    return jsonify(status)


@app.post("/api/device")
def api_switch_device():
    device = (_body().get("device") or "").strip()
    if device not in pipeline.DEVICES:
        return jsonify({"error": f"Unknown device: {device}"}), 400
    if _sessions.has_active_jobs():
        return jsonify({"error": "busy", **_manager.status()}), 409
    try:
        status = _manager.set_device(device)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    _sessions.set_setting("device", device)
    return jsonify(status)


# --- Translation -------------------------------------------------------------
@app.post("/api/sessions/<session_id>/translate")
def api_translate_session(session_id: str):
    row = _sessions.get(session_id)
    if row is None:
        abort(404)
    if row["status"] != "done":
        return jsonify({"error": "Transcript is not ready to translate."}), 409
    target = (_body().get("target_language") or "").strip()
    if not _valid_lang(target):
        return jsonify({"error": "Invalid target language."}), 400
    if not secret_store.resolve_google_api_key():
        return jsonify({"error": "Add a Google Translation API key in Settings first."}), 400
    service = _sessions.get_setting("translation_service", DEFAULT_SERVICE)
    if service not in SERVICES:
        service = DEFAULT_SERVICE
    _translate_queue.submit(session_id, target, service)
    return jsonify({"lang": target, "status": "running", "service": service})


@app.get("/api/sessions/<session_id>/translation/<lang>")
def api_view_translation(session_id: str, lang: str):
    if _sessions.get(session_id) is None or not _valid_lang(lang):
        abort(404)
    overlay = _sessions.load_translation(session_id, lang)
    if overlay is None:
        abort(404)
    _result, orig = _current_segments(session_id)
    segs = translation_overlay.apply_overlay(orig, overlay)
    names = _sessions.get_speaker_names(session_id)
    return jsonify({
        "target_language": lang,
        "turns": _build_turns(segs, names),
        "segments": segs,
    })


# --- Onboarding --------------------------------------------------------------
@app.get("/api/onboarding")
def api_onboarding():
    status = _manager.status()
    return jsonify({
        "token": secret_store.resolve_hf_token() or "",
        "sizes": ONBOARDING_SIZES,
        "selected_size": _onboarding_size(status["active"]),
        "models": status,
        "diarize_model": secret_store.DIARIZE_MODEL,
        "backup": _backup_json(),
    })


@app.post("/api/onboarding/verify")
def api_onboarding_verify():
    token = (_body().get("token") or "").strip()
    ok, detail = secret_store.verify_token(token)
    return jsonify({"ok": ok, "detail": detail})


@app.post("/api/onboarding")
def api_onboarding_finish():
    data = _body()
    token = (data.get("token") or "").strip()
    model = (data.get("model") or "").strip()
    device = (data.get("device") or "").strip()
    try:
        model = pipeline.WhisperModel(model).value
    except ValueError:
        return jsonify({"error": f"Unknown model: {model}"}), 400
    if device not in pipeline.DEVICES:
        return jsonify({"error": f"Unknown device: {device}"}), 400
    if token:
        ok, detail = secret_store.verify_token(token)
        if not ok:
            return jsonify({"error": detail or "Token did not verify."}), 400
        try:
            secret_store.set_hf_token(token)
        except secret_store.SecretStoreUnavailable as exc:
            return jsonify({"store_error": str(exc)}), 500
    _sessions.set_setting("active_model", model)
    _sessions.set_setting("device", device)
    _manager.set_active(model)
    if device != _manager.device:
        try:
            _manager.set_device(device)
        except ValueError:
            pass
    _manager.reset_diarize()
    _sessions.set_setting("onboarded", "1")
    return jsonify({"ok": True})


# --- Settings ----------------------------------------------------------------
def _settings_payload() -> dict:
    return {
        "default_language": _sessions.get_setting("default_language", ""),
        "languages": TRANSCRIBE_LANGUAGES,
        "models": _manager.status(),
        "translation_service": _sessions.get_setting("translation_service", DEFAULT_SERVICE),
        "translation_services": [
            {"id": name, "label": cls.label} for name, cls in SERVICES.items()
        ],
        "translation_languages": TRANSLATION_LANGUAGES,
        "google_key": {"key_set": bool(secret_store.resolve_google_api_key())},
        "diarize": {
            "version": diarize_model.derive_version(diarize_model.resolve_local_model()),
            "model_name": diarize_model.REPO_ID,
            "token_set": bool(secret_store.resolve_hf_token()),
        },
        "backup": _backup_json(),
        "onboarded": _sessions.get_setting("onboarded") == "1",
    }


@app.get("/api/settings")
def api_settings():
    return jsonify(_settings_payload())


@app.post("/api/settings")
def api_save_settings():
    lang = (_body().get("default_language") or "").strip()
    _sessions.set_setting("default_language", lang)
    return jsonify({"ok": True, "default_language": lang})


def _hf_token_payload(notice: str, notice_ok: bool) -> dict:
    return {"token_set": bool(secret_store.resolve_hf_token()),
            "notice": notice, "notice_ok": notice_ok}


@app.post("/api/settings/hf-token")
def api_hf_token():
    token = (_body().get("hf_token") or "").strip()
    ok, detail = secret_store.verify_token(token)
    if not ok:
        return jsonify(_hf_token_payload(detail, False)), 400
    try:
        secret_store.set_hf_token(token)
    except secret_store.SecretStoreUnavailable as exc:
        return jsonify(_hf_token_payload(str(exc), False)), 500
    _manager.reset_diarize()
    return jsonify(_hf_token_payload("Token saved and verified.", True))


@app.post("/api/settings/hf-token/clear")
def api_hf_token_clear():
    secret_store.delete_hf_token()
    _manager.reset_diarize()
    token_set = bool(secret_store.resolve_hf_token())
    notice = ("Cleared the stored token, but HF_TOKEN is still set in the environment."
              if token_set else "Token cleared. Speaker diarization is now disabled.")
    return jsonify({"token_set": token_set, "notice": notice, "notice_ok": True})


def _google_key_payload(notice: str, notice_ok: bool) -> dict:
    return {"key_set": bool(secret_store.resolve_google_api_key()),
            "notice": notice, "notice_ok": notice_ok}


@app.post("/api/settings/google-key")
def api_google_key():
    key = (_body().get("google_key") or "").strip()
    ok, detail = secret_store.verify_google_api_key(key)
    if not ok:
        return jsonify(_google_key_payload(detail, False)), 400
    try:
        secret_store.set_google_api_key(key)
    except secret_store.SecretStoreUnavailable as exc:
        return jsonify(_google_key_payload(str(exc), False)), 500
    return jsonify(_google_key_payload("Key saved and verified.", True))


@app.post("/api/settings/google-key/clear")
def api_google_key_clear():
    secret_store.delete_google_api_key()
    key_set = bool(secret_store.resolve_google_api_key())
    notice = ("Cleared the stored key, but GOOGLE_TRANSLATE_API_KEY is still set "
              "in the environment." if key_set else "Key cleared. Translation is now disabled.")
    return jsonify({"key_set": key_set, "notice": notice, "notice_ok": True})


@app.post("/api/settings/translation-service")
def api_translation_service():
    service = (_body().get("translation_service") or "").strip()
    if service not in SERVICES:
        return jsonify({"error": "Unknown translation service."}), 400
    _sessions.set_setting("translation_service", service)
    return jsonify({"ok": True})


@app.post("/api/settings/diarize-model/refresh")
def api_diarize_refresh():
    def _payload(notice, notice_ok):
        return {
            "version": diarize_model.derive_version(diarize_model.resolve_local_model()),
            "model_name": diarize_model.REPO_ID,
            "token_set": bool(secret_store.resolve_hf_token()),
            "notice": notice, "notice_ok": notice_ok,
        }
    token = secret_store.resolve_hf_token()
    if not token:
        return jsonify(_payload("Add a Hugging Face token above to fetch model updates.", False)), 400
    try:
        dest = diarize_model.vendor(token, dest_root=diarize_model.data_root())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Diarization model refresh failed")
        return jsonify(_payload(f"Refresh failed: {exc}", False)), 500
    _manager.reset_diarize()
    sha8 = dest.name.rsplit(".", 1)[-1]
    return jsonify(_payload(f"Refreshed to revision {sha8}.", True))


# --- Backup (one JSON surface; SSE state via /backup/status/events) -----------
@app.get("/api/backup/status")
def api_backup_status():
    return jsonify(_backup_json())


@app.post("/api/backup/connect")
def api_backup_connect():
    if _backup.is_linked():
        return jsonify(_backup_json())
    _apply_backup_folder(_body().get("backup_folder"))
    # Reuse the existing background consent flow; the SPA watches /backup/events
    # for the terminal {status, backup} event.
    _start_backup_link()
    return jsonify({"connecting": True})


@app.post("/api/backup/disconnect")
def api_backup_disconnect():
    from app.backup import oauth
    oauth.unlink()
    return jsonify(_backup_json(notice="Disconnected. Local data is untouched."))


@app.post("/api/backup/now")
def api_backup_now():
    if not _backup.is_linked():
        return jsonify({"error": "Backup backend is not linked."}), 409
    _run_backup_async()
    return jsonify(_backup_json()), 202


@app.post("/api/backup/restore")
def api_backup_restore():
    try:
        n = _backup.restore()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"restored": n, "backup": _backup_json()})


@app.post("/api/backup/bootstrap/adopt")
def api_backup_adopt():
    try:
        n = _backup.adopt_remote()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"restored": n, "backup": _backup_json()})


@app.post("/api/backup/bootstrap/overwrite")
def api_backup_overwrite():
    try:
        r = _backup.overwrite_remote()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"uploaded": r.uploaded, "skipped": r.skipped, "backup": _backup_json()})


@app.get("/api/backup/remote-info")
def api_backup_remote_info():
    try:
        remote = _backup.bootstrap()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"remote": _remote_json(remote)})


# --- SPA: serve the built Svelte client for all non-API/SSE/binary paths -------
# The Vite build lands in static/spa/ (gitignored; built by `cd app/web && bun run
# build`). Concrete routes above (/api/*, /sessions/<id>/{audio,events,download,…},
# /healthz, /models/events, /backup/*) win by specificity; everything else returns
# index.html so the client router can handle deep links and reloads.
_SPA_DIR = os.path.join(os.path.dirname(__file__), "static", "spa")
_SPA_RESERVED = {"api", "healthz", "backup", "models", "static", "oauth"}


@app.route("/")
@app.route("/<path:spa_path>")
def spa(spa_path: str = ""):
    head = spa_path.split("/", 1)[0]
    if head in _SPA_RESERVED:
        abort(404)  # unknown API/asset path — don't return the shell
    # A built asset request (has a file extension) that no concrete/static route
    # served is a genuine miss; only bare client routes get the shell.
    index_html = os.path.join(_SPA_DIR, "index.html")
    if not os.path.exists(index_html):
        return ("SPA not built — run: cd app/web && bun run build", 500)
    if "." in head and not os.path.exists(os.path.join(_SPA_DIR, spa_path)):
        abort(404)
    return send_file(index_html)


# Kick off model loading at import time (works under both `flask run` and __main__).
# Tests set WHISPERX_NO_WARM=1 to import the app without pulling a model.
if os.environ.get("WHISPERX_NO_WARM") != "1":
    threading.Thread(target=_warm_models, name="warm-models", daemon=True).start()


_shutdown_done = threading.Event()


def _shutdown(*_args) -> None:
    """Flush state on quit so a SIGTERM from the launcher doesn't strand WAL writes
    or the job executor. Idempotent (signal handler + finally may both call it)."""
    if _shutdown_done.is_set():
        return
    _shutdown_done.set()
    try:
        _queue.shutdown()
    except Exception:  # noqa: BLE001 - best effort on the way out
        pass
    try:
        _sessions.close()
    except Exception:  # noqa: BLE001
        pass


def _choose_port() -> int:
    """Honour ``PORT`` (default 5000); if it's already bound, grab an ephemeral free
    port so a relaunch (or a leftover process) never blocks startup."""
    import socket

    host = os.environ.get("HOST", "127.0.0.1")
    preferred = int(os.environ.get("PORT", "5000"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, preferred))
            return preferred
        except OSError:
            sock.bind((host, 0))
            return sock.getsockname()[1]


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGTERM, lambda *_: (_shutdown(), os._exit(0)))
    host = os.environ.get("HOST", "127.0.0.1")
    port = _choose_port()
    # Publish the port we actually bound (PORT may have been taken) so a wrapping
    # launcher/Tauri shell knows where to point the webview.
    (paths.data_dir() / "runtime-port").write_text(str(port))
    if os.environ.get("WHISPERX_OPEN_BROWSER") == "1":
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    logger.info("Serving on http://%s:%d", host, port)
    try:
        app.run(host=host, port=port, threaded=True)
    finally:
        _shutdown()
