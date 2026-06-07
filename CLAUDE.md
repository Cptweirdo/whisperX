# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

WhisperX is a CLI + Python library for fast, time-accurate speech recognition. It wraps OpenAI Whisper (via the faster-whisper / CTranslate2 backend) and adds three things Whisper lacks: VAD-based batched inference, word-level timestamps via forced phoneme alignment (wav2vec2), and speaker diarization (pyannote-audio).

## Commands

Uses `uv` for dependency management (not plain pip for dev work).

```bash
uv sync --all-extras                 # install project + dev deps (pytest)
uv run pytest tests/ -v              # run all tests
uv run pytest tests/test_word_timestamp_interpolation.py -v   # single test file
uv run pytest tests/test_word_timestamp_interpolation.py::test_name   # single test
uv lock --check                      # verify lockfile is up to date (CI gate)
uv run whisperx audio.mp3 --model large-v2 --diarize --highlight_words True   # run CLI
```

CI (`.github/workflows/`) tests against Python 3.10–3.13. `python-compatibility.yml` enforces `uv lock --check` and a bare `import whisperx` smoke test — keep both green.

### Web app (`app/`)

```bash
cd app/web && bun install      # install SPA deps (Svelte/Vite/Shoelace)
bun run dev                     # Vite dev server :5173, proxies API/SSE to Flask :5000
bun run build                   # build the SPA into app/static/spa (served by Flask)
bun run test                    # Vitest unit tests
bun run check                   # svelte-check (types)
./app/start.sh                  # build SPA if missing, install deps, run the Flask server
```

The Flask backend is a pure JSON API that serves the built SPA; run it with `python -m app.server` (or `start.sh`). Backend tests: `uv run pytest tests/test_api.py tests/test_sse.py -v`.

### GPU / CUDA

Requires CUDA toolkit 12.8 for GPU. torch is pinned to a CUDA 12.8 index (`cu128`) on x86_64 Linux and CPU index elsewhere — see `[tool.uv.sources]` in `pyproject.toml`. cuDNN library load failures are common; see `CUDNN_TROUBLESHOOTING.md`.

## Architecture

The pipeline runs in three sequential, independently-loaded stages. Each stage loads its own model and the result of one feeds the next:

1. **Transcribe** (`asr.py`) — `load_model()` returns a `FasterWhisperPipeline` (subclass of HF `Pipeline`) wrapping a custom `WhisperModel` (subclass of `faster_whisper.WhisperModel`). VAD (`vads/`) segments audio first, then batches the voiced segments through Whisper. Default ASR options live in `default_asr_options` inside `load_model()`.
2. **Align** (`alignment.py`) — `load_align_model()` + `align()` force-align the transcript to audio with a per-language wav2vec2 model to get word-level timestamps. Language→model mapping is in `DEFAULT_ALIGN_MODELS_TORCH` / `DEFAULT_ALIGN_MODELS_HF`. Alignment is skipped for `--task translate`.
3. **Diarize** (`diarize.py`) — `DiarizationPipeline` (pyannote) labels speakers; `assign_word_speakers()` maps speaker turns onto aligned words.

`transcribe.py::transcribe_task()` orchestrates all three from CLI args. `__main__.py::cli()` defines every CLI flag and is the `whisperx` entry point. The package `__init__.py` exposes the public API (`load_model`, `load_align_model`, `align`, `load_audio`, `assign_word_speakers`) via **lazy imports** — heavy deps (torch, pyannote) are only imported when the function is actually called, so adding a top-level eager import there will slow `import whisperx` and can break the compatibility smoke test.

### Key modules

- `audio.py` — audio loading + mel spectrogram. Defines core constants (`SAMPLE_RATE = 16000`, `N_SAMPLES`, frame/token rates) reused across stages.
- `vads/` — VAD backends behind a common interface; `--vad_method` selects `pyannote` (default) or `silero`.
- `schema.py` — `TypedDict` result shapes (`TranscriptionResult`, `AlignedTranscriptionResult`, `SingleSegment`, `SingleWordSegment`, …). The contract between stages; update here when changing result structure.
- `utils.py` — `get_writer()` and output writers (srt, vtt, txt, tsv, json, aud); also `LANGUAGES` / `TO_LANGUAGE_CODE`.
- `SubtitlesProcessor.py` — subtitle line splitting/formatting.
- `conjunctions.py` — per-language conjunction lists used for sentence/segment splitting.
- `log_utils.py` — `setup_logging()` / `get_logger()`; modules get loggers via `get_logger(__name__)`.

## Conventions

- **Commits:** never add a `Co-Authored-By:` trailer (or any AI co-author line) to commit messages.

## Notes

- Diarization and the pyannote VAD/diarization models are gated on Hugging Face — needs `--hf_token` (or `HF_TOKEN`) and accepting the model user agreements.
- `pyproject.toml` carries non-obvious `torchcodec` override-dependencies because it has no wheels for Linux aarch64.
- Version is bumped manually in `pyproject.toml`; releases go through `build-and-release.yml`.
- **Web app architecture — pure JSON API (`app/server.py`) + standalone Svelte SPA (`app/web/`):** the backend is a Flask **JSON API only** (no Jinja/htmx). All data routes live under `/api/*`; routes the browser hits as URLs stay at root: `/healthz`, the SSE streams, binary `/sessions/<id>/{audio,download/<fmt>,export.md}` + `/sessions/<id>/translation/<lang>/download/<fmt>`. A catch-all in `server.py` (`spa()`) serves the built SPA (`app/static/spa/index.html`) for every other path so client-side routing + deep links work; concrete routes win by URL specificity (reserved roots `{api,healthz,backup,models,static,oauth}` 404 instead of returning the shell). The **only** remaining server-rendered HTML is the two `app/templates/oauth_callback_*.html` pages, served by the loopback OAuth server in `app/backup/oauth.py` (not Flask). `app/render.py` keeps only `render_markdown` (the `.md` export) + `resolve_label`; the SPA renders transcripts client-side.
- **Web app — SPA lives in `app/web/` (Bun + Vite + Svelte 5 runes):** dev = `cd app/web && bun install && bun run dev` (Vite :5173, proxies `/api`,`/sessions/<id>/<sub>`,`/healthz`,`/models`,`/backup`,`/oauth` to Flask :5000 — see `vite.config.ts`); prod = `bun run build` → `app/static/spa/` (gitignored; built by the Dockerfile `assets` stage and `packaging/macos/build.py`, served by Flask). Tauri is unchanged — it still polls `/healthz` and navigates the webview to `/`. State is in `.svelte.ts` runes stores (`src/lib/stores/`: `sessions`, `session`, `models`, `backup`, `settings`, `ui`, `toast`); `src/lib/api.ts` is the typed fetch wrapper (uploads via XHR for progress); `src/lib/router.svelte.ts` is a tiny history-API router. **Shoelace** is used directly as custom elements (`src/shoelace.ts`); bind values via `onsl-input`/`onsl-change` into `$state` (the old htmx FormData-merge hack is gone). Icons load at runtime from `BASE_URL + 'shoelace'`, served by the `shoelaceAssets` Vite plugin (dev middleware + build copy). Reference SPA-bundled assets via `import.meta.env.BASE_URL` (e.g. the favicon), since prod `base` is `/static/spa/`. Tests: `cd app/web && bun run test` (Vitest + happy-dom) and `bun run check` (svelte-check; sl-button a11y warnings are expected false-positives).
- **Web app — `turns` are grouped server-side (don't reimplement in TS):** `GET /api/sessions/<id>` and the edit/undo/reassign/translation endpoints return pre-grouped `turns` (`{index,speaker,label,start,end,words:[{text,start?,end?,stale?}],text}`) built by `server.py::_build_turns` reusing `edits.group_turns` + `render.resolve_label`, **plus** the raw `segments`. The SPA renders from `turns` and edits by `turn.index`; keeping grouping server-side means the index can't drift from `edits.group_turns` (the contract that makes edits target the right turn — covered by `tests/test_api.py`).
- **Web app — live job progress via Server-Sent Events:** the upload→transcribe pipeline pushes stage updates (`decoding`/`transcribing`/`loading_align`/`aligning`/`diarizing`) over SSE. (1) `sse.py::Broker` — in-process per-`session_id` pub/sub; the job thread `publish()`es, each open SSE request drains a `subscribe()`d `queue.Queue`. (2) `pipeline.run_job(progress=…)` → `server.py::_on_stage` both `mark_stage`s the SQLite row **and** `broker.publish`es — **the DB row is the durable source of truth**, the broker carries live deltas; terminal (`done`/`error`) events fire *after* the store is updated. (3) every SSE endpoint streams `text/event-stream` via `sse.py::sse_response(broker, channel, *, initial=, terminal=, pending=)` (owns subscribe/unsubscribe, the 15s `:` keepalive, no-buffering headers). Channels: `/sessions/<id>/events` (terminal `done`/`error`), `/sessions/<id>/translate/events` (per-lang terminal), `/models/events` + `/backup/status/events` (persistent), `/backup/events` (one-shot OAuth, `pending` replay). **Payloads are JSON** (backup events carry `{status, backup}` state, not HTML). **When adding a stream, route it through `sse_response`.** Client side: `app/web/src/lib/sse.ts` — `openSSE(url, onData)` (EventSource + JSON-parse guard + auto-reconnect) and `sseStream(url, {onData, terminal})` (auto-close + a `stop()` for `$effect` cleanup); reuse these rather than `new EventSource` inline. Tests: the broker has `tests/test_sse.py` (Python); `sse.ts` has `app/web/tests/sse.test.ts` (Vitest, fake EventSource). New columns on the `sessions` table (e.g. `stage`) need a `PRAGMA table_info` + `ALTER TABLE` migration in `SessionStore._migrate` — SQLite has no `ADD COLUMN IF NOT EXISTS`.
- **Tauri boot splash (`packaging/macos/tauri/dist/index.html`):** the Rust shell (`src/main.rs`) creates the window on this **local** page, then `navigate()`s it to `http://127.0.0.1:<port>/` once `/healthz` returns 200. It loads *before* the Flask server is up, so it must be self-contained — **no React/Babel, no `app/static` assets, no localhost deps**. Fonts come from Google Fonts `<link>` (degrade to Georgia/system-mono offline; the real UI self-hosts them via `vendor.css`). The current design is "Loading Screen" option A · Parchment from the Claude Design handoff (`SplashParchment`): a faux borderless light card floating on a parchment backdrop, baked monogram squircle SVG (superellipse `n≈5`, path precomputed — no JS), Literata wordmark + JetBrains Mono tags + shimmer bar. It's a *mock* of a borderless window rendered inside the real (decorated) Tauri window — the stage backdrop fills the actual webview; don't expect `main.rs` to be borderless.
- **Verifying UI with `playwright-cli`:** first **load the skill** (`Skill` tool, `playwright-cli`) before any CLI call — it surfaces the command set and is required for the CLI to work in-session. Then two gotchas: (1) `playwright-cli open` **defaults to `--browser=chrome`, which isn't installed** on this machine → `open --browser=chromium` (the bundled Playwright Chromium). (2) The **`file:` protocol is blocked** — serve static files over http first (`cd <dir> && python3 -m http.server <port> &`) and `goto http://127.0.0.1:<port>/…`. Typical flow: `open --browser=chromium` → `resize W H` → `goto <url>` → `screenshot --filename=…`, then `close` + kill the http server.
