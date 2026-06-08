# C++ Host Swap — Status (native oat++ server)

> The host swap replaces the Python Flask host (`app/`) with a native **oat++**
> HTTP/SSE server that links the C++ engine core directly, so there is **no Python
> at runtime**. The engine core itself (decode/VAD/ASR/align/diarize/assign +
> `SessionStore` + writers + the `run_job` orchestrator) was already landed in
> migration phases 0–6; this is the deferred "host swap" from
> [`cpp-core-spa-architecture.md`](./cpp-core-spa-architecture.md) §3, reproducing
> the SPA's JSON/SSE contract ([`api-reference.md`](./api-reference.md)).

## Decisions (settled with the user)

- **Acceleration:** CPU-only v1 — sherpa-onnx/ORT on CPU. faster-whisper, torch
  wav2vec2, pyannote, mlx, whisper.cpp/Metal are gone with Python. GPU (ORT CUDA EP)
  and Metal/GGML are explicit later passes.
- **Scope:** Core MVP + translation + the Google Drive **auth link, Drive REST
  client, and the backup engine** — all built. The full cloud-backup path
  (manifest pointer + content-addressed object store + GC + streamed restore +
  bootstrap/conflict + periodic) now works end-to-end; only later upload hardening
  (chunk-resume) and alternate backends remain.
- **Backup auth:** a **hand-rolled OAuth2 (auth-code + PKCE/S256) lib** over the
  existing libcurl client — NOT `google_auth_oauthlib`/google-cloud-cpp. Provider
  config is injectable (Google wired now); credentials persist in the OS keyring
  under `google_drive_creds`, **byte-compatible with the Python host's entry** so a
  link made by either host is interchangeable. PKCE crypto is dependency-free
  (self-contained SHA-256 + base64url + OS CSPRNG — no OpenSSL coupling, since
  libcurl's TLS backend is platform-chosen). Loopback redirect reuses the running
  oat++ server (`/oauth/callback`); consent runs on a dedicated thread (never the
  transcription worker) and reports progress over SSE. PKCE is an intentional
  improvement on the Python oracle, which used the bare auth-code flow.
- **Drive client:** `drive::DriveClient` wraps Drive v3 `files.*` (list/get/
  download/create/update/delete + `find_child`/`ensure_folder`) over libcurl —
  the native analog of `app/backup/gdrive.py`'s googleapiclient calls. Auth +
  HTTP transport are injectable, so request shaping is unit-tested offline. Large
  objects stream: `upload_resumable` (single streamed PUT via a resumable session)
  up, `download_to_file` down, so 300-400 MB audio blobs never sit in memory.
- **Backup engine:** `backup::BackupEngine` (port of `app/backup/service.py`)
  over a `StorageBackend` abstraction — `GDriveBackend` (Drive) + `LocalFsBackend`
  (filesystem reference / offline tests). Content-addressed `objects/<sha256>` +
  an atomic `manifest.json` commit pointer; `backup_now`+GC, streamed `restore`+
  prune, bootstrap/`assess_link` (fresh/remote-only/in-sync/diverged), adopt/
  overwrite, a periodic auto-backup thread, and a per-device `.backup/state.json`
  sidecar. The `BackupService` facade composes it: OAuth link + engine, two SSE
  channels (`__backup__` connect, `__backup_status__` persistent).
- **Translation:** Google Cloud Translation **v2 REST (API key)** — **built**. Ported
  `app/translation/google.py` 1:1 over the existing libcurl client (NOT google-cloud-cpp,
  which is v3/gRPC + service-account auth — it would break the API-key UI/keyring contract
  and balloon the build). Structure-locked overlay + on-demand exports match the oracle.
- **Model assets:** a libcurl HF-mirror downloader + libarchive `.tar.bz2` fallback —
  **built** (task 7). Models lazy-download from the public sha-pinned mirrors
  (`KonstantK/whisper-onnx-sherpa` / `wav2vec2-align-onnx` / `diarize-onnx-sherpa`),
  falling back to sherpa-onnx's release tarballs for un-mirrored Whisper models; cached
  under `WHISPERX_SHERPA_CACHE` (default `~/.cache/whisperx-sherpa`). Local-dir env vars
  (below) still win for dev/CI.
- **Secrets:** native OS keyring via **hrantzsch/keychain** — **built** (task 7). The HF
  token is stored/verified on onboarding (`HF_TOKEN` env wins); no functional runtime
  consumer in CPU-sherpa (mirrors are public, diarizer is token-free) — kept for
  onboarding UX parity + as the gdrive/translate-secret foundation.
- **URL decoding:** kept a small local `url_decode` (oatpp 1.3.0 ships no URL
  percent-decoder; it exists only on unreleased oatpp master, which also removed
  `core/` and relocated `ObjectMapper` — not worth the churn for a rare form-body
  fallback, since the SPA sends JSON).

## What's built (this branch)

New adapter `adapters/server/`, a CMake target (`whisperx_server`) gated behind
`-DWHISPERX_BUILD_SERVER=ON` (requires `WHISPERX_CORE_AUDIO=ON`). Pulls **oat++ 1.3.0
+ spdlog 1.14.1** via FetchContent (the light-dep pattern); links
`whisperx_core_audio`. The Python `app/` stays in-repo as the parity oracle.

| Area | Files | Notes |
|---|---|---|
| Config + .env | `config.{hpp,cpp}` | port of `server.py::_load_dotenv` precedence + `paths.data_dir` + `WHISPERX_*` knobs |
| Logging | `log/log.{hpp,cpp}` | spdlog console + rotating file (`<data_dir>/logs`), level via `WHISPERX_LOG_LEVEL` |
| SSE | `sse/broker.{hpp,cpp}`, `sse/sse_response.{hpp,cpp}` | port of `app/sse.py` (bounded queue, drop-on-full, keepalive, initial/terminal/pending) + an oat++ streaming body |
| Job queue | `jobs/jobs.{hpp,cpp}` | port of `app/jobs.py` (single worker, cancel-at-stage-boundary, terminal-publish-after-store) |
| Pipeline runner | `jobs/runner.{hpp,cpp}` | ports the pybind `run_job` Steps wiring (decode→silero VAD→merge→sherpa Whisper→align→diarize→assign) + native writers + `mark_done` |
| Models | `models/model_manager.{hpp,cpp}`, `models/assets.{hpp,cpp}` | CPU/sherpa-only resident-engine manager + `ModelStatus` shape; local-dir asset resolver → mirror/downloader fallback |
| Downloader | `assets/downloader.{hpp,cpp}`, `http/curl_client.{hpp,cpp}` | libcurl HF-mirror fetch + cache + libarchive `.tar.bz2` extract (port of `asr_sherpa`/`diarize_sherpa`/`export_align_onnx` resolvers). `curl_client` also grew `post`/`request` (PATCH/DELETE via CUSTOMREQUEST)/`form_encode` for the OAuth + Drive calls |
| Secrets | `secrets/keyring.{hpp,cpp}`, `secrets/hf_verify.{hpp,cpp}` | hrantzsch/keychain wrapper (port of `secret_store.py`) + live HF token verify (whoami + gated-model check) |
| Backup auth (OAuth) | `oauth/{crypto,pkce,credentials,client,flow,backup_service}.{hpp,cpp}`, `oauth/{provider,token_store}.hpp` | hand-rolled auth-code+PKCE/S256 lib (port of `app/backup/oauth.py`): dep-free crypto, Python-compatible creds JSON in keyring, `OAuthClient` (authorize/exchange/refresh/access-token), `LinkFlow` (consent thread + `/oauth/callback` promise + SSE), `BackupService` facade |
| Drive client | `drive/drive_client.{hpp,cpp}` | Drive v3 `files.*` REST wrapper (port of `app/backup/gdrive.py`): list/get/download/create(+multipart upload)/update/delete + `find_child`/`ensure_folder`; injectable auth + transport |
| Translation | `translate/google.{hpp,cpp}`, `translate/overlay.{hpp,cpp}`, `translate/queue.{hpp,cpp}` | Cloud Translation **v2 REST (API key)** over the libcurl client + `verify_api_key`, with request batching (100 seg / 25k char caps) — 1:1 port of `translation/google.py`; structure-locked overlay (`start_key`/`build_entries`/`apply_overlay`, v2 + legacy-v1, stale fallback — port of `translation_overlay.py`); single-worker network executor with durable status + `translate:<id>` SSE (port of `translate_job.py`) |
| Routes | `http/api_controller.hpp`, `http/spa_controller.hpp`, `http/views.{hpp,cpp}`, `http/http_util.{hpp,cpp}`, `http/app_state.hpp` | all MVP endpoints + SPA catch-all + response shaping (`_card`/`_build_turns`/`_models_event`/`render_markdown` ports) |
| Entry | `main.cpp` | wires collaborators, `reconcile_startup` requeue, background model warm, oat++ server |
| Tests | `tests/test_{broker,jobs,config,url,downloader,keyring,translate,oauth,drive}.cpp` | Catch2 + CTest under ASan/UBSan (61 cases) |

**Routes implemented** (route-for-route per `api-reference.md`): `GET /healthz`;
sessions `GET/POST /api/sessions` (multipart upload → temp spool → duration probe →
enqueue), `GET /api/sessions/{id}`, `rename`, `delete`; transcript `turns/{i}`,
`undo`; speakers `GET/POST speakers`, `turns/{i}/speaker`; `GET /api/models`,
`POST /api/models/active`, `POST /api/device` (cpu-only); `GET/POST /api/settings`,
`GET/POST /api/onboarding`, `verify`; binary `audio`, `download/{fmt}`, `export.md`;
translation `POST /api/sessions/{id}/translate`, `GET /api/sessions/{id}/translation/{lang}`,
`GET /sessions/{id}/translation/{lang}/download/{fmt}`; settings `POST /api/settings/google-key`
+ `/clear`, `POST /api/settings/translation-service`; backup `GET /api/backup/status`
(real link state), `POST /api/backup/connect` (starts the consent flow),
`POST /api/backup/disconnect`, `GET /oauth/callback` (loopback redirect target +
success/error page); SSE `sessions/{id}/events`, `sessions/{id}/translate/events`,
`models/events`, `backup/events` (connecting→linked/idle); SPA catch-all + `/static/*`.

### Verification done
- `whisperx_server` + `whisperx_server_tests` build clean (`WHISPERX_BUILD_SERVER=ON`).
- **61 Catch2 cases green** under ASan/UBSan (broker + jobs + config + url +
  downloader URL/map/cache-hit + keyring env-override/roundtrip + translate
  overlay/v1-legacy/stale + key-verify/translate guards + **oauth** PKCE-RFC7636-vector/
  creds-roundtrip/authorize-url/refresh-token-retention/form-encode + **drive**
  query-encoding/multipart-body/pagination/`q`-escaping/ensure-folder/error-mapping).
- Backup-auth live smoke (no real Google): unconfigured → `/api/backup/status`
  `configured:false`, `connect` → 400; with dummy creds → `connecting:true`, status
  flips to `connecting`, `/oauth/callback` fulfils the flow and the **CSRF state
  check rejects a mismatched `state`**, success/error HTML served. The real consent
  (real Desktop-app creds + a human in the browser) is the one manual step.
- Live smoke test: `/healthz`, `/api/models` (correct `ModelStatus`, device `cpu`),
  `/api/sessions`, `/api/settings`, SPA catch-all (500 "not built"), reserved-root
  404 — all correct; model-warm failure with no local assets is logged, not fatal.
- Translation live smoke (seeded done-session + es overlay, no real Google call):
  `settings` exposes the google service + 14 languages + `google_key.key_set`;
  `translation-service` rejects non-google (400) / accepts google; `google-key`
  set rejects empty (400) + clear; `view_translation` joins turns/segments with the
  translated text (speakers preserved); `download/{srt,vtt,txt,json}` render via the
  native writers; editing a turn flips that segment to `stale` with the source
  fallback; missing session/lang/fmt → 404; `translate` with no key → 400; the
  `translate/events` SSE emits the durable `{translations:{}}` map on connect. The
  live Google API call (real key + network) is the one manual step, as in the oracle.

## Deferred / not yet done

- **Task 8 — full e2e parity gate.** The transcribe→edit→export flow + SPA e2e
  (Flask vs oat++) on a seeded clip + a no-Python-on-PATH runtime proof. The downloader
  now resolves align/diarize assets, so a clean clip can run without pre-fetched dirs.
- **Backup engine / restore.** The OAuth link (PKCE) + the Drive `files.*` client
  are **built** (above); what's left is the engine on top — the `manifest.json`
  pointer, content-addressed `objects/<sha256>` store (`put/has/get/list/
  delete_object`), GC, and restore (port of `app/backup/{backend,manifest}.py` +
  the sync loop). `BackupService::drive()` hands the engine a token-bound
  `DriveClient`. Note: `create_file`/`update_content` send **in-memory** bodies —
  large object blobs will want streaming/resumable uploads. A local-filesystem
  backend is also not yet ported.
- **GPU / Metal** — CPU-only; ORT CUDA EP and whisper.cpp/GGML are later passes.
- **Packaging** — lean binary + SPA assets; Tauri/macOS packager repoint to the C++
  server port. Not started.

## Build & run (dev)

```bash
# build (sherpa-onnx is already vendored under build/ from the engine-core work)
cmake -S . -B build -G Ninja -DWHISPERX_CORE_AUDIO=ON -DWHISPERX_BUILD_SERVER=ON
cmake --build build --target whisperx_server whisperx_server_tests
./build/adapters/server/whisperx_server_tests        # 61 cases, ASan/UBSan

# run — assets now lazy-download from the public mirrors (no env vars required);
# the WHISPERX_*_DIR vars below are optional and only override the downloader for dev.
export LD_LIBRARY_PATH="build/lib:build/_deps/oatpp-build/src:\
$(dirname $(find build/_deps -name libonnxruntime.so | head -1)):$LD_LIBRARY_PATH"
export WHISPERX_DATA_DIR=/tmp/whisperx
export WHISPERX_MODEL=tiny WHISPERX_PORT=8000
# optional overrides (skip to let the downloader fetch into WHISPERX_SHERPA_CACHE):
#   WHISPERX_SHERPA_WHISPER_DIR / WHISPERX_ALIGN_ONNX_DIR / WHISPERX_DIARIZE_ONNX_DIR
export WHISPERX_SILERO_ONNX=models/silero_vad.onnx
./build/whisperx_server
```

Relevant env: `WHISPERX_DATA_DIR`, `WHISPERX_PORT`/`WHISPERX_HOST`,
`WHISPERX_LOG_LEVEL`, `WHISPERX_MAX_UPLOAD_MB`, `WHISPERX_MAX_AUDIO_HOURS`,
`WHISPERX_MODEL`, `WHISPERX_SPA_DIR`/`WHISPERX_STATIC_DIR`, and the asset-dir vars
(`WHISPERX_SHERPA_MODELS_ROOT`/`WHISPERX_SHERPA_WHISPER_DIR`,
`WHISPERX_ALIGN_ONNX_ROOT`/`WHISPERX_ALIGN_ONNX_DIR`, `WHISPERX_DIARIZE_ONNX_DIR`,
`WHISPERX_SILERO_ONNX`).
