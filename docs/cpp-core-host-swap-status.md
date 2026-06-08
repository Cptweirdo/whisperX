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
- **Scope:** Core MVP. Backup (Google Drive OAuth + Drive REST) and translation are
  **deferred** (the server returns SPA-compatible stub shapes so the UI degrades).
- **Model assets:** a libcurl HF-mirror downloader is the plan; **not yet built** —
  v1 resolves models from local directories via env vars (below).
- **Secrets:** native OS keyring is the plan; **not yet built**.
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
| Models | `models/model_manager.{hpp,cpp}`, `models/assets.{hpp,cpp}` | CPU/sherpa-only resident-engine manager + `ModelStatus` shape; local-dir asset resolver |
| Routes | `http/api_controller.hpp`, `http/spa_controller.hpp`, `http/views.{hpp,cpp}`, `http/http_util.{hpp,cpp}`, `http/app_state.hpp` | all MVP endpoints + SPA catch-all + response shaping (`_card`/`_build_turns`/`_models_event`/`render_markdown` ports) |
| Entry | `main.cpp` | wires collaborators, `reconcile_startup` requeue, background model warm, oat++ server |
| Tests | `tests/test_broker.cpp`, `tests/test_jobs.cpp`, `tests/test_config.cpp` | Catch2 + CTest under ASan/UBSan |

**Routes implemented** (route-for-route per `api-reference.md`): `GET /healthz`;
sessions `GET/POST /api/sessions` (multipart upload → temp spool → duration probe →
enqueue), `GET /api/sessions/{id}`, `rename`, `delete`; transcript `turns/{i}`,
`undo`; speakers `GET/POST speakers`, `turns/{i}/speaker`; `GET /api/models`,
`POST /api/models/active`, `POST /api/device` (cpu-only); `GET/POST /api/settings`,
`GET/POST /api/onboarding`, `verify`; binary `audio`, `download/{fmt}`, `export.md`;
SSE `sessions/{id}/events`, `models/events`; SPA catch-all + `/static/*`.

### Verification done
- `whisperx_server` + `whisperx_server_tests` build clean (`WHISPERX_BUILD_SERVER=ON`).
- **21 Catch2 cases green** under ASan/UBSan (broker 7 + jobs 4 + config 10).
- Live smoke test: `/healthz`, `/api/models` (correct `ModelStatus`, device `cpu`),
  `/api/sessions`, `/api/settings`, SPA catch-all (500 "not built"), reserved-root
  404 — all correct; model-warm failure with no local assets is logged, not fatal.

## Deferred / not yet done

- **Task 7 — OS keyring + libcurl HF downloader.** Secret storage (HF token / Google
  key / gdrive creds) and lazy model download are stubbed. Onboarding `verify` and
  `hf-token` storage return clear "not available in this build" notices; model assets
  load from local dirs only (env vars below).
- **Task 8 — full e2e parity gate.** The transcribe→edit→export flow + SPA e2e
  (Flask vs oat++) on a seeded clip. Needs the align ONNX model present locally
  (`whisper-tiny/` is cached at the repo root; align/diarize need the downloader or
  pre-fetched dirs).
- **Translation** (Google Translate v2 + overlay) — stubbed: `/api/.../translate`
  returns 400 "not available"; settings expose empty translation services/languages.
- **Backup / restore** (Google Drive OAuth + Drive REST + local backend) — stubbed:
  `/api/backup/status` returns an idle/unlinked `BackupStatus`.
- **GPU / Metal** — CPU-only; ORT CUDA EP and whisper.cpp/GGML are later passes.
- **Packaging** — lean binary + SPA assets; Tauri/macOS packager repoint to the C++
  server port. Not started.

## Build & run (dev)

```bash
# build (sherpa-onnx is already vendored under build/ from the engine-core work)
cmake -S . -B build -G Ninja -DWHISPERX_CORE_AUDIO=ON -DWHISPERX_BUILD_SERVER=ON
cmake --build build --target whisperx_server whisperx_server_tests
./build/adapters/server/whisperx_server_tests        # 21 cases, ASan/UBSan

# run (point the asset env vars at local model dirs until the downloader lands)
export LD_LIBRARY_PATH="build/lib:build/_deps/oatpp-build/src:\
$(dirname $(find build/_deps -name libonnxruntime.so | head -1)):$LD_LIBRARY_PATH"
export WHISPERX_DATA_DIR=/tmp/whisperx
export WHISPERX_SHERPA_WHISPER_DIR=whisper-tiny/sherpa-onnx-whisper-tiny   # cached
export WHISPERX_DIARIZE_ONNX_DIR=diarize-onnx                              # cached
export WHISPERX_ALIGN_ONNX_DIR=<dir with model.onnx + meta.json>           # needs fetch
export WHISPERX_SILERO_ONNX=models/silero_vad.onnx
export WHISPERX_MODEL=tiny WHISPERX_PORT=8000
./build/whisperx_server
```

Relevant env: `WHISPERX_DATA_DIR`, `WHISPERX_PORT`/`WHISPERX_HOST`,
`WHISPERX_LOG_LEVEL`, `WHISPERX_MAX_UPLOAD_MB`, `WHISPERX_MAX_AUDIO_HOURS`,
`WHISPERX_MODEL`, `WHISPERX_SPA_DIR`/`WHISPERX_STATIC_DIR`, and the asset-dir vars
(`WHISPERX_SHERPA_MODELS_ROOT`/`WHISPERX_SHERPA_WHISPER_DIR`,
`WHISPERX_ALIGN_ONNX_ROOT`/`WHISPERX_ALIGN_ONNX_DIR`, `WHISPERX_DIARIZE_ONNX_DIR`,
`WHISPERX_SILERO_ONNX`).
