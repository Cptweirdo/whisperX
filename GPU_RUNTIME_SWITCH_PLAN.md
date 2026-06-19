# Runtime CPU⇄GPU Device Switching — whisperX native C++ server

## Context

The native compute server (`adapters/server/`) runs all three inference stages on
CPU, hardcoded. Today `ModelManager` is built on the explicit invariant *"device is
fixed to cpu, no eviction"* (`model_manager.hpp:1-8`), and `/api/device` rejects
anything but `cpu`. We want an in-app lever to move the pipeline onto a GPU at
runtime without restarting the server.

The hard part is **not** the GPU code — every stage already runs under ONNX Runtime.
The hard part is that an ORT/sherpa session bakes its execution provider in at
construction, so switching device means **tearing down and rebuilding every resident
engine**, and the job runner holds **raw references into those caches for the whole
job** → evicting mid-job is a use-after-free.

### Decisions (confirmed with user)

- **Scope = switch architecture only (CPU-testable).** Build the entire switching
  machine + capability detection, but **do not** flip `SHERPA_ONNX_ENABLE_GPU` yet.
  On the CPU-only build, requesting `cuda` returns **400 "not available"**. The
  GPU-ORT build flip + on-hardware spike (sherpa v1.13.2 vs CUDA ORT, real CUDA
  dispatch, Stage-2 EP symbols) is a **separate follow-up** — everything here is pure
  plumbing, fully testable on CPU now.
- **Lifetime = `shared_ptr` borrows.** The runner holds `shared_ptr<engine>` for the
  whole job; `set_device` clears the manager's maps but in-flight jobs keep their
  engine alive until done. UAF-safe regardless of thread races.
- **Busy job = reject 409.** Keep the existing busy-gate; a running job blocks the
  switch, user retries. With `shared_ptr` underneath, the gate is UX-only, not the
  safety mechanism.

---

## Process

- **Store this plan in the project root** as `GPU_RUNTIME_SWITCH_PLAN.md` (next to
  `GPU_INTEGRATION.md`) so it is version-controlled with the work it describes.
- The implementation is sliced into the numbered Steps below. **After completing each
  slice**, stop and: (1) review the full plan against what actually landed, (2) update
  this file — mark the slice done, record deviations, signatures that changed, and any
  new unknowns surfaced — then (3) proceed to the next slice. The plan stays a living
  document, not a one-shot snapshot.

## Implementation

### Step 1 — `shared_ptr` engine refactor (land first; pure refactor, no behavior change) — ✅ DONE

> All 187 C++ tests pass; `whisperx_server` links clean. Landed as planned: caches +
> `AlignHandle`/`AlignEntry` now `shared_ptr`; `dictionary` →
> `shared_ptr<const map<string,int>>` (one `make_shared` from the moved
> `assets->dictionary`). `runner.cpp` holds the shared_ptrs for the job; diarize lambda
> captures by value (`[&, diarizer]`); `main.cpp` warm path `auto d`.
> **Deviations:** none. **Note:** pybind `adapters/py/whisperx_core.cpp:871` has its own
> local `AlignHandle` — separate, untouched.

This is the load-bearing change; everything else sits on it. Ship it standalone with
`device_` still effectively `"cpu"` and prove the full test suite is green.

**`model_manager.hpp`:**
- `asr_`: `map<string, unique_ptr<WhisperSherpa>>` → `map<string, shared_ptr<WhisperSherpa>>`.
- `diarize_`: `unique_ptr<SherpaDiarizer>` → `shared_ptr<SherpaDiarizer>` (keep the
  `diarize_loaded_` + null sentinel).
- `AlignEntry::dictionary` → `shared_ptr<const map<string,int>>`.
- `AlignHandle` (`:38-42`) becomes self-owning value:
  `{ shared_ptr<Wav2Vec2Onnx> model; shared_ptr<const map<string,int>> dictionary; bool batchable; }`.
- Return types: `shared_ptr<WhisperSherpa> load_asr(...)`,
  `shared_ptr<SherpaDiarizer> ensure_diarize()`.

**`model_manager.cpp`:** update `load_asr` (`:80-127`), `ensure_diarize` (`:151-189`),
`align_for` (`:191-221`) to make/return shared_ptr; `status_locked` `loaded` check
unchanged.

**Call sites:**
- `runner.cpp:110` `WhisperSherpa& asr =` → `auto asr =` (deref `asr->...` at `:161,163`).
- `runner.cpp:111` `SherpaDiarizer* diarizer =` → `auto diarizer =`; **capture the
  shared_ptr by value** in the `steps.diarize` lambda so the job owns its ref.
- `runner.cpp:143,178` `AlignHandle` now value-owning; `align_handle` already lives the
  whole job — fine.
- `main.cpp:104-105` warm path → `auto` (shared_ptr).

**Why safe:** `set_device`'s `asr_.clear()` / `diarize_.reset()` (Step 4) drops only
the manager's ref; an in-flight job's `shared_ptr` keeps its engine alive until job end.

### Step 2 — Provider param threaded through engine ctors (mechanical) — ✅ DONE

> All 187 tests pass; full tree builds clean (CPU build = `WHISPERX_GPU_BUILD` off, so
> the CUDA-EP append compiles out via `#else (void)provider;`). `WhisperSherpa` +
> `Wav2Vec2Onnx` ctors now take a trailing `const std::string& provider = "cpu"`;
> `whisper_sherpa.cpp:66` uses `provider.c_str()`; `make_options(int, const string&)`
> appends the CUDA EP only under `WHISPERX_GPU_BUILD`. `SherpaDiarizer` unchanged
> (already had the param). VAD documented CPU-pinned. **Deviations:** none yet — but see
> the `enum class Device` decision recorded under Step 3/4 (engine ctors stay
> string-typed at the `core/` boundary; the enum lives at the server layer).

### Step 2 — Provider param threaded through engine ctors (mechanical)

- **`WhisperSherpa`** (`core/asr/whisper_sherpa.{hpp,cpp}`): add
  `const std::string& provider = "cpu"` (append to ctor `:48`); add owned `std::string
  provider;` on `Impl` (same lifetime discipline as `encoder`/`decoder`); replace the
  hardcoded `config.model_config.provider = "cpu"` (`:66`) with `provider.c_str()`.
- **`Wav2Vec2Onnx`** (`core/align/wav2vec2_onnx.{hpp,cpp}`): add
  `const std::string& provider = "cpu"` to ctor (`:23`); `make_options(int)` →
  `make_options(int, const std::string&)`. The actual CUDA-EP append is **deferred to
  the GPU follow-up** and guarded by a `WHISPERX_GPU_BUILD` compile define — on the CPU
  build the symbol may be absent in the transitively-included `onnxruntime_cxx_api.h`,
  so writing it now would break compilation. Leave the CPU `MemoryInfo` (`:107`) as-is.
- **`SherpaDiarizer`**: no engine change — it already takes `provider="cpu"`
  (`diarize_sherpa.hpp:40`, applied `:29,33,70`). Only its construction site changes.
- **VAD** (`vad_silero.cpp:36`): **deliberately CPU-pinned** (tiny, streamed). Add a
  one-line comment; do not thread device into `silero_segments`.

### Step 3 — `WHISPERX_DEVICE` config knob + boot precedence — ✅ DONE

> All 187 tests pass; tree builds clean.
> **DEVIATION (user request): introduced `enum class Device { Cpu, Cuda }`** instead of
> raw strings. Lives at the **server layer** (`config.hpp`) with `parse_device(string)
> →optional<Device>` (single validation point) + `to_string(Device)` (used for the JSON
> status, the persisted setting, AND the sherpa/ORT provider string — they coincide).
> **Engine ctors stay string-typed** (`core/` must not depend on a server enum; sherpa
> wants a `const char*`); the 3 construction sites pass `to_string(device_)`.
> `Config::device` is now `Device` (parsed from `WHISPERX_DEVICE`); `main.cpp` resolves
> precedence persisted > env > cpu and passes a `Device` to the ctor.
> **Also folded in here** (natural compile boundary, was nominally Step 4):
> `ModelManager` ctor now takes `Device`, stores `device_` (guarded by `lock_`), threads
> `to_string(device_)` into all three engine construction sites, and `status()` reports
> the real `device_`. Runtime `set_device()` + eviction + `num_threads` bump remain in
> Step 4; `cuda_available` stays hardcoded `false` until Step 6.

- `config.hpp:33-49`: add `std::string device = "cpu"; // WHISPERX_DEVICE (cpu|cuda)`.
- `config.cpp` `load_config()`: `c.device = env_str("WHISPERX_DEVICE", "cpu");`.
- `main.cpp` (alongside `active` resolution `:82`): mirror the `active_model`
  precedence — **persisted setting wins over env**:
  `store.get_setting("device", cfg.device).value_or(cfg.device)`. Pass into the
  `ModelManager` ctor.

### Step 4 — `ModelManager` device state + `set_device()` — ✅ DONE

> All 187 tests pass; builds clean. `device_`/ctor/threading/status already landed in
> Step 3, so this slice added the runtime switch only.
> - `json set_device(Device dev)` (enum, not string — matches the Step-3 deviation).
> - **Atomic rollback:** extracted private `build_asr_engine(name, dev)` (resolve +
>   construct, no cache mutation; shared by `load_asr`); `set_device` builds the active
>   model on the NEW device **before** evicting — on failure nothing is mutated, manager
>   stays on the old device. Cost: ~2× the active model briefly co-resident (accepted).
> - Eviction under `lock_`: clears `asr_`/`align_`, resets `diarize_`+flags+`errors_`,
>   sets `device_`, pre-warms the rebuilt active ASR. align/diarize rebuild lazily.
> - `load_lock_` held across the whole switch → no load can read `device_` mid-rebuild.
> - **CPU `num_threads` bump:** `threads_for(Device)` → `hardware_concurrency()/2` (≥1)
>   for CPU, `1` for cuda; applied at all three construction sites (was hardcoded `1`).
> - **Deferred:** dedicated UAF/eviction unit test → GPU follow-up (cpu→cpu is a no-op;
>   cpu→cuda needs Slice 6 capability + real assets). The shared_ptr borrow is the
>   structural guarantee; the existing 187 cover the refactor.

**`model_manager.hpp`:** rewrite the stale `:1-8` comment; add `std::string device_;`
(guarded by `lock_`); ctor → `ModelManager(std::string active, std::string device,
OnChange = nullptr)`; add `json set_device(const std::string& dev);` and a private
`bool cuda_available()` (Step 6).

**`model_manager.cpp`:**
- Ctor: validate/store `device_` (`{"cpu","cuda"}`, default `"cpu"`).
- `status_locked` `:66-67`: `{"device", device_}`, `{"cuda_available", cuda_available()}`.
- Pass `device_` (read under `lock_`) to the three construction sites: `load_asr` `:105`
  (+ update the `"on cpu"` log `:103`), `ensure_diarize` `:175`
  (`SherpaDiarizer(seg, embed, num_threads, /*provider=*/device_)`), `align_for` `:214`.
- `num_threads` bump: when `device_=="cpu"`, raise the hardcoded `1` (sites `:106` +
  ctor defaults) to e.g. `max(1u, hardware_concurrency()/2)`; `1` for cuda. Free CPU win.
- `set_device(dev)` — mirror `set_active` (`:139-149`) load/warm/notify pattern, plus
  eviction:
  ```
  validate dev ∈ {cpu,cuda}; if cuda && !cuda_available() throw  // → 400 upstream
  std::lock_guard load(load_lock_);              // blocks concurrent heavy loads
  { std::lock_guard lk(lock_);
    if (dev == device_) return status_locked();  // no-op
    // rollback safety: eager validation load on new device BEFORE evicting
    asr_.clear(); align_.clear();
    diarize_.reset(); diarize_loaded_ = false; diarize_error_.reset();
    loading_.clear(); errors_.clear();
    device_ = dev; }
  warm(active_);            // background reload of active ASR on new device
  notify_change();          // push device/cuda_available to /models/events + SPA
  return status();
  ```
  **Rollback (failure atomicity):** before clearing, attempt one cheap session
  construction on the new provider; on throw, leave `device_`/maps untouched and
  rethrow → endpoint 400s, manager stays fully usable on the old device. (`warm` itself
  is detached + swallows errors into `errors_`, so a later GPU failure can't corrupt
  state.)

### Step 5 — `/api/device` + `onboarding_finish` — ✅ DONE

> All 187 tests pass; **live-server smoke test** on `whisperx_server` confirms every
> path: `{cpu}`→200 (no-op), `{cuda}`→400 "CUDA device not available in this build."
> (+full status body), `{gpu0}`→400 "Unknown device: gpu0", `/api/models` reports
> `device:"cpu"`,`cuda_available:false`. `switch_device` now: `parse_device` (400 on
> unknown) → cuda-availability check via `status()["cuda_available"]` (400) → 409 busy
> gate kept → `try { set_device(*dev); persist on success } catch { 400 + status }`.
> `onboarding_finish(model, Device)` persists `to_string(device)` + calls `set_device`
> in try/catch (`device_error` non-fatal, mirrors the keyring path).
> **Coverage gap:** the existing `tests/test_*_e2e.py` / `test_api.py` target the **old
> Flask app**, not the native C++ server — these endpoints have no automated C++ test;
> verified manually via the live binary. A native controller test harness is a future
> item.
> **Decoupling note:** the endpoint reads `status()["cuda_available"]` rather than a
> public `cuda_available()`, so Step 6 only flips the hardcoded `false` — endpoint
> unchanged.

**`api_controller.hpp:467-488` `switch_device`** — replace the cpu-only branch:
```
device = body.value("device","");
if (device != "cpu" && device != "cuda") return 400 {"Unknown device: "+device};
if (device == "cuda" && !status["cuda_available"]) return 400 {"CUDA not available"};
if (store.has_active_jobs()) return 409 {status + "busy"};   // keep the gate
try { json st = manager.set_device(device);
      store.set_setting("device", device);                   // persist only on success
      return 200 st; }
catch (e) { json st = manager.status(); st["error"]=e.what(); return 400 st; }
```
Wrap in try/catch so no exception escapes onto the request thread (server has no signal
handler → uncaught = fatal).

**`onboarding_finish` (`:578-614`):** accept `cuda` when available (same validation);
`finish_onboarding(model)` → `finish_onboarding(model, device)`: persist chosen device,
call `manager.set_device(device)` (skip if equal), wrapped in try/catch like the
existing keyring-failure handling.

### Step 6 — Capability detection (`cuda_available()`) — ✅ DONE

> All 187 tests pass; CPU build reports `cuda_available:false` (probe compiles out).
> - CMake: `if(SHERPA_ONNX_ENABLE_GPU) add_compile_definitions(WHISPERX_GPU_BUILD)`
>   placed right after the flag (skipped now, propagates when the GPU follow-up flips
>   it ON; reaches the core libs + server declared after).
> - Probe lives in `core/align/wav2vec2_onnx.{hpp,cpp}` as free fn
>   `whisperx::align::ort_cuda_available()` — that TU already owns the direct ORT C++
>   API. Under `WHISPERX_GPU_BUILD` it scans `Ort::GetAvailableProviders()` for
>   `"CUDAExecutionProvider"` (try/catch → false); `#else return false`.
> - `ModelManager::cuda_available()` (static, cached `static const bool`) delegates to
>   it; `status_locked()` now emits `cuda_available()`. The `/api/device` +
>   `onboarding` cuda-availability checks (Step 5) flow through this unchanged.

Replace hardcoded `false` (`model_manager.cpp:67`):
1. CMake sets `target_compile_definitions(... WHISPERX_GPU_BUILD)` **only** when
   `SHERPA_ONNX_ENABLE_GPU` is ON. On the current CPU build the define is absent →
   `cuda_available()` returns `false` immediately, so `cuda` honestly 400s.
2. When `WHISPERX_GPU_BUILD` is defined (the follow-up), probe
   `Ort::GetAvailableProviders()` for `"CUDAExecutionProvider"`, cache in a `static`.

Used by both `status_locked()` and `set_device`'s validation — the gate that stops a
session silently falling back to CPU.

---

## Critical files

- `adapters/server/models/model_manager.{hpp,cpp}` — `device_`, `set_device`, eviction,
  shared_ptr maps, real status, capability detection
- `adapters/server/jobs/runner.cpp` — shared_ptr borrows held for job duration (UAF-critical)
- `adapters/server/http/api_controller.hpp` — `/api/device` + `onboarding_finish`
- `core/asr/whisper_sherpa.{hpp,cpp}`, `core/align/wav2vec2_onnx.{hpp,cpp}` — provider ctor param
- `adapters/server/config.{hpp,cpp}`, `adapters/server/main.cpp` — WHISPERX_DEVICE + boot precedence
- `CMakeLists.txt` — `WHISPERX_GPU_BUILD` compile define (gated on the future GPU flag)

## Out of scope (this slice)

- `SHERPA_ONNX_ENABLE_GPU ON` flip + actual CUDA-EP append in `make_options` +
  on-hardware spike → **follow-up** (needs CUDA toolkit + GPU).
- VAD on GPU (deliberately CPU-pinned). `IoBinding` device-memory optimization.

## Verification

- **Refactor (Step 1):** existing job/runner + api tests pass unchanged with device cpu.
- **UAF guard:** unit test — load an ASR, hold the returned `shared_ptr`, force
  eviction via `set_device`, assert the held engine still transcribes. Run a thread
  loop hammering `set_device` against `load_asr`/`align_for` under **ASan** → clean.
- **Endpoints:** `POST /api/device {cpu}` → 200; `{cuda}` on CPU build → 400 "not
  available"; `{bogus}` → 400; busy session → 409; assert `set_setting("device")`
  persisted only on 200. Mirror for onboarding.
- **Boot precedence:** persisted `device` beats `WHISPERX_DEVICE` env beats `"cpu"`.
- **Status/SSE:** `set_device` → `notify_change()` pushes new `device`/`cuda_available`
  to `/models/events` (SPA toggle reads these).
- Build commands: `uv run pytest` for any Python-side API tests; C++ server tests via
  the project's CMake/ctest target.
