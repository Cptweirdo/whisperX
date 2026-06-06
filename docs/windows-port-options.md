# Windows x64 Port of WhisperX — Options & Decision Doc

> Companion to [`android-port-options.md`](./android-port-options.md) and
> [`ios-port-options.md`](./ios-port-options.md). **Read this one differently:**
> the mobile docs are about *reimplementing* WhisperX because its native stack
> doesn't exist on-device. **Windows x64 is the opposite** — the real Python
> stack runs natively, so this is a **packaging & GPU-distribution** problem, not
> a port in the mobile sense.

## Context

We want WhisperX usable on **Windows x64** — ideally as a double-clickable
desktop app for non-technical users, with all four stages (VAD → transcribe →
align → diarize) running locally.

**The engine already runs on Windows x64 today.** `pyproject.toml` treats Windows
as first-class:

- `torch`/`torchaudio` resolve to the **CUDA 12.8 (`cu128`) index** on `AMD64`
  (`pyproject.toml:73,78`) — i.e. **real NVIDIA GPU acceleration**, unlike the
  Mac build which is CPU/MPS only.
- `torchcodec` is included for `win32` (`pyproject.toml:23,66`).
- `triton` is correctly excluded (Linux-only; Windows doesn't need it).
- `pywhispercpp` (whisper.cpp backend) builds cross-platform.
- ffmpeg, CTranslate2, pyannote all have Windows x64 builds.

So `uv sync` + `whisperx ...` or the `app/` Flask server **work on Windows now**
for developers. There is **no "can't run the models" blocker** like on mobile.

**What's actually missing:** (1) a redistributable, signed Windows app a
non-technical user can install without Python/CUDA setup, and (2) a coherent
**GPU story** (NVIDIA CUDA vs CPU vs non-NVIDIA) including the well-known
faster-whisper/CTranslate2 **cuDNN-matching** pain on Windows.

### We already have a reference: the macOS packaging pipeline

[`MACOS_INSTALLER.md`](../MACOS_INSTALLER.md) + `packaging/macos/` implement and
validate a self-contained desktop app: **design (B)** = embedded
`python-build-standalone` interpreter + pre-resolved venv + bundled ffmpeg, with
**design (D)** = a Tauri shell over the existing `app/` Flask server, state kept
**outside** the bundle so updates don't destroy it. **The Windows port is
overwhelmingly "re-run that same design with Windows substitutions."**

### macOS → Windows substitution map

| macOS (built) | Windows equivalent |
|---|---|
| Tauri **WKWebView** shell | Tauri **WebView2** shell (evergreen runtime; bundler offers bootstrapper/offline/fixed modes) |
| **DMG** (drag-to-Applications) | **MSI** (WiX) or **NSIS** `.exe` installer — both built by Tauri's bundler |
| `codesign` + **notarize + staple** | **Authenticode** (`signtool`) + **SmartScreen reputation** (see GPU/Signing notes) |
| Hardened runtime + `disable-library-validation` (landmine 1) | **N/A** — Windows has no equivalent; *simpler* |
| ATS exception for `http://127.0.0.1` (landmine 7) | **N/A** — WebView2 loads localhost HTTP fine |
| macOS **Keychain** via `keyring` | **Windows Credential Manager** via `keyring` — `app/secret_store.py` is **already cross-platform**, no change |
| `~/Library/Application Support/WhisperX` | `%LOCALAPPDATA%\WhisperX` — **add a Windows branch to `app/paths.py::data_dir()`** |
| `~/.cache/huggingface` (weights, survives update) | `%USERPROFILE%\.cache\huggingface` (HF default on Windows) — same survive-update story |
| `Contents/Resources/bin/ffmpeg` (signed) | `bin\ffmpeg.exe` (Windows static build) on the child's `PATH` |
| venv `bin/`, `pyvenv.cfg home` relocation (finding 1) | venv `Scripts\`; same "run base interpreter + venv `site-packages` on `PYTHONPATH`" relocation workaround |
| ASR backends: `whispercpp` + `mlx` | `mlx` is **Apple-only**; Windows ships **faster-whisper (CUDA)** as default + optional `whispercpp` |

The code touch-points are tiny: a Windows branch in **`app/paths.py`**, a Windows
**Tauri shell** crate (mirror `packaging/macos/tauri/src/main.rs` — pick a free
`127.0.0.1` port → `PORT`, spawn the interpreter, poll `/healthz`, navigate), and
a **`packaging/windows/`** build driver mirroring `packaging/macos/build.py`.
`app/secret_store.py`, `app/store.py`, `app/jobs.py`, `/healthz`, graceful
shutdown — all already platform-neutral.

---

## The net-new Windows decision: the GPU/installer matrix

This is the part with no macOS precedent and it drives bundle size and support load.

- **NVIDIA (CUDA 12.8)** — the default `cu128` wheel. Real GPU speedup, but the
  **biggest bundle (~2.5–3.5 GB)** and the classic Windows landmine:
  **CTranslate2 ≥4.5 needs CUDA 12 + cuDNN 9**, and its cuDNN expectations are
  *separate* from the cuDNN PyTorch bundles — this is the #1 faster-whisper
  Windows GPU failure mode (see `CUDNN_TROUBLESHOOTING.md`). Bundling a known-good
  cuDNN 9 set (à la Purfview's whisper-standalone-win) into the venv removes the
  user's setup burden.
- **CPU-only** — ship the **`cpu` torch index** instead. Much smaller, runs on any
  Win10/11 x64 box, no CUDA/cuDNN matching. Slower, obviously.
- **Non-NVIDIA GPU (AMD/Intel)** — torch has **no good Windows GPU path** (no
  ROCm on Windows). If we care, the route is the **whisper.cpp `Vulkan`** backend
  for ASR (`asr_whispercpp.py`) while align/diarize stay torch-CPU — a partial
  speedup, not full-pipeline GPU. (DirectML via onnxruntime-directml would mean
  reworking the engine — out of scope.)

**Recommendation: ship two installer flavors — "CUDA (NVIDIA)" and "CPU"** —
selected on the download page. `app/pipeline.py` already auto-detects the device,
so a single codebase backs both; the difference is purely which torch index the
bundled venv was resolved from. (A single CUDA build that *falls back* to CPU also
works but forces ~3 GB on CPU-only users.)

---

## The options

### Option A — Mirror macOS (B)+(D): embedded interpreter + venv + Tauri WebView2, packaged as signed MSI/NSIS  ✅ Recommended

`python-build-standalone` (Windows MSVC build) + pre-resolved venv (torch +
whisperx + faster-whisper [+ `whispercpp`]) + bundled `ffmpeg.exe`, fronted by a
Tauri WebView2 shell over the existing `app/` server, wrapped in a WiX/NSIS
installer and **Authenticode-signed**.

**Pros:** Maximum reuse of an **already-validated** design and of `app/`
verbatim; real interpreter (no freezer guesswork with whisperx's lazy imports /
pyannote+nltk+VAD data files); Windows is *simpler* than macOS here (no hardened
runtime, no notarization, no ATS, no library-validation); native GPU on NVIDIA.
**Cons:** ~3 GB (CUDA flavor); a Windows Tauri crate + `packaging/windows/`
build driver to write (mechanical mirror of the macOS ones); SmartScreen
reputation must accrue (below); cuDNN-matching must be baked correctly.

### Option B — PyInstaller / Nuitka frozen app + installer

Freeze the Flask app (or CLI) into a one-folder dist and wrap with Inno
Setup/NSIS.

**Pros:** Familiar Windows recipe; no separate interpreter management. **Note:**
the macOS doc rejected freezing because the freezer **rewrites Mach-O load paths
and invalidates the ad-hoc signatures inside the torch/ctranslate2 wheels** —
**that specific failure does not exist on Windows** (PE/Authenticode doesn't sign
individual wheel DLLs the same way), so PyInstaller is *less* risky here than on
Mac. **Cons:** still brittle with whisperx's **dynamic/lazy imports + data files**
(`hiddenimports`/`collect-all` lists for pyannote, nltk, VAD assets, faster-whisper
assets); large multi-GB CUDA freeze; harder to keep in lockstep with the
`packaging/macos` design. Viable **secondary/fallback**, not the primary.

### Option C — WinGet / `uv` first-run bootstrap (power-user channel)

A tiny installer (or `winget install`) that fetches `python-build-standalone` +
the resolved wheel set on first run (mirror of macOS design (C)).

**Pros:** Tiny download; trivial to publish; great for **developers/power users**.
**Cons:** First run needs network + a multi-GB download → slow, failure-prone for
non-technical users. Good as a *second distribution channel*, not the consumer one.

### Option D — Docker Desktop / WSL2 (reuse the existing image)

The repo already ships a **CPU Docker image** (see `HOSTING.md`); it runs on
Windows via Docker Desktop/WSL2, and NVIDIA GPU works through **WSL2 CUDA
passthrough**.

**Pros:** Zero new packaging; identical to the Linux server path; good for
self-hosters/devs. **Cons:** Requires Docker/WSL2 — not a native double-click
app; GPU passthrough is an advanced setup. **Dev/server audience only.**

### Option E — `pip install whisperx` into the user's Python (status quo)

Already works on Windows x64 today (AMD64 + CUDA 12.8 markers). **Pros:** nothing
to build. **Cons:** assumes Python + CUDA toolkit + cuDNN setup — **developer
audience only**, exactly the burden the installer exists to remove.

---

## Recommendation

**Option A** (mirror the macOS (B)+(D) pipeline into `packaging/windows/`), shipped
as **two Authenticode-signed installer flavors — CUDA (NVIDIA) and CPU** — with
**Option E/C as the developer channels** and **Option D for self-hosters**. This
reuses a validated design and the entire `app/` backend, and confines net-new work
to: a `app/paths.py` Windows branch, a Windows Tauri/WebView2 shell, a
`packaging/windows/` build driver, and getting the **cuDNN 9 + CTranslate2**
bundling right.

### Signing / SmartScreen reality (the macOS "notarization" analogue)

- Use **Authenticode** (`signtool`) on the installer + the shell `.exe`.
- **EV certificates no longer bypass SmartScreen** (changed 2024) — EV and OV now
  build reputation the same way, so **don't pay the EV premium just for SmartScreen**.
- Microsoft's modern, token-free option is **Azure Trusted/Artifact Signing**
  (CI-friendly), functionally equivalent to an OV cert for SmartScreen.
- Expect a **reputation ramp**: even correctly signed, SmartScreen may warn until
  enough clean installs accrue (weeks). Plan messaging/docs for early users.

---

## Suggested sequencing (Windows PoC)

1. **`app/paths.py`** — add `%LOCALAPPDATA%\WhisperX` (honor `WHISPERX_DATA_DIR`
   first, as today). Confirm `keyring` writes to Credential Manager and `app/`
   runs from source on Windows (`/healthz`, port pick, graceful shutdown).
2. **Bundle PoC (no Tauri yet)** — `python-build-standalone` (MSVC) + resolved
   venv (CUDA flavor) + `ffmpeg.exe`; launch `python -m app.server` with
   `WHISPERX_OPEN_BROWSER=1`. **Verify a real GPU transcription end-to-end** —
   this is where the cuDNN/CTranslate2 issue surfaces (smoke-test the model
   import + a real run, not just the window).
3. **CPU flavor** — repeat (2) resolving torch from the `cpu` index; verify it
   runs on a box with no NVIDIA driver.
4. **Tauri WebView2 shell** — mirror `packaging/macos/tauri`; verify localhost
   HTTP loads (no ATS equivalent needed) and child-process SIGTERM-equivalent on
   window close.
5. **Installer + signing** — WiX/NSIS via Tauri bundler; Authenticode-sign;
   confirm install/uninstall preserves `%LOCALAPPDATA%\WhisperX` + HF cache +
   Credential Manager secrets across an upgrade (the design-(B) thesis).

## Verification strategy

- **GPU import + transcription smoke test on every build** — the cuDNN/CTranslate2
  match is the load-bearing Windows risk (analogous to the macOS hardened-runtime
  torch-import smoke test). Test a real ASR+align run, not just app launch.
- **Update preservation** — sessions DB, `sessions/<id>/`, HF weights, and
  Credential Manager secrets must survive an installer upgrade.
- **Clean-VM run** — install the signed build on a fresh Windows VM with no
  Python/CUDA to validate the non-technical path (and observe SmartScreen).
- Existing `uv run pytest tests/` continues to gate the engine; `python-compatibility.yml`
  already covers Windows-relevant Python versions.

---

## Sources (Windows packaging/GPU grounding)

- Tauri Windows installer (NSIS/WiX, WebView2 install modes, sidecar bundling):
  <https://v2.tauri.app/distribute/windows-installer/>
- Windows code signing + SmartScreen reputation (EV no longer bypasses as of 2024;
  Azure Trusted/Artifact Signing):
  <https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation> ·
  <https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options>
- PyTorch CUDA 12.8 Windows wheels + faster-whisper/CTranslate2 cuDNN 9 requirement:
  <https://pypi.org/project/faster-whisper/> ·
  <https://github.com/SYSTRAN/faster-whisper/issues/1086>
- python-build-standalone (Windows MSVC redistributable interpreter):
  <https://github.com/astral-sh/python-build-standalone>
