# Cross-Platform Packaging — Shared vs. Bespoke

> Synthesis doc tying together the per-platform research:
> [`android-port-options.md`](./android-port-options.md),
> [`ios-port-options.md`](./ios-port-options.md),
> [`windows-port-options.md`](./windows-port-options.md), and the already-built
> macOS pipeline ([`MACOS_INSTALLER.md`](../MACOS_INSTALLER.md) /
> `packaging/macos/`). It answers one question: **across all targets, what is
> bespoke per platform and what can we share?**
>
> See also [`single-language-runtime-options.md`](./single-language-runtime-options.md)
> for whether one language (C++, C#, Dart, …) could cover every dependency,
> [`pipeline-reference.md`](./pipeline-reference.md) for the per-stage spec, and
> the two delivery designs: [`flutter-migration.md`](./flutter-migration.md)
> (all-Dart app) and [`cpp-core-spa-architecture.md`](./cpp-core-spa-architecture.md)
> (C++ engine core + SPA/FFI adapters) — the latter's *how* lives in
> [`cpp-core-migration-plan.md`](./cpp-core-migration-plan.md) (strangler-fig
> migration, DB compatibility, test/timing suite, build tooling).

## The key split: two sharing models, not one

The four targets don't share at a single level — they fall into **two tracks**
that share *different things*:

- **Desktop (macOS ✅ built · Windows 📋 planned · Linux/Docker ✅ exists)** — all
  run the **real Python engine**, so they share **actual code**: the whole
  `whisperx/` package *and* the entire `app/` Flask backend. The bespoke bits are
  just the OS shell around an identical core.
- **Mobile (Android · iOS)** — **cannot run Python at all**, so they share a
  **spec + algorithms + a model**, not code. Each reimplements WhisperX's glue
  natively against per-platform inference engines.

So "what we can share" forms three concentric rings.

---

## Ring 1 — Shared by *everything* (all four platforms)

| Shared asset | Where it lives | Desktop uses it… | Mobile uses it… |
|---|---|---|---|
| **Result contract** | `whisperx/schema.py` (`TranscriptionResult`, `AlignedTranscriptionResult`, `SingleWordSegment`, …) | as-is (Python) | as the cross-language data shape to match |
| **Pure algorithms (the real IP)** | Viterbi `get_trellis`/`backtrack`/interpolation (`alignment.py:425-490`); `assign_word_speakers`+`IntervalTree` (`diarize.py`); VAD batching (`vads/`); subtitle formatting (`SubtitlesProcessor.py`, `conjunctions.py`) | run as-is | **ported** to Kotlin/Swift — logic + test vectors are shared |
| **wav2vec2 alignment model** | `DEFAULT_ALIGN_MODELS_*` tables + one export pipeline | torch | one export → ONNX (Android/sherpa), Core ML/MLX (iOS) |
| **Golden test vectors** | dump intermediates (CTC emissions, trellis, word timings, diarization turns) from the Python pipeline | reference impl | every platform validates against them within tolerance |

**The standout:** the **wav2vec2 forced-alignment stage is the one deliverable
that touches every platform and is logically bespoke on none of them.** Build it
once — model export + algorithm + golden tests — and all four consume it. It is
also the *only* custom piece on mobile, so shared effort here pays the most.

---

## Ring 2 — Shared across the *desktop* track (macOS + Windows + Linux)

Nearly everything is common; the validated macOS pipeline is the template.

- **The entire `app/`** — Flask server, SSE broker (`sse.py`), SQLite store
  (`store.py`), job queue (`jobs.py`), backup (`backup/`), and `secret_store.py`
  (already cross-platform via `keyring`).
- **Design (B) bundling** — `python-build-standalone` + pre-resolved venv +
  bundled ffmpeg.
- **Tauri shell *architecture*** — pick a free `127.0.0.1` port → spawn the
  interpreter → poll `/healthz` → navigate → terminate child on close. The Rust
  logic is ~90% identical; only the webview backend and child-termination signal
  swap.
- **`build.py` phase structure** — skeleton → runtime → app → ffmpeg → shell →
  sign → package.
- **`app/paths.py`** is *already the seam* — one `data_dir()` function with per-OS
  branches.

---

## Ring 3 — Bespoke per platform

| | macOS ✅ | Windows 📋 | Android | iOS |
|---|---|---|---|---|
| **Engine** | Python (CPU/MPS) | Python (**CUDA 12.8**/CPU) | **reimplement** (LiteRT) | **reimplement** (Core ML/ANE) |
| **Webview / shell** | WKWebView | **WebView2** | native Android | native SwiftUI |
| **Installer** | DMG | **MSI / NSIS** | APK / AAB | App Store / IPA |
| **Signing** | codesign + **notarize+staple** | **Authenticode + SmartScreen rep** | Play signing | provisioning + App Store review |
| **Secrets** | Keychain | Credential Manager | Keystore | Keychain |
| **Data dir** | `~/Library/Application Support` | `%LOCALAPPDATA%` | app sandbox | app sandbox |
| **Audio in** | ffmpeg CLI | ffmpeg.exe | AudioRecord / MediaCodec | AVFoundation |
| **Diarization** | pyannote (torch) | pyannote (torch) | **sherpa-onnx** | **SpeakerKit / sherpa-onnx** |
| **GPU quirk** | MPS→align-on-CPU workaround | **cuDNN/CTranslate2 match + GPU installer flavors** | NNAPI / GPU delegate | ANE scheduling |
| **OS-specific extras** | hardened runtime, `disable-library-validation`, ATS exception, frozen Team/bundle ID | *none of those* (simpler) | model bundling/download | App Store review constraints |

`keyring` (secrets) and `app/paths.py` (data dir) already absorb two of these rows
on the desktop track — only the *value* differs per OS, not the code.

---

## Two consolidation moves worth deciding early

1. **Mobile: pick `sherpa-onnx` and two ports collapse toward one.** It is the
   *only* engine spanning Android **and** iOS (same ONNX models, Swift + Kotlin
   bindings). Choosing it (iOS Option B / Android Option B) makes VAD + Whisper +
   diarization a single shared native stack — trading some iOS ANE performance for
   roughly halving the mobile work. The alternative (WhisperKit on iOS + LiteRT on
   Android) is faster per-platform but doubles the engine integration. **The
   wav2vec2 alignment port is shared either way.**

2. **Desktop: unify the Tauri crate across macOS + Windows.** Tauri is
   cross-platform by design — one Rust crate with `#[cfg(target_os)]` branches for
   the webview/port/spawn specifics. The macOS build deliberately uses plain
   `cargo build` (not `cargo tauri build`) so `build.py` owns the bundle/deep-sign;
   on Windows the Tauri bundler producing MSI/NSIS is more idiomatic. **Open
   decision:** keep one shared shell crate + per-OS build drivers, or fully
   separate `packaging/<os>/` trees.

---

## Net picture

- **One shared "headless core"** — `schema.py` + the pure algorithms + the
  wav2vec2 export, validated by golden vectors. Consumed by all four.
- **One shared desktop app** — `app/` + design-(B) bundling + a `cfg`'d Tauri
  shell. macOS and Windows differ only at the rim.
- **A thin bespoke rim per OS** — signing, installer, webview, paths, GPU.
- **The mobile ports reuse the core's *spec*, not its *code*** — and
  `sherpa-onnx` is the lever that makes the two mobile rims nearly one.

The cheapest path overall: invest in **Ring 1** (especially the platform-agnostic
wav2vec2 alignment) and the **two consolidation moves**, because that work is
amortized across the most platforms.
