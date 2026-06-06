# WhisperX → Flutter Migration Plan

> Detailed plan to deliver WhisperX as a **single Flutter (Dart) app** across
> Android, iOS, Windows, macOS, and Linux, with **native inference via FFI** and
> **all glue in pure Dart**, validated to numerical parity against the Python
> reference. Builds on:
> [`pipeline-reference.md`](./pipeline-reference.md) (the stage spec we port),
> [`single-language-runtime-options.md`](./single-language-runtime-options.md)
> (why Dart + native runtimes), and the per-platform port docs
> ([android](./android-port-options.md) / [ios](./ios-port-options.md) /
> [windows](./windows-port-options.md) / [synthesis](./packaging-shared-vs-bespoke.md)).

## 1. Context & goal

WhisperX today is a **Python library + Flask web app** (`app/`) bound to PyTorch /
CTranslate2 / pyannote and an **ffmpeg subprocess** — none of which run on mobile
(see [`pipeline-reference.md`](./pipeline-reference.md)). The proven on-device
pattern we already have is **Flutter UI + Dart FFI into a compiled native
inference lib** — an earlier LiteRT prototype demonstrated this works and
performs; we now standardize that native lib on **ONNX Runtime (via
`sherpa_onnx`)** for one runtime across all platforms. This plan generalizes the
pattern into one codebase for all five platforms.

**Principles**
- **Dart for everything we author** — UI, orchestration, and all pure-algorithm
  glue (VAD chunk-merge, alignment Viterbi, diarization interval-tree, output
  formatting).
- **Native inference under FFI, but as precompiled dependencies** — `sherpa_onnx`
  (VAD + Whisper + diarization, all on **ONNX Runtime**). No Rust, no
  C++ we maintain (see [`single-language-runtime-options.md`](./single-language-runtime-options.md)).
- **The Python pipeline is the spec, not a dependency** — every ported stage is
  checked against golden vectors dumped from `whisperx`.

## 2. Key decisions to confirm

These shape scope; recommendations given, please confirm in review.

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | **Scope of migration** | (a) Flutter everywhere — replaces the desktop Python/Flask app too; (b) Flutter mobile-only, keep the Python+Tauri desktop | **(a)** — one codebase, kills the torch/CUDA desktop packaging pain. Keep Python desktop only if its richer features are must-have near-term. |
| D2 | **Whisper engine** — ✅ *resolved* | `sherpa_onnx` Whisper on **ONNX Runtime** | **Committed.** One runtime (ORT) covers VAD + ASR + diarization on every platform; LiteRT is dropped. |
| D3 | **App feature set** | (a) simple one-shot transcribe→export; (b) full parity with `app/` (sessions DB, live progress, Drive backup) | Start **(a)**; add session history + live progress in phase 4; treat Drive backup as optional/later. |
| D4 | **Diarization model** | sherpa-onnx pyannote-seg-3.0 + CAM++ (ONNX, ungated mirror) vs full pyannote | **sherpa-onnx** — on-device, no HF gating, all 5 platforms. |

The rest of this plan assumes **D1(a) + D2(ORT) + D3(a)→(b) + D4(sherpa)**.

## 3. Target architecture

```
┌───────────────────────────────────────────────────────────┐
│ Flutter UI (Dart)  — one widget tree, 5 platforms          │
│  screens: import/record · progress · transcript · export   │
├───────────────────────────────────────────────────────────┤
│ Pipeline orchestrator (Dart)  — mirrors transcribe_task()  │
│  loads stages, streams progress, assembles result schema   │
├──────────────┬───────────────────────────┬────────────────┤
│ PURE DART     │ NATIVE (FFI, precompiled) │ PLATFORM APIs   │
│ glue:         │ inference:                │ audio capture:  │
│ • vad merge   │ • sherpa_onnx VAD         │ • record pkg →  │
│ • Viterbi     │ • sherpa_onnx Whisper     │   AVAudioEngine │
│   align       │ • sherpa_onnx diarization │   /AudioRecord  │
│ • interval    │ • wav2vec2 align (ONNX,   │ • file decode → │
│   tree assign │   same ORT instance)      │   platform/wav  │
│ • writers     │                           │                 │
│ • schema      │                           │                 │
└──────────────┴───────────────────────────┴────────────────┘
```

`sherpa_onnx` internally uses ONNX Runtime → **Core ML EP on iOS (ANE)**, **NNAPI
on Android**, CPU/GPU on desktop — so the managed Dart layer still gets hardware
acceleration.

## 4. Stage-by-stage migration map

Each row maps a pipeline stage (from [`pipeline-reference.md`](./pipeline-reference.md))
to its Flutter implementation. **Port = rewrite the pure algorithm in Dart;
Native = call a precompiled lib; Replace = new platform mechanism.**

| Stage | Python source | Flutter implementation | Kind |
|---|---|---|---|
| **1 Audio load** | `audio.py:25` ffmpeg subprocess | `record` pkg (mic) + platform decode (AVAudioFile/MediaExtractor) or wav via sherpa; resample to 16 kHz mono f32 | **Replace** |
| **1 Mel/STFT** | `audio.py:112` `torch.stft` | **sherpa_onnx computes Whisper features internally** → not needed | N/A |
| **2 VAD** | `vads/silero.py`, `vads/pyannote.py` | `sherpa_onnx` VAD (silero) — native | **Native** |
| **2 Chunk merge** | `vads/vad.py:19` `Vad.merge_chunks` + `Binarize` | **Port to Dart** (pure interval math) | **Port** |
| **3 Transcribe** | `asr.py` CTranslate2 | `sherpa_onnx` Whisper (ONNX Runtime) → emit `TranscriptionResult` | **Native** |
| **4 Align — model** | `alignment.py:256` wav2vec2 fwd | wav2vec2 ONNX via `sherpa_onnx`/onnxruntime FFI (per-language model) | **Native** |
| **4 Align — algo** | `alignment.py:425` `get_trellis`, `:455` `backtrack`, `:508` `merge_repeats`, `:298-411` char→word + `interpolate_nans` | **Port to Dart** (the DIY core — see §8) | **Port** |
| **4 Sentence split** | `alignment.py:189` NLTK punkt | Dart sentence splitter (bundled rules) or simple punctuation heuristic | **Port** |
| **5 Diarize — model** | `diarize.py:103` pyannote | `sherpa_onnx` diarization (pyannote-seg + CAM++) | **Native** |
| **5 Assign speakers** | `diarize.py:14` `IntervalTree`, `:185` `assign_word_speakers` | **Port to Dart** (sorted array + binary search) | **Port** |
| **6 Output writers** | `utils.py:215-468` srt/vtt/txt/tsv/json | **Port to Dart** (string formatting) | **Port** |
| **Schema** | `schema.py` TypedDicts | Dart data classes / `freezed` models | **Port** |
| **Orchestration** | `transcribe.py:20` | Dart pipeline service with progress stream | **Port** |

**Net:** 3 stages are off-the-shelf native (`sherpa_onnx`); everything else is
pure-Dart port. The only genuinely novel build is **wav2vec2 alignment** (model +
Viterbi) — sherpa doesn't provide it.

## 5. Dependencies

**Dart / pub.dev**
- `sherpa_onnx` — VAD + Whisper + diarization (+ its platform sub-packages). Core.
- `record` — cross-platform audio capture.
- `just_audio` — playback for transcript review (optional).
- `path_provider`, `path` — app data/model dirs.
- `ffi`, `package:ffi` — custom FFI for the wav2vec2 ONNX pass if run outside sherpa.
- `drift` or `sqflite` — local session DB (phase 4, mirrors `app/store.py`).
- `dio` / `http` + `archive` — model download + unzip.
- `freezed` + `json_serializable` — schema models.

**Native (precompiled, via the above packages)**
- ONNX Runtime (inside `sherpa_onnx`) — Core ML / NNAPI / CPU EPs. The single
  runtime for every model.

**Models (assets or first-run download)**
- Whisper (sherpa ONNX) — size per chosen size class.
- silero VAD ONNX (small).
- pyannote-segmentation-3.0 ONNX + CAM++ embedding ONNX.
- wav2vec2-CTC per language (start with English; map the rest from
  `DEFAULT_ALIGN_MODELS_*` in `alignment.py:32-77`).

> **Audio-decode caveat:** `ffmpeg_kit_flutter` was retired in 2025 — don't depend
> on it. Prefer **platform-native decode** (AVAudioFile / MediaExtractor via a thin
> plugin) or operate on PCM/WAV captured by `record`. Flag compressed-format
> decode (mp3/m4a) as an explicit task.

## 6. Proposed project structure

```
whisperx_app/
├─ lib/
│  ├─ main.dart
│  ├─ ui/                 screens + widgets
│  ├─ pipeline/
│  │   ├─ orchestrator.dart      # mirrors transcribe_task()
│  │   ├─ audio.dart             # capture/decode/resample
│  │   ├─ vad.dart               # sherpa VAD + merge_chunks port
│  │   ├─ transcribe.dart        # sherpa_onnx Whisper backend
│  │   ├─ align/
│  │   │   ├─ model.dart         # wav2vec2 ONNX wrapper
│  │   │   ├─ trellis.dart       # get_trellis + backtrack (ported)
│  │   │   └─ words.dart         # char→word→sentence + interpolation
│  │   ├─ diarize.dart           # sherpa diarization + IntervalTree port
│  │   └─ writers/               # srt/vtt/txt/tsv/json
│  ├─ models/             schema (freezed): SingleWordSegment, …
│  └─ services/           model download, storage, settings
├─ assets/models/         bundled small models (vad, …)
├─ test/                  unit + golden-parity tests
│  └─ golden/             vectors dumped from Python whisperx
├─ android/ ios/ windows/ macos/ linux/   platform projects
└─ pubspec.yaml
```

## 7. Models & assets strategy

- **Bundle** small, always-needed models (silero VAD; optionally English wav2vec2).
- **Download on first use** large/optional models (bigger Whisper sizes, extra
  alignment languages, diarization) with a progress UI; cache under
  `path_provider` app-support dir (mirrors the Python `~/.cache` survive-update
  idea).
- Diarization via sherpa's **ungated** ONNX mirror → no HF token flow on device.
- Keep a manifest (model → url, sha256, size) for verified downloads.

## 8. Golden-parity test strategy (the load-bearing risk)

The ported algorithms must match Python within tolerance. Build a harness:

1. In Python, run `whisperx` on a fixed clip set and **dump intermediates** to JSON:
   VAD segments + merged chunks, CTC emissions/`trellis`, `backtrack` path, final
   word timings, diarization turns, speaker assignment.
2. Commit these under `test/golden/`.
3. Dart unit tests feed the **same inputs** to the ported functions and assert:
   - `merge_chunks` → identical chunk boundaries.
   - `get_trellis`/`backtrack` → identical path (and word start/end within ~1 frame).
   - `IntervalTree.assign` → identical speaker labels.
   - writers → byte-identical srt/vtt/txt/tsv/json.
4. End-to-end: a couple of full clips compared at the `AlignedTranscriptionResult`
   level (word timings within tolerance, speakers exact).

This replaces the "shared code via PyO3" guarantee we gave up by going all-Dart.

## 9. Platform-specific notes

- **Android** — mic permission; `record` → PCM; sherpa uses **NNAPI**; min SDK per
  ORT; ship arm64 (+ optionally armeabi-v7a). Package AAB.
- **iOS** — mic permission (`NSMicrophoneUsageDescription`); **AVAudioEngine** for
  capture; sherpa uses **Core ML EP → ANE**; no JIT concerns (Dart is AOT). App
  Store review for model downloads.
- **Windows/macOS/Linux** — Flutter desktop; sherpa CPU/GPU; **this replaces the
  Tauri+Python desktop** (D1a) → no `python-build-standalone`, no torch/CUDA
  bundling, no cuDNN matching (the bulk of [`windows-port-options.md`](./windows-port-options.md)
  and [`MACOS_INSTALLER.md`](../MACOS_INSTALLER.md) packaging pain disappears).
  Signing still applies (Authenticode / Apple notarization) but over a lean app.

## 10. Migration roadmap

| Phase | Goal | Deliverable / exit criteria |
|---|---|---|
| **0** | Spike | Flutter app loads `sherpa_onnx`, transcribes a bundled WAV on one platform; prints text |
| **1** | Transcribe path | Audio capture/decode → VAD (sherpa) + **merge_chunks port** → Whisper → `TranscriptionResult`; matches Python segment text on golden clips |
| **2** | Alignment | wav2vec2 ONNX (English) + **Viterbi/word port**; golden word-timing parity. *Highest-risk phase — do early.* |
| **3** | Diarization | sherpa diarization + **assign_word_speakers port**; speaker-labelled words on a 2-speaker clip |
| **4** | App UX | Import/record UI, live progress stream, transcript view, **writers port** (export srt/vtt/…); optional session history (drift) |
| **5** | All platforms | Android + iOS + 3 desktop builds green; model download/cache; permissions |
| **6** | Polish/package | Signing, store/installer packaging, error/failure-path UX, more alignment languages |

Sequence rationale: **prove alignment (phase 2) before investing in UI** — it's the
one piece with no off-the-shelf binding and the main technical risk.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| wav2vec2 ONNX export / numerical drift | Validate export early (phase 2); golden CTC-emission tests; pin opset |
| Compressed-audio decode (no ffmpeg_kit) | Platform-native decode plugin; restrict v1 to wav/PCM if needed |
| Dart numeric perf on the Viterbi | Likely fine (dominated by model fwd); if hot, isolate to a `dart:ffi` C kernel later — but don't pre-optimize |
| Diarization quality vs full pyannote | Accept sherpa pyannote-seg+CAM++; A/B on real clips |
| Model bundle size / download UX | Tiered bundle+download with progress + checksums |
| Losing `app/` features (sessions/backup) | Phase 4 reintroduces history; backup deferred/optional (D3) |
| sherpa_onnx maintainer bus-factor | Pin versions; ONNX Runtime is the stable substrate underneath — the wav2vec2 ONNX model and ported glue can run on ORT directly if sherpa stalls |

## 12. Out of scope / open

- Web (Flutter web) target — not in the 5; revisit if needed (transformers.js/ORT-Web exists).
- Real-time streaming transcription — v1 is file/record-then-process; streaming is a later epic.
- Full feature parity with the Flask `app/` (Drive backup, multi-session SSE) — phased/optional per D3.

## 13. Verification

- **Per-stage golden tests** (§8) gate every ported algorithm in CI (`flutter test`).
- **End-to-end** on reference clips: assert `AlignedTranscriptionResult` word
  timings within tolerance and speaker labels exact vs Python.
- **On-device** smoke + latency/RAM benchmarks per platform (esp. iOS ANE vs
  Android NNAPI vs desktop CPU).
- Keep the Python `whisperx` + `tests/` as the **reference oracle** throughout.

## 14. References

- Pipeline spec: [`pipeline-reference.md`](./pipeline-reference.md)
- Language/runtime rationale: [`single-language-runtime-options.md`](./single-language-runtime-options.md)
- Per-platform context: [android](./android-port-options.md) · [ios](./ios-port-options.md) · [windows](./windows-port-options.md) · [synthesis](./packaging-shared-vs-bespoke.md)
- `sherpa_onnx` Flutter (VAD/ASR/diarization, all 5 platforms): <https://pub.dev/packages/sherpa_onnx>
- ONNX Runtime mobile EPs (NNAPI/Core ML): <https://onnxruntime.ai/docs/tutorials/mobile/>
