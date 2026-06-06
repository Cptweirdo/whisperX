# Single-Language / Unified-Runtime Options

> Companion to [`packaging-shared-vs-bespoke.md`](./packaging-shared-vs-bespoke.md)
> and the per-platform port docs. Answers: **is there one language that can cover
> every dependency — Whisper, wav2vec2 alignment, pyannote diarization, silero VAD,
> audio decode — stably, across all five targets (Android, iOS, Windows, macOS,
> Linux)?** And drills into **how C# in particular works on Android/iOS**, since
> it's the strongest *managed* single-language answer.

## The core truth

**No high-level language has a stable, from-scratch implementation of all these
models.** Whisper, wav2vec2-CTC, pyannote (segmentation + embedding), and silero
VAD all bottom out in the *same* C/C++ inference engines — **ONNX Runtime**,
**GGML/whisper.cpp**, **LiteRT**, **Core ML**. So the real question is not "which
language reimplements the deps" but:

> **Which single language has stable bindings to a runtime that can run all four
> models, on all five platforms, plus the pure-algorithm glue?**

Under that framing, every "single-language" plan is really **"one language + ONNX
Runtime (or GGML/LiteRT) under the hood."** The languages differ only in *how
native* that coverage is and *how vendor-stable* the bindings are.

### The one constant, in every language

**wav2vec2 forced alignment has no off-the-shelf binding in any language.** A
runtime (ORT/LiteRT/sherpa) will run the wav2vec2 *model*, but the Viterbi trellis
+ word-timestamp logic (`alignment.py:425-490`) is yours to write everywhere. No
language choice eliminates this — see [`pipeline-reference.md`](./pipeline-reference.md)
Stage 4. It just changes the language you write it in.

## Candidate ranking

| Language | How it covers the deps | Cross-platform | Stability | Catch |
|---|---|---|---|---|
| **C++** | *It is the runtime layer.* sherpa-onnx (VAD + Whisper + diarization) + ONNX Runtime (wav2vec2 align) + a C audio decoder + your Viterbi = the whole pipeline, natively | All 5, first-class | **Highest** — these libs are C++ | Worst ergonomics; no UI story (pair with Flutter/Qt); manual memory/builds |
| **C# / .NET** | Microsoft owns **both** ONNX Runtime *and* .NET → first-class ORT bindings; Whisper.net; sherpa-onnx C# bindings; **MAUI** UI on all targets; NativeAOT | All 5 (MAUI) | **Very high** — one vendor backs runtime + platform | Runtime weight; MAUI ecosystem smaller than native; audio capture is platform glue |
| **Dart / Flutter** *(current stack)* | `sherpa_onnx` pub package (VAD+ASR+diarize, all 5) + `tflite_flutter`/ORT FFI for the wav2vec2 pass; glue in pure Dart | All 5 | High for bindings; smaller maintainer (k2-fsa) than MS | Not self-sufficient — every model is native-under-FFI; no Dart-native ML runtime |
| **JS / TypeScript** | transformers.js runs Whisper + wav2vec2 + VAD on ORT-Web/WASM; onnxruntime-react-native (mobile); Electron (desktop) | All 5 (RN + Electron) | High (ORT-backed) | WASM/JS perf lower; diarization support thin; heavier shells |
| **Rust** | `ort` + `candle` + `sherpa-rs`; the memory-safe version of the C++ answer | All 5 | High | candle-on-mobile less proven; diarization via bindings/DIY |

## Takeaways

1. **The single most "stable, covers-everything, all-platforms" language is C++** —
   precisely *because* the inference engines are C++. sherpa-onnx alone gives 3 of
   the 4 models in one C++ library across every target; ORT adds the 4th.
   Everything else in the table is *binding to that same C++*.
2. **The strongest *managed* single-language answer is C#** — the only option where
   the *same vendor* (Microsoft) ships the runtime (ONNX Runtime) **and** the
   cross-platform app framework (.NET MAUI). That single-vendor alignment is
   exactly what "stable" buys.
3. **For the current Flutter reality, the pragmatic "one language *you write*" is
   Dart** — but it's "one language for UI + glue, native runtimes under FFI," not
   "Dart covers the deps natively." The `sherpa_onnx` package + the existing LiteRT
   DLL already prove this; the only DIY piece is the alignment Viterbi (in Dart).

---

## How C# works on Android & iOS

C# is the strongest non-C++ single-language candidate, so its mobile story matters.
This is the **Xamarin lineage, now unified into "one .NET"** — target frameworks
`net10.0-android` / `net10.0-ios`, with **.NET MAUI** as the cross-platform UI layer.

### Runtime model (how the C# physically runs)

The runtime today is **Mono** on both platforms (supports JIT + AOT):

- **Android** — C#/IL runs on the Mono runtime embedded in the APK, bridging to the
  Android Java/Kotlin SDK via generated **JNI bindings**. Release builds default to
  **Mono AOT** (IL → native ARM64 at build time) with the Mono **interpreter** as
  fallback. Ships as a normal APK/AAB. *(.NET 10: CoreCLR is an experimental Android
  runtime; .NET 11 makes it the Android release default. Mono is the stable path
  now.)*
- **iOS** — Apple **forbids JIT**, so .NET uses **full AOT**: Mono AOT compiles IL →
  native ARM64 at build time, plus the interpreter for code that can't be AOT'd
  (this satisfies the no-JIT rule). A **NativeAOT** path also exists for
  iOS/Mac Catalyst (fully native, no Mono runtime — smaller/faster). Apple APIs via
  Objective-C bindings. Ships as a normal IPA.

Both cases: managed C# → AOT-compiled to native → bundled runtime + interop bridge
to OS APIs. **The same architectural shape as the Flutter stack** (Dart AOT + FFI),
just Mono / P-Invoke instead.

### How the models run there (with hardware acceleration)

C# is **not** CPU-bound on mobile. The **ONNX Runtime NuGet ships native Android +
iOS binaries** (mobile support since ORT 1.10) and dispatches to on-device
accelerators via **execution providers**:

- **Android** → **NNAPI** (+ XNNPACK, + QNN for Qualcomm NPUs, + CPU)
- **iOS** → **Core ML EP** → the **Apple Neural Engine / GPU** (+ XNNPACK + CPU)

The Core ML EP is the key fact: a wav2vec2 or Whisper ONNX model run from C# on
iPhone hits the **same ANE WhisperKit uses** — no acceleration lost by being in a
managed language. The native ORT `.so`/`.dylib` is bundled; C# P/Invokes into it, so
it's one C# API. **sherpa-onnx** also has C# bindings + Android/iOS support, so VAD +
Whisper + diarization are reachable from C# too — i.e. all four models, accelerated,
on both platforms.

### Frictions to plan for

- **App size** — Mono runtime + ORT native libs add weight; trimming / NativeAOT
  mitigate.
- **iOS AOT constraints** — reflection-heavy / dynamic-codegen libraries can
  struggle, but ORT is native (P/Invoke), so the inference path is unaffected; the
  interpreter fallback covers edge cases.
- **Audio capture isn't abstracted** — MAUI won't give mic capture/decode for free;
  write platform glue (AVAudioEngine / AudioRecord) via bindings or a plugin. (Same
  as Flutter.)
- **Ecosystem** — Microsoft-backed and real, but smaller than native Kotlin/Swift;
  fine for a transcription tool.

### Verdict for this project

Mechanically, C#/.NET on mobile works **the same way the current Flutter+FFI stack
does**: managed language AOT-compiled to native, calling a native ONNX/inference lib
through interop, accelerated via Core ML/NNAPI. The ORT Core ML EP ≈ what
`sherpa_onnx`/LiteRT already provide under Flutter.

So C# is a **fully viable, vendor-stable** option — but for this project it's a
**lateral move, not an upgrade**: the Flutter + LiteRT stack already runs on the same
architectural pattern. **Pick C# only if you specifically want the .NET / ONNX-Runtime
ecosystem** (e.g. shared code with a C# backend, or a C#-first team). Otherwise,
staying on Dart/Flutter and reaching the same ORT/Core ML acceleration via
`sherpa_onnx` gets there without redoing the inference integration and UI.

---

## Bottom line

There is no language where these ML deps exist as pure, native, stable libraries
**except C++** (the substrate the runtimes are written in), with **candle-in-Rust**
and **transformers.js-in-JS** as from-scratch re-implementations riding a lower
tensor library. Every managed "single-language" plan is **one language + ONNX Runtime
/ GGML / LiteRT under the hood**. The most vendor-stable expressions of that are
**C++** (native) and **C#** (managed); **Dart** is the right fit for the current
Flutter + FFI reality. And in all of them, **wav2vec2 forced alignment stays a DIY
piece.**

## Sources

- .NET MAUI runtimes & compilation: <https://learn.microsoft.com/en-us/dotnet/maui/deployment/runtimes-compilation?view=net-maui-10.0>
- MAUI NativeAOT on iOS / Mac Catalyst: <https://learn.microsoft.com/en-us/dotnet/maui/deployment/nativeaot?view=net-maui-9.0>
- ONNX Runtime — deploy on mobile (NNAPI/Core ML EPs): <https://onnxruntime.ai/docs/tutorials/mobile/>
- ONNX Runtime — Core ML execution provider: <https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html>
- sherpa-onnx (multi-language bindings incl. C# + Dart, Android/iOS): <https://github.com/k2-fsa/sherpa-onnx>
