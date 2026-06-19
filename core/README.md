# WhisperX C++ engine core

`libwhisperx` — the C++ reimplementation of the WhisperX pipeline on
**ONNX Runtime + sherpa-onnx**, replacing the Python `whisperx/` internals **stage
by stage** behind the `whisperx_core` pybind11 module. The Python `app/` stays the
host and parity oracle. Design + roadmap live in
[`../docs/cpp-core-handoff.md`](../docs/cpp-core-handoff.md) and the briefs it links.
This directory is **pure engine code, no transport/UI**.

Status: **Phases 0–5 landed.** Every stage — store, decode, VAD, alignment, ASR,
diarization, writers, and the native decode-once orchestrator — has a C++
implementation, each gated behind a composable `WHISPERX_CORE_STAGES` token. Phase 5
is complete bar two deferred items (e2e Playwright host gate, conditional `std::pmr`
arena — see the handoff).

## The strangler seam

Each Python module is a **facade**: it routes to C++ when its token is in the
`WHISPERX_CORE_STAGES` env var (comma-separated, composable), else runs the
untouched Python body as the oracle. A stage is "done" when it matches Python on the
golden set within the brief's tolerances with its token on.

| Token | Stage | Native code | Facade |
|---|---|---|---|
| `db` | SQLite session store | `core/db/session_store.*` | `app/store.py` |
| `edits` | turn edits / undo / sidecars | `core/edits/edits.*`, `core/text/sequence_matcher.*` | `app/edits.py`, `app/store.py` |
| `vad` | `merge_chunks` | `core/vad/merge_chunks.*` | `whisperx/vads/vad.py` |
| `decode` | in-process ffmpeg decode + ORT silero VAD | `core/audio/decode.*`, `core/audio/vad_silero.*` | `whisperx/audio.py`, `whisperx/vads/silero.py` |
| `align` | Viterbi + char→word→sentence assembly | `core/align/{trellis,align,interpolate}.*`, `core/text/sentence_split.*` | `whisperx/alignment.py` |
| `align_onnx` | wav2vec2 ONNX forward + emission post | `core/align/{wav2vec2_onnx,emission_post}.*` | `whisperx/alignment.py` |
| `align_driver` | native `align()` entrypoint (char-clean + gather + batched forward) | `core/align/{align_driver,char_clean}.*` | `whisperx/alignment.py` |
| `asr` | sherpa-onnx Whisper backend | `core/asr/whisper_sherpa.*` | `whisperx/asr_sherpa.py`, `whisperx/asr.py` |
| `assign` | `assign_word_speakers` + IntervalTree | `core/diarize/{assign_speakers,interval_tree}.*` | `whisperx/diarize.py` |
| `diarize` | sherpa-onnx speaker diarization | `core/diarize/diarize_sherpa.*` | `whisperx/diarize_sherpa.py`, `whisperx/diarize.py` |
| `writers` | srt/vtt/txt/tsv/aud/json output writers | `core/writers/writers.*` | `whisperx/utils.py` |
| `orchestrate` | native decode-once `run_job` sequencer | `core/orchestrate/orchestrate.*` | `app/pipeline.py` |

The full native chain is
`decode,vad,align_onnx,align_driver,asr,assign,diarize,writers,orchestrate`
(plus `db,edits` for the host store). Each is also valid alone or in any subset.

### Decoupled goldens

Whisper text and fp32 tensors are not byte-stable across engines, so parity is
judged the right way per surface: exact `==` for pure algorithms (paths,
char-segments, merged chunks, speaker labels, tokens, writer bytes); `atol`-compare
for fp32 tensors (`emission_atol = 0.006`, timings ±1 frame, scores ±0.01); WER/CER
for ASR; DER-regression for diarization (A/B, not parity — different models). The
CTC emission, the VAD raw segments, and the diarization RTTM turns are dumped as
**fixed-input goldens** so each downstream stage replays torch-free.

## Layout

```
core/
  text/        edit_distance (WER/CER), sequence_matcher (difflib port),
               sentence_split (punkt replacement + Moses prefixes), utf8 cursor
  edits/       app/edits.py port — turn grouping, realign_words, gap interpolation
  db/          SessionStore (SQLiteCpp) — full app/store.py surface
  vad/         merge_chunks
  audio/       in-process libav decode, AudioBuffer, ORT silero VAD, constants
  align/       trellis (Viterbi), align (assembly), char_clean, wav2vec2_onnx
               (raw Ort::Session), emission_post (log_softmax + OOV wildcard),
               align_driver (native align() entrypoint), interpolate
  asr/         whisper_sherpa (sherpa OfflineRecognizer)
  diarize/     interval_tree, assign_speakers, diarize_sherpa (sherpa OfflineSpeakerDiarization)
  writers/     writers (ResultWriter family + iterate_result)
  orchestrate/ orchestrate (pure run_job sequencer, compute injected as std::function Steps)
  tests/       Catch2 unit tests (CTest, ASan/UBSan)
adapters/py/   pybind11 module whisperx_core (bind_* per stage)
bindings/test/ Python parity tests (C++ result == Python oracle)
vcpkg.json     production dependency manifest
CMakeLists.txt at the repo root, drives all of the above
```

### Two static libs

- **`whisperx_core_lib`** — the always-built, **dep-free** pure algorithm IP (text,
  edits, db, vad-merge, the Viterbi/assembly/char-clean/emission-post align pieces,
  diarize glue, writers, orchestrate sequencer). Deps via FetchContent only
  (nlohmann/json, SQLiteCpp, utf8proc). Its ASan/UBSan tests never pull ffmpeg/ORT.
- **`whisperx_core_audio`** — the heavy stage (decode + silero VAD + wav2vec2 ONNX
  forward + align_driver + sherpa Whisper + sherpa diarize), built only under
  `-DWHISPERX_CORE_AUDIO=ON`. Pulls system **ffmpeg** (pkg-config) + a vendored
  **sherpa-onnx** (FetchContent, which brings its own ONNX Runtime).

**Two settled audio build gotchas:** (1) sherpa bundles its own `nlohmann_json` and
unconditionally `add_subdirectory()`s it — a CMake patch guards that line so it
reuses ours. (2) ORT is linked **shared**, not the static archive — the prebuilt
`libonnxruntime.a` is built against an old (glibc2_17) libstdc++ and corrupts the
heap (`free(): invalid pointer` in `std::regex` device discovery) when pulled into a
system-libstdc++ binary; `BUILD_SHARED_LIBS ON` (sherpa scope only) selects the
clean C-API boundary.

### Float parity

Several TUs compile with **`-ffp-contract=off`** (per-TU, GCC/Clang) so arithmetic
matches CPython's never-fused IEEE-754 doubles bit-for-bit where the gate is exact
`==`: `edits.cpp` (gap-interpolation borrow), `align.cpp` (timing ratios),
`emission_post.cpp` (log_softmax vs torch), `assign_speakers.cpp` +
`interval_tree.cpp` (overlap/midpoint sums).

## Build & test

### Dep-free fast lane (default, `WHISPERX_CORE_AUDIO=OFF`)

Light deps (pybind11, Catch2, nlohmann/json, SQLiteCpp, utf8proc) are pulled by
CMake **FetchContent**, so a fresh checkout builds with no extra setup:

```bash
cmake -S . -B build -G Ninja \
  -DPython_EXECUTABLE="$(uv run python -c 'import sys;print(sys.executable)')"
cmake --build build

ctest --test-dir build --output-on-failure          # Catch2 under ASan/UBSan
PYTHONPATH=build uv run --no-project --with pytest \
  pytest bindings/test/test_core_parity.py -v        # a Python parity oracle
```

The pybind module lands at `build/whisperx_core.cpython-*.so`; put `build/` on
`PYTHONPATH` (or install it) to `import whisperx_core`. With `AUDIO=OFF` the module
exposes only the dep-free stages (`db`/`edits`/`vad`/`align`/`assign`/`writers`/
`orchestrate` sequencer); the heavy `decode`/`align_onnx`/`asr`/`diarize` symbols
are `hasattr`-guarded in their facades and absent.

### Audio stage (heavy deps)

```bash
cmake -S . -B build -G Ninja -DWHISPERX_CORE_AUDIO=ON \
  -DPython_EXECUTABLE="$(uv run python -c 'import sys;print(sys.executable)')"
cmake --build build
```

Needs ffmpeg dev libs (`libavformat libavcodec libswresample libavutil` via
pkg-config); sherpa-onnx + ORT are vendored/downloaded by the build.

### Production toolchain (vcpkg)

`vcpkg.json` is the source of truth for the heavy runtime deps. Pass the vcpkg
toolchain file and the same `find_package()` calls resolve from vcpkg instead of
FetchContent:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

### Parity tests

Each stage has a `bindings/test/test_<stage>_parity.py` asserting C++ == the Python
oracle on the goldens. The ones that need the ONNX model mirror (`align_onnx`,
`align_driver`, `asr`, `diarize`, `orchestrate`) are opt-in via `RUN_MIRROR=1` or a
local model dir env var. Run the full host suite across token sets to prove parity:

```bash
WHISPERX_CORE_STAGES=                                   uv run pytest tests/   # oracle
WHISPERX_CORE_STAGES=db,edits,vad,align_onnx,align_driver,asr,assign,diarize,writers,orchestrate \
  uv run pytest tests/                                                          # full native
```

## CMake options

| Option | Default | Effect |
|---|---|---|
| `WHISPERX_CORE_AUDIO` | `OFF` | build the heavy decode/VAD/ONNX/sherpa stage (ffmpeg + sherpa-onnx) |
| `WHISPERX_CORE_SANITIZE` | `ON` | ASan + UBSan on the native test target |
| `WHISPERX_CORE_PYMODULE` | `ON` | build the pybind11 module |
| `WHISPERX_CORE_TESTS` | `ON` | build the Catch2 suite |

Sanitizers are applied to `whisperx_core_tests` only; the pybind module is kept
clean (ASan inside a dlopened Python extension needs `libasan` `LD_PRELOAD`ed).

CI: [`.github/workflows/cpp-core.yml`](../.github/workflows/cpp-core.yml) — a dep-free
lane (build + CTest + parity, no heavy `whisperx` sync) and an `audio-stage` lane
(system ffmpeg + cached sherpa) running the ONNX/sherpa smokes.
