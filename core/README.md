# WhisperX C++ engine core

Phase-0 scaffold for the C++ reimplementation of the WhisperX pipeline
(ONNX Runtime + sherpa-onnx). Design + roadmap live in
[`../docs/cpp-core-handoff.md`](../docs/cpp-core-handoff.md) and the briefs it
links. This directory is `libwhisperx` — **pure engine code, no transport/UI**.

## Layout

```
core/          this library — pure, ported algorithms (text/, build_info, …)
  text/        edit_distance (WER/CER) — first ported algorithm
  tests/       Catch2 unit tests (CTest, ASan/UBSan)
adapters/py/   pybind11 module `whisperx_core` — the Python parity oracle
bindings/test/ Python parity tests (C++ result == Python reference)
vcpkg.json     production dependency manifest (ORT, sherpa-onnx, …)
CMakeLists.txt at the repo root, drives all of the above
```

As stages land they fill in `core/{audio,vad,asr,align,diarize,schema,pipeline}/`
per the architecture doc; each is exposed through `adapters/py` and diffed against
the live Python pipeline.

## Build & test

Light deps (pybind11, Catch2, nlohmann/json) are pulled by CMake **FetchContent**,
so a fresh checkout builds with no extra setup:

```bash
# configure against the project's (uv-managed) interpreter
cmake -S . -B build -G Ninja \
  -DPython_EXECUTABLE="$(uv run python -c 'import sys;print(sys.executable)')"
cmake --build build

ctest --test-dir build --output-on-failure          # Catch2 under ASan/UBSan
PYTHONPATH=build uv run --no-project --with pytest \
  pytest bindings/test/test_core_parity.py -v        # Python parity oracle
```

The pybind module lands at `build/whisperx_core.cpython-*.so`; put `build/` on
`PYTHONPATH` (or install it) to `import whisperx_core`.

### Production toolchain (vcpkg)

`vcpkg.json` is the source of truth for the heavy runtime deps (ONNX Runtime,
SQLiteCpp, Eigen, Google Benchmark; sherpa-onnx is vendored until it lands in the
registry). Pass the vcpkg toolchain file and the same `find_package()` calls
resolve from vcpkg instead of FetchContent:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

## Options

| CMake option | Default | Effect |
|---|---|---|
| `WHISPERX_CORE_SANITIZE` | `ON` | ASan + UBSan on the native test target |
| `WHISPERX_CORE_PYMODULE` | `ON` | build the pybind11 module |
| `WHISPERX_CORE_TESTS` | `ON` | build the Catch2 suite |

Sanitizers are applied to `whisperx_core_tests` only; the pybind module is kept
clean (ASan inside a dlopened Python extension needs `libasan` `LD_PRELOAD`ed).

CI: [`.github/workflows/cpp-core.yml`](../.github/workflows/cpp-core.yml).
