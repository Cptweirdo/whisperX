"""Standalone CUDA whisper-decode bench (CUDA_DECODE_FINDINGS.md harness).

Drives core/asr WhisperSherpa through the pybind module directly — no server,
no VAD, no align/diarize — so decode experiments iterate in minutes. Synthetic
fixed spans (29.5 s tiles) stand in for VAD output; RTF here is comparable to
the server's `stage=transcribing` RTF on the same file (measured baseline
0.443 serial CUDA on the 607 s e2e_long.wav, large-v3-turbo int8).

Usage (from repo root, after `cmake --build build --target whisperx_core`):
  .venv\\Scripts\\python scripts\\bench_cuda_decode.py build\\e2e_long.wav \
      --provider cuda --batch 1 --language en
"""

import argparse
import hashlib
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

if os.name == "nt":
    # Python 3.8+: the pyd's DLL deps (sherpa GPU ORT) resolve via
    # add_dll_directory, not PATH — same dirs bindings/test/conftest.py adds,
    # plus torch/lib for cudnn64_9.dll (same trick as devenv.py's run PATH).
    torch_lib = ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    for d in (BUILD, BUILD / "bin", torch_lib):
        if d.is_dir():
            os.add_dll_directory(str(d))
    # ORT resolves the CUDA EP's cudnn dependency at LoadLibrary time, which
    # ignores add_dll_directory for dependent DLLs — preload it by full path.
    if torch_lib.is_dir():
        import ctypes

        # cudnn64_9 lazily LoadLibrary()s its sub-DLLs (graph/ops/...) by bare
        # name; preload the whole family by absolute path so resolution can't
        # miss (graph first — cudnnCreate needs it).
        for name in sorted(torch_lib.glob("cudnn*_9.dll"),
                           key=lambda p: "graph" not in p.name):
            ctypes.WinDLL(str(name))
sys.path.insert(0, str(BUILD))

import whisperx_core  # noqa: E402

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache"
    / "whisperx-sherpa"
    / "KonstantK"
    / "whisper-onnx-sherpa"
    / "large-v3-turbo"
)
WINDOW_S = 29.5  # kMaxDecodeSamples window the core splits spans into


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as r:
        assert r.getframerate() == 16000, f"need 16 kHz, got {r.getframerate()}"
        assert r.getnchannels() == 1, f"need mono, got {r.getnchannels()}"
        assert r.getsampwidth() == 2, f"need 16-bit, got {r.getsampwidth()}"
        data = r.readframes(r.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def find_assets(model_dir: Path):
    import json

    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    return (
        str(model_dir / meta["encoder"]),
        str(model_dir / meta["decoder"]),
        str(model_dir / meta["tokens"]),
        int(meta.get("feature_dim", 80)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", type=Path)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--encoder", help="override encoder onnx path")
    ap.add_argument("--decoder", help="override decoder onnx path")
    # plain "cpu"/"cuda", or sherpa's "cuda:<config-file>" form (key=value: e.g.
    # ProfilingFilePrefix / LogSeverityLevel). NB the config-file form disables
    # sherpa's whisper IOBinding (IsCudaProvider matches "cuda" exactly).
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--language", default="en")
    ap.add_argument("--num-threads", type=int, default=None,
                    help="default: 1 for cuda (= server threads_for), cpu/2 cores for cpu")
    ap.add_argument("--limit-s", type=float, default=None,
                    help="only decode the first N seconds (quick screen)")
    args = ap.parse_args()

    enc, dec, tok, feat_dim = find_assets(args.model_dir)
    if args.encoder:
        enc = args.encoder
    if args.decoder:
        dec = args.decoder

    if args.num_threads is None:
        args.num_threads = 1 if args.provider == "cuda" else max(1, (os.cpu_count() or 2) // 2)

    audio = load_wav(args.wav)
    dur = len(audio) / 16000.0
    if args.limit_s:
        audio = audio[: int(args.limit_s * 16000)]
        dur = len(audio) / 16000.0

    # Synthetic VAD: tile the file in full decode windows.
    spans = []
    t = 0.0
    while t < dur:
        spans.append((t, min(t + WINDOW_S, dur)))
        t += WINDOW_S

    print(f"file={args.wav} dur={dur:.1f}s spans={len(spans)}")
    print(f"model={Path(enc).name}/{Path(dec).name} feat_dim={feat_dim} "
          f"provider={args.provider} threads={args.num_threads} batch={args.batch} "
          f"language={args.language}")

    t0 = time.perf_counter()
    model = whisperx_core.WhisperSherpa(
        enc, dec, tok,
        num_threads=args.num_threads, feature_dim=feat_dim,
        provider=args.provider, batch_size=args.batch,
    )
    t_load = time.perf_counter() - t0
    print(f"load={t_load:.1f}s")

    t0 = time.perf_counter()
    chunks = model.transcribe(audio, spans, language=args.language)
    wall = time.perf_counter() - t0

    text = " ".join(c["text"] for c in chunks)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    words = len(text.split())
    print(f"\ntranscribe wall={wall:.2f}s rtf={wall / dur:.3f} "
          f"chunks={len(chunks)} words={words} sha1={digest}")
    print(f"text[:200]: {text[:200]}")


if __name__ == "__main__":
    main()
