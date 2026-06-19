"""WER/CER of a sherpa Whisper model against the golden references.

Validation gate for model-variant swaps (int8 → fp32 → fp16, see
CUDA_DECODE_FINDINGS.md): transcribes every golden clip that has a reference
(using its committed VAD spans) and prints per-clip WER/CER with the same
metric as the parity suite (tests/test_baseline_golden.py::wer_cer).
Optionally gates against a baseline WER table (--gate name=wer ... --margin).

Usage:
  .venv\\Scripts\\python scripts\\wer_golden.py \
      --encoder build\\turbo-fp16\\turbo-encoder.fp16.onnx \
      --decoder build\\turbo-fp16\\turbo-decoder.fp16.onnx \
      --tokens build\\turbo-fp32\\turbo-tokens.txt --feature-dim 128 \
      --provider cuda --gate-fp32 --margin 0.05
"""

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

if os.name == "nt":
    torch_lib = ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    for d in (BUILD, BUILD / "bin", torch_lib):
        if d.is_dir():
            os.add_dll_directory(str(d))
    if torch_lib.is_dir():
        for name in sorted(torch_lib.glob("cudnn*_9.dll"),
                           key=lambda p: "graph" not in p.name):
            ctypes.WinDLL(str(name))
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import whisperx_core as wc  # noqa: E402  (import the pyd before whisperx/hf deps)
from test_baseline_golden import wer_cer  # noqa: E402
from whisperx import load_audio  # noqa: E402

# fp32 turbo on CUDA, measured 2026-06-11 (CUDA_DECODE_FINDINGS.md) — the
# reference column the fp16 gate compares against.
FP32_TURBO_WER = {
    "de_dialog": 0.008, "en_dialog": 0.124, "ru_cv_71085": 0.000,
    "ru_cv_71379": 0.100, "ru_cv_71606": 0.167, "ru_cv_79125": 0.000,
    "ru_dialog": 0.122,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--feature-dim", type=int, default=128)
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--gate-fp32", action="store_true",
                    help="fail if any clip exceeds the fp32-turbo baseline + margin")
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()

    base = json.loads((ROOT / "golden" / "baseline.json").read_text(
        encoding="utf-8"))["clips"]
    model = wc.WhisperSherpa(
        encoder=args.encoder, decoder=args.decoder, tokens=args.tokens,
        num_threads=args.num_threads, feature_dim=args.feature_dim,
        provider=args.provider)

    failures = []
    for name, clip in base.items():
        if not clip.get("reference"):
            continue
        vad = json.loads((ROOT / "golden" / "intermediates" /
                          f"{name}.vad.json").read_text(encoding="utf-8"))
        spans = [(c["start"], c["end"]) for c in vad["merged_chunks"]]
        audio = load_audio(str(ROOT / "golden" / "clips" / f"{name}.wav"))
        outs = model.transcribe(audio, spans, clip["lang"], "transcribe")
        hyp = " ".join(o["text"].strip() for o in outs).strip()
        wer, cer = wer_cer(clip["reference"], hyp)
        line = f"{name:16s} lang={clip['lang']}  WER={wer:.3f} CER={cer:.3f}"
        if args.gate_fp32 and name in FP32_TURBO_WER:
            limit = FP32_TURBO_WER[name] + args.margin
            ok = wer <= limit
            line += f"  (limit {limit:.3f}: {'ok' if ok else 'FAIL'})"
            if not ok:
                failures.append(name)
        print(line)

    if failures:
        print(f"\nGATE FAILED: {', '.join(failures)}")
        return 1
    print("\ngate passed" if args.gate_fp32 else "\ndone (no gate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
