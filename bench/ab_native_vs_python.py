#!/usr/bin/env python3
"""Phase-6 A/B (local): native ``whisperx_core.run_job`` vs the Python-staged path.

Both paths use the **identical native engines** (sherpa Whisper + wav2vec2 ONNX +
sherpa diarizer); the only difference is the orchestration:

  * **native**  — one decode-once ``AudioBuffer`` driven through every stage in C++;
    the waveform/segments/emissions never cross the pybind seam between stages.
  * **staged**  — the Python pipeline calls each native primitive separately, so the
    decoded waveform (a numpy array) is marshaled **across the seam on every stage**
    (silero → merge → detect → transcribe → align_run → diarize → assign), each call
    copying it in and copying json out — what ``app/pipeline.py::run_job`` does with
    ``orchestrate`` off.

The wall-time delta is the **data-stays-native** win the migration buys (the seam
marshaling the native orchestrator eliminates; decode-once is structural on top).

Opt-in / local only — needs ``RUN_MIRROR=1``, the ``whisperx_core`` audio build on
``PYTHONPATH``, numpy + huggingface_hub, and the sherpa Whisper / diarize ONNX
(``WHISPERX_SHERPA_WHISPER_DIR`` / ``WHISPERX_DIARIZE_ONNX_DIR`` or the public
releases). It does **not** import ``whisperx`` (no torch/pyannote) — the point is to
measure the C++ engine path, the stack the migration removes.

    RUN_MIRROR=1 PYTHONPATH=build python bench/ab_native_vs_python.py [--clip en_dialog] [--runs 5]
"""
import argparse
import glob
import os
import statistics
import sys
import time
from pathlib import Path

if os.environ.get("RUN_MIRROR") != "1":
    sys.exit("set RUN_MIRROR=1 to run the native-vs-Python A/B (downloads models)")

import numpy as np  # noqa: E402
import whisperx_core as wc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "golden" / "clips"
SR, N30 = 16000, 16000 * 30
ONSET, OFFSET, CHUNK = 0.5, 0.363, 30.0
ALIGN_REPO = os.environ.get("MIRROR_REPO", "KonstantK/wav2vec2-align-onnx")
ALIGN_FOLDER = {"en": "WAV2VEC2_ASR_BASE_960H"}


def _whisper_assets():
    d = os.environ.get("WHISPERX_SHERPA_WHISPER_DIR", "whisper-tiny/sherpa-onnx-whisper-tiny")
    enc = sorted(glob.glob(f"{d}/*encoder*.onnx"), key=lambda p: "int8" in p)[0]
    dec = sorted(glob.glob(f"{d}/*decoder*.onnx"), key=lambda p: "int8" in p)[0]
    tok = glob.glob(f"{d}/*tokens*.txt")[0]
    return enc, dec, tok


def _diarize_assets():
    d = os.environ.get("WHISPERX_DIARIZE_ONNX_DIR", "diarize-onnx")
    seg = glob.glob(f"{d}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx")[0]
    emb = glob.glob(f"{d}/*CAM++.onnx")[0]
    return seg, emb


def _align(lang):
    import huggingface_hub as hf
    import json
    folder = ALIGN_FOLDER[lang]
    onnx = hf.hf_hub_download(ALIGN_REPO, f"{folder}/model.onnx")
    meta = json.loads(Path(hf.hf_hub_download(ALIGN_REPO, f"{folder}/meta.json")).read_text())
    return wc.Wav2Vec2Onnx(onnx), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="en_dialog")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--silero", default="models/silero_vad.onnx")
    args = ap.parse_args()

    enc, dec, tok = _whisper_assets()
    asr = wc.WhisperSherpa(encoder=enc, decoder=dec, tokens=tok, num_threads=4, feature_dim=80)
    seg, emb = _diarize_assets()
    diar = wc.SherpaDiarizer(segmentation=seg, embedding=emb, num_threads=4, provider="cpu")
    model, meta = _align(args.lang)
    wav = str(CLIPS / f"{args.clip}.wav")
    audio = np.ascontiguousarray(wc.load_audio(wav), dtype=np.float32)
    dur = len(audio) / SR

    def native():
        wc.run_job(wav, asr, args.silero, ONSET, OFFSET, CHUNK, args.lang, "transcribe",
                   lambda lang: (model, meta["dictionary"], bool(meta["batchable"])),
                   "nearest", False, diar, 0, None, None)

    def staged():
        a = np.ascontiguousarray(wc.load_audio(wav), dtype=np.float32)   # decode (numpy)
        segs = wc.silero_segments(a, args.silero, SR, ONSET, CHUNK)
        merged = wc.merge_chunks(segs, CHUNK, ONSET, OFFSET)
        spans = [(m["start"], m["end"]) for m in merged]
        lang = asr.detect_language(a[:N30]) if spans else args.lang
        chunks = asr.transcribe(a, spans, lang, "transcribe")
        tr = [{"text": c["text"].strip(), "start": round(s[0], 3), "end": round(s[1], 3),
               "avg_logprob": float(c["avg_logprob"])} for s, c in zip(spans, chunks)]
        res = wc.align_run(tr, model, meta["dictionary"], a, lang, bool(meta["batchable"]),
                           "nearest", False, None)
        raw = diar.diarize(a, 0)
        turns = [(r["start"], r["end"], f"SPEAKER_{r['speaker']:02d}") for r in raw]
        wc.assign_word_speakers(turns, res["segments"], False)

    def bench(fn):
        fn()  # warmup
        ts = []
        for _ in range(args.runs):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return statistics.median(ts)

    nat, stg = bench(native), bench(staged)
    print(f"clip {args.clip}  ({dur:.1f}s, median of {args.runs})")
    print(f"  {'path':<8} {'wall_s':>9} {'RTF':>8}")
    print(f"  {'native':<8} {nat:>9.3f} {nat / dur:>8.4f}")
    print(f"  {'staged':<8} {stg:>9.3f} {stg / dur:>8.4f}")
    print(f"  staged/native = {stg / nat:.3f}x  (seam-marshaling overhead removed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
