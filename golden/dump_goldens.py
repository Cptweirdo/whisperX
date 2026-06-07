#!/usr/bin/env python3
"""Dump per-stage golden intermediates from the live Python WhisperX pipeline.

The end-to-end baseline (`tests/test_baseline_golden.py`) captures timings + WER;
this captures the **tensor / structural intermediates** the C++ core is diffed
against in Phases 2–3 (decoupled-goldens, see docs/cpp-core-migration-briefs.md
"How to read", fact 3). Per clip it records:

  * transcript        — the ASR segments, as a **separate fixed artifact** so the
                        downstream align/writer goldens read it as input (the ASR
                        text itself is judged by WER, never byte-parity).
  * vad               — `Vad.merge_chunks` output (the merged voiced chunks).
  * align             — per segment: the wildcard-extended CTC `emission`, the
                        `get_trellis` matrix, the `backtrack` path, `merge_repeats`
                        char-segments, tokens/blank_id — captured by wrapping the
                        real `whisperx.alignment` functions (zero duplication, the
                        exact values the pipeline computed).
  * words             — final aligned word timings (+ speaker for dialogs).

Tensors (emission/trellis, fp32) go to a compressed ``.npz`` per clip; everything
structural is JSON. A per-run ``intermediates/manifest.json`` pins versions, sha256
of every artifact, and the **starting tolerance budget** (refined once an ORT
export exists to measure real torch-vs-ORT drift).

Writer byte-goldens are **not** dumped here — they already live in
``tests/test_pipeline_contract.py`` (and port straight to Catch2).

Usage::

    uv run python golden/dump_goldens.py                 # all clips, tiny/int8/cpu
    uv run python golden/dump_goldens.py --clips en_libri ru_cv_71085
    uv run python golden/dump_goldens.py --no-diarize    # skip the slow stage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import app.*` resolves when run as a script
GOLDEN = ROOT / "golden"
MANIFEST = GOLDEN / "manifest.json"
OUTDIR = GOLDEN / "intermediates"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lang_of(name: str) -> str:
    return name.split("_", 1)[0]


# --- align spies: wrap the real functions, record what they compute -----------
class AlignCapture:
    """Records the emission/trellis/path/char-segments of every align segment by
    temporarily replacing the module-level alignment functions."""

    def __init__(self):
        self.segments: list[dict] = []
        self._pending: dict | None = None

    def __enter__(self):
        import whisperx.alignment as A
        self._A = A
        self._orig = (A.get_trellis, A.backtrack, A.merge_repeats)

        def trellis_spy(emission, tokens, blank_id=0):
            tr = self._orig[0](emission, tokens, blank_id)
            self._pending = {
                "tokens": [int(t) for t in tokens],
                "blank_id": int(blank_id),
                "emission": emission.cpu().numpy().astype(np.float32).copy(),
                "trellis": tr.cpu().numpy().astype(np.float32).copy(),
            }
            return tr

        def backtrack_spy(trellis, emission, tokens, blank_id=0):
            path = self._orig[1](trellis, emission, tokens, blank_id)
            if self._pending is not None:
                self._pending["path"] = (
                    None if path is None
                    else [[p.token_index, p.time_index, round(float(p.score), 6)]
                          for p in path])
                if path is None:  # backtrack failed -> finalize now
                    self.segments.append(self._pending)
                    self._pending = None
            return path

        def merge_spy(path, transcript):
            segs = self._orig[2](path, transcript)
            if self._pending is not None:
                self._pending["text_clean"] = transcript
                self._pending["char_segments"] = [
                    [s.label, int(s.start), int(s.end), round(float(s.score), 6)]
                    for s in segs]
                self.segments.append(self._pending)
                self._pending = None
            return segs

        A.get_trellis, A.backtrack, A.merge_repeats = (
            trellis_spy, backtrack_spy, merge_spy)
        return self

    def __exit__(self, *exc):
        self._A.get_trellis, self._A.backtrack, self._A.merge_repeats = self._orig


def _wrap_vad(pipe, sink: dict):
    """Shadow the pipeline's merge_chunks to record its output."""
    orig = pipe.vad_model.merge_chunks

    def spy(segments, chunk_size, onset, offset):
        merged = orig(segments, chunk_size, onset, offset)
        # Raw pre-merge VAD segments — the *fixed input* the decoupled merge_chunks
        # parity gate replays through the C++ port (silero ≠ pyannote and torch
        # silero ≠ ORT silero, so the model output can't be byte-compared, but the
        # merge is pure and exactly reproducible from this input).
        sink["segments"] = [
            {"start": round(float(s.start), 4), "end": round(float(s.end), 4),
             "speaker": s.speaker}
            for s in segments]
        sink["merged_chunks"] = [
            {"start": round(float(m["start"]), 4), "end": round(float(m["end"]), 4),
             "n_segments": len(m["segments"])}
            for m in merged]
        sink["params"] = {"chunk_size": chunk_size, "onset": onset, "offset": offset}
        return merged

    pipe.vad_model.merge_chunks = spy


def _dump_clip(name, meta, *, wx, pipe, align_cache, diarizer, outdir):
    lang = _lang_of(name)
    audio_path = GOLDEN / meta["file"]
    audio = wx.load_audio(str(audio_path))
    audio_s = len(audio) / 16000
    multispeaker = (meta.get("n_speakers") or 1) > 1

    # 1) transcribe (capture merged VAD chunks) -> transcript artifact
    vad_sink: dict = {}
    _wrap_vad(pipe, vad_sink)
    tr = pipe.transcribe(audio, batch_size=8, language=lang)
    segments = tr["segments"]
    transcript = {
        "language": tr.get("language", lang),
        "segments": [{"start": round(float(s["start"]), 3),
                      "end": round(float(s["end"]), 3),
                      "text": s["text"].strip()} for s in segments],
    }

    # 2) align (capture emissions/trellis/path/char-segments)
    if lang not in align_cache:
        align_cache[lang] = wx.load_align_model(language_code=lang, device="cpu")
    amodel, ameta = align_cache[lang]
    with AlignCapture() as cap:
        aligned = wx.align(segments, amodel, ameta, audio, "cpu",
                           return_char_alignments=False)

    # 3) diarize (dialogs only) -> per-word speaker labels
    if diarizer is not None and multispeaker:
        diar_df = diarizer(audio)
        wx.assign_word_speakers(diar_df, aligned)

    words = [w for s in aligned["segments"] for w in s.get("words", [])]

    # --- serialize ----------------------------------------------------------
    # Only the CTC `emission` is stored as a tensor — it is the model output, the
    # fp32-epsilon reference the C++/ORT export is diffed against. The `trellis`
    # is deterministically recomputable from emission+tokens (`get_trellis`, a
    # verbatim port), so we keep only its shape; the parity gate downstream of it
    # is the integer `path` + `char_segments`, both stored below.
    tensors: dict[str, np.ndarray] = {}
    align_segs = []
    for i, seg in enumerate(cap.segments):
        emi, tre = seg["emission"], seg["trellis"]
        tensors[f"seg{i}_emission"] = emi
        align_segs.append({
            "text_clean": seg.get("text_clean"),
            "blank_id": seg["blank_id"],
            "tokens": seg["tokens"],
            "emission_shape": list(emi.shape),
            "emission_sha256": hashlib.sha256(emi.tobytes()).hexdigest(),
            "trellis_shape": list(tre.shape),
            "path": seg.get("path"),
            "char_segments": seg.get("char_segments"),
        })

    npz_path = outdir / f"{name}.tensors.npz"
    np.savez_compressed(npz_path, **tensors)

    artifacts = {}

    def _write(suffix, obj):
        p = outdir / f"{name}.{suffix}.json"
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
        artifacts[suffix] = {"file": p.name, "sha256": _sha256(p)}

    _write("transcript", transcript)
    _write("vad", vad_sink)
    _write("align", {"segments": align_segs})
    _write("words", {"words": [
        {k: (round(float(v), 3) if isinstance(v, float) else v)
         for k, v in w.items()} for w in words]})
    artifacts["tensors"] = {"file": npz_path.name, "sha256": _sha256(npz_path),
                            "arrays": sorted(tensors)}

    return {
        "lang": lang,
        "audio_s": round(audio_s, 3),
        "n_transcript_segments": len(segments),
        "n_align_segments": len(align_segs),
        "n_words": len(words),
        "multispeaker": multispeaker,
        "artifacts": artifacts,
    }


def _dump_vad_only(clips, outdir, *, wx, pipe):
    """Surgically (re)write only the raw VAD segments into each ``*.vad.json``.

    The decoupled merge_chunks parity gate needs the raw pre-merge segments as a
    fixed input; the original goldens stored only the merged chunks. This runs
    just the (cheap) silero VAD — no whisper / align / diarize — and augments each
    ``*.vad.json`` in place, asserting the merge of the captured segments still
    reproduces the committed ``merged_chunks`` (a drift guard) before overwriting.
    The pinned tensor goldens and the measured tolerance budget are left untouched.
    """
    from whisperx.audio import SAMPLE_RATE
    from whisperx.vads.vad import Vad

    man = json.loads((outdir / "manifest.json").read_text())
    changed = 0
    for name, meta in sorted(clips.items()):
        vad_path = outdir / f"{name}.vad.json"
        existing = json.loads(vad_path.read_text())
        p = existing["params"]
        audio = wx.load_audio(str(GOLDEN / meta["file"]))
        waveform = pipe.vad_model.preprocess_audio(audio)
        raw = pipe.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        merged = Vad._py_merge_chunks(raw, p["chunk_size"], p["onset"], p["offset"])
        new_merged = [
            {"start": round(float(m["start"]), 4), "end": round(float(m["end"]), 4),
             "n_segments": len(m["segments"])} for m in merged]
        if new_merged != existing["merged_chunks"]:
            print(f"! {name}: merged_chunks drift vs committed golden — skipped "
                  f"(silero non-determinism / version change)")
            continue
        out = {
            "segments": [
                {"start": round(float(s.start), 4), "end": round(float(s.end), 4),
                 "speaker": s.speaker} for s in raw],
            "merged_chunks": existing["merged_chunks"],
            "params": p,
        }
        vad_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        man["clips"][name]["artifacts"]["vad"]["sha256"] = _sha256(vad_path)
        changed += 1
        print(f"  {name:16} +{len(out['segments'])} raw segments")
    (outdir / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    print(f"\naugmented {changed}/{len(clips)} vad.json -> {outdir}/manifest.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", help="subset of clip names (default: all)")
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--compute", default="int8")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--vad-only", action="store_true",
                    help="only (re)write raw VAD segments into *.vad.json (cheap; "
                         "no whisper/align/diarize; preserves tensor goldens)")
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    man = json.loads(MANIFEST.read_text())
    clips = {n: m for n, m in man["clips"].items()
             if not args.clips or n in args.clips}
    if not clips:
        raise SystemExit(f"no matching clips (have: {sorted(man['clips'])})")

    import whisperx as wx
    pipe = wx.load_model(args.model, device=args.device,
                         compute_type=args.compute, vad_method="silero")

    if args.vad_only:
        _dump_vad_only(clips, outdir, wx=wx, pipe=pipe)
        return

    align_cache: dict = {}

    diarizer = None
    if not args.no_diarize:
        from app.diarize_model import resolve_local_model
        local = resolve_local_model()
        if local is None:
            print("! no vendored diarization checkpoint — skipping diarize")
        else:
            from whisperx.diarize import DiarizationPipeline
            diarizer = DiarizationPipeline(model_name=str(local), device=args.device)

    index = {}
    for name, meta in sorted(clips.items()):
        t0 = time.perf_counter()
        index[name] = _dump_clip(name, meta, wx=wx, pipe=pipe,
                                 align_cache=align_cache, diarizer=diarizer,
                                 outdir=outdir)
        print(f"  {name:16} {index[name]['n_align_segments']:2d} align-seg "
              f"{index[name]['n_words']:3d} words  ({time.perf_counter()-t0:5.1f}s)")

    manifest = {
        "config": {
            "model": args.model, "device": args.device, "compute_type": args.compute,
            "vad_method": "silero", "diarize": diarizer is not None,
            "whisperx": version("whisperx"),
            "faster_whisper": version("faster-whisper"),
            "torch": version("torch"),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        # starting tolerance budget — refine once an ORT export lets us measure
        # real torch-vs-ORT emission drift (Phase 3 prep). Structural items
        # (tokens/path/char-segments/merged-chunks) are EXACT.
        "tolerances": {
            "emission_atol": 1e-3, "trellis_atol": 1e-3,
            "time_frames": 1, "score_atol": 0.01,
            "exact": ["tokens", "path", "char_segments", "merged_chunks",
                      "speaker_labels"],
        },
        "clips": index,
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(index)} clips -> {outdir}/manifest.json")


if __name__ == "__main__":
    main()
