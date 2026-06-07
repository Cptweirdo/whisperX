#!/usr/bin/env python3
"""Synthesize a multi-speaker dialog from a Common Voice *Spontaneous Speech* corpus.

CV-SPS clips are single-speaker (one `client_id` per clip), so there is no native
dialog audio. This stitches clips from N **distinct speakers** into one track,
round-robin (A,B,C,D,A,B…) to mimic turn-taking, with a small silence between
turns — and emits the **ground-truth RTTM** (speaker turns at the exact concat
offsets) that serves as the diarization golden, plus a per-turn transcript.

Reusable across languages: any CV-SPS download has the same layout
(`audios/` + `ss-corpus-<lang>.tsv` with `client_id,audio_file,duration_ms,
transcription,split,gender` columns), so point `--corpus` at a German/etc. one.

Deterministic: `--seed` fixes speaker + clip choice → byte-identical output, so it
can be pinned as a committed golden.

Usage:
    uv run python golden/synthesize_dialog.py \
        --corpus golden/clips/sps-corpus-3.0-2026-03-09-ru \
        --lang ru --speakers 4 --turns 8 --out ru_dialog

    # another language, same dataset family:
    uv run python golden/synthesize_dialog.py \
        --corpus golden/clips/sps-corpus-3.0-…-de --lang de --speakers 3

Writes (under golden/):
    clips/<out>.wav     16 kHz mono synthesized dialog
    clips/<out>.rttm    ground-truth speaker turns (diarization golden)
    clips/<out>.json    per-turn {speaker,start,end,text} + provenance
and registers the clip in golden/manifest.json (sha256 of wav + rttm).

Requires: ffmpeg on PATH; `soundfile`, `numpy`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000  # whisperx/audio.py::SAMPLE_RATE

HERE = Path(__file__).resolve().parent
CLIPS_DIR = HERE / "clips"
MANIFEST = HERE / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def decode_16k_mono(src: Path) -> np.ndarray:
    """ffmpeg-decode any audio (mp3/…) to 16 kHz mono float32 in memory."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-c:a", "pcm_s16le", tmp.name],
            check=True, capture_output=True,
        )
        data, _ = sf.read(tmp.name, dtype="float32")
    return data


def load_corpus(corpus: Path, lang: str) -> list[dict]:
    tsv = corpus / f"ss-corpus-{lang}.tsv"
    if not tsv.exists():
        # fall back to the single ss-corpus-*.tsv present
        hits = list(corpus.glob("ss-corpus-*.tsv"))
        if len(hits) != 1:
            raise SystemExit(f"can't find ss-corpus-{lang}.tsv in {corpus} (found {hits})")
        tsv = hits[0]
    return list(csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"))


def pick_turns(rows: list[dict], n_speakers: int, n_turns: int,
               min_s: float, max_s: float, rng: np.random.Generator) -> list[dict]:
    """Choose n_speakers distinct speakers, then round-robin one clip each per turn."""
    def is_speech(text: str) -> bool:
        # drop non-speech clips whose transcription is only bracket markers
        # ([silence], [click], …) — strip [..] tokens and check for real words
        import re
        return bool(re.sub(r"\[[^\]]*\]", "", text).strip())

    usable = [r for r in rows
              if is_speech(r.get("transcription", ""))
              and r.get("duration_ms")
              and min_s * 1000 <= int(r["duration_ms"]) <= max_s * 1000]
    if not usable:
        raise SystemExit("no clips match the duration window")

    by_speaker: dict[str, list[dict]] = {}
    for r in usable:
        by_speaker.setdefault(r["client_id"], []).append(r)
    # only speakers with enough clips to cover their share of turns
    per_speaker = -(-n_turns // n_speakers)  # ceil
    eligible = [sid for sid, cl in by_speaker.items() if len(cl) >= per_speaker]
    if len(eligible) < n_speakers:
        raise SystemExit(
            f"need {n_speakers} speakers with >={per_speaker} clips each in "
            f"[{min_s},{max_s}]s; only {len(eligible)} qualify")

    speakers = list(rng.choice(eligible, size=n_speakers, replace=False))
    pools = {sid: list(rng.permutation(by_speaker[sid])) for sid in speakers}

    turns = []
    for i in range(n_turns):
        sid = speakers[i % n_speakers]
        turns.append(pools[sid].pop())
    return turns


def synthesize(turns: list[dict], corpus: Path, gap_s: float):
    """Concatenate decoded turns with `gap_s` silence; return audio + turn spans."""
    gap = np.zeros(int(gap_s * SAMPLE_RATE), dtype="float32")
    # stable speaker labels SPK0..N in first-appearance order
    label = {}
    pieces, spans = [], []
    cursor = 0
    for t in turns:
        sid = t["client_id"]
        spk = label.setdefault(sid, f"SPK{len(label)}")
        audio = decode_16k_mono(corpus / "audios" / t["audio_file"])
        start = cursor / SAMPLE_RATE
        pieces.append(audio)
        cursor += len(audio)
        end = cursor / SAMPLE_RATE
        spans.append({"speaker": spk, "client_id": sid[:16], "start": round(start, 3),
                      "end": round(end, 3), "gender": t.get("gender", ""),
                      "audio_file": t["audio_file"], "text": t["transcription"].strip()})
        pieces.append(gap)
        cursor += len(gap)
    full = np.concatenate(pieces[:-1]) if pieces else np.zeros(0, dtype="float32")
    return full, spans


def write_rttm(spans: list[dict], uri: str, path: Path) -> None:
    lines = []
    for s in spans:
        dur = s["end"] - s["start"]
        lines.append(
            f"SPEAKER {uri} 1 {s['start']:.3f} {dur:.3f} <NA> <NA> {s['speaker']} <NA> <NA>")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path, help="CV-SPS corpus dir")
    ap.add_argument("--lang", required=True, help="language code (matches ss-corpus-<lang>.tsv)")
    ap.add_argument("--out", help="output basename (default: <lang>_dialog)")
    ap.add_argument("--speakers", type=int, default=4, help="distinct speakers (2-4 typical)")
    ap.add_argument("--turns", type=int, default=8, help="total turns (round-robin)")
    ap.add_argument("--min-s", type=float, default=4.0, help="min per-clip seconds")
    ap.add_argument("--max-s", type=float, default=12.0, help="max per-clip seconds")
    ap.add_argument("--gap-s", type=float, default=0.4, help="silence between turns")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (pins the selection)")
    args = ap.parse_args(argv)

    out = args.out or f"{args.lang}_dialog"
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = load_corpus(args.corpus, args.lang)
    turns = pick_turns(rows, args.speakers, args.turns, args.min_s, args.max_s, rng)
    audio, spans = synthesize(turns, args.corpus, args.gap_s)

    wav = CLIPS_DIR / f"{out}.wav"
    rttm = CLIPS_DIR / f"{out}.rttm"
    meta = CLIPS_DIR / f"{out}.json"
    sf.write(str(wav), audio, SAMPLE_RATE, subtype="PCM_16")
    write_rttm(spans, out, rttm)
    n_spk = len({s["speaker"] for s in spans})
    meta.write_text(json.dumps(
        {"uri": out, "lang": args.lang, "n_speakers": n_spk, "n_turns": len(spans),
         "duration_s": round(len(audio) / SAMPLE_RATE, 3), "gap_s": args.gap_s,
         "seed": args.seed, "corpus": args.corpus.name, "turns": spans},
        indent=2, ensure_ascii=False) + "\n")

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"clips": {}}
    manifest["clips"][out] = {
        "file": f"clips/{wav.name}", "rttm": f"clips/{rttm.name}",
        "transcript": f"clips/{meta.name}", "sha256": sha256(wav),
        "rttm_sha256": sha256(rttm), "sample_rate": SAMPLE_RATE,
        "note": f"{args.lang.upper()} SYNTHETIC {n_spk}-speaker dialog (CV-SPS concat) - diarization",
        "synthetic": True, "n_speakers": n_spk, "seed": args.seed,
        "duration_s": round(len(audio) / SAMPLE_RATE, 3), "corpus": args.corpus.name,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"{out}: {n_spk} speakers, {len(spans)} turns, "
          f"{round(len(audio)/SAMPLE_RATE,1)}s -> {wav.relative_to(HERE)}")
    for s in spans:
        print(f"  {s['start']:6.2f}-{s['end']:6.2f}  {s['speaker']}  {s['text'][:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
