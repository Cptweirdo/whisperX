#!/usr/bin/env python3
"""Fetch + trim the EN/DE golden clip set and write a pinned manifest.

Pulls four short (<=30 s) clips for the C++ core golden-vector generator
(see docs/cpp-core-migration-briefs.md, Phase 0):

  en_libri    LibriSpeech test-clean  - EN single-speaker ASR/align
  de_cv       Common Voice 17 (de)    - DE single-speaker ASR/align (gated, needs HF token)
  en_ami      AMI meeting (mix-headset) + RTTM - EN 3-4 speaker diarization
  de_voxpop   VoxPopuli (de)          - DE multi-speaker debate diarization

One clip is additionally re-encoded to m4a to exercise the ffmpeg decode path.

Everything is PINNED: dataset revisions + the exact source ids/offsets live in
CLIPS below; outputs are sha256'd into golden/manifest.json. Re-running with the
same pins reproduces byte-identical clips. Changing a pin is a deliberate,
reviewed act (regenerates goldens downstream).

Usage:
    uv run python golden/fetch_datasets.py            # all clips
    uv run python golden/fetch_datasets.py en_ami     # one clip
    HF_TOKEN=... uv run python golden/fetch_datasets.py de_cv   # gated clip

Requires: ffmpeg on PATH; `datasets` + `soundfile` (not project deps - install
with `uv pip install datasets soundfile`, or add to a `golden` extra).

"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

SAMPLE_RATE = 16000  # whisperx/audio.py::SAMPLE_RATE - clips are normalized to this

HERE = Path(__file__).resolve().parent
CLIPS_DIR = HERE / "clips"
MANIFEST = HERE / "manifest.json"

# --- pinned clip specs -------------------------------------------------------
# Each entry is one golden clip. `kind` selects the fetcher below.
# `start`/`dur` are the trim window in seconds (<=30 s). Adjust the source ids
# only with intent - they are the pin.
CLIPS: dict[str, dict] = {
    "en_libri": {
        "kind": "hf",
        "out": "en_libri.wav",
        "repo": "openslr/librispeech_asr",
        "config": "clean",
        "split": "test",
        "revision": "main",   # pin a concrete commit before committing goldens
        "index": 0,            # nth streamed example
        "start": 0.0,
        "dur": 20.0,
        "note": "EN single-speaker, torchaudio align loader",
    },
    "de_cv": {
        "kind": "hf",
        "out": "de_cv.wav",
        "repo": "mozilla-foundation/common_voice_17_0",
        "config": "de",
        "split": "test",
        "revision": "main",   # pin a concrete commit before committing goldens
        "gated": True,         # accept terms on HF + export HF_TOKEN
        "index": 0,
        "start": 0.0,
        "dur": 0.0,            # 0 = whole clip (CV clips are already short)
        "note": "DE single-speaker, torchaudio align loader",
    },
    "de_voxpop": {
        "kind": "hf",
        "out": "de_voxpop.wav",
        "repo": "facebook/voxpopuli",
        "config": "de",
        "split": "test",
        "revision": "main",   # pin a concrete commit before committing goldens
        "index": 0,            # TODO: pick an index whose window has 2-4 speaker_ids
        "start": 0.0,
        "dur": 30.0,
        "note": "DE multi-speaker debate, diarization (verify <=4 speaker_ids)",
    },
    "en_ami": {
        # AMI mix-headset wav + ground-truth RTTM. Diarization golden uses the
        # RTTM, not ASR labels. Pick a <=30 s window with exactly 3-4 active
        # speakers (helper prints active-speaker count per candidate window).
        "kind": "ami",
        "out": "en_ami.wav",
        "meeting": "ES2004a",  # pin: AMI meeting id
        "audio_url": "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav",
        "rttm_url": "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/ES2004a.rttm",
        "start": 0.0,          # TODO: set to a 3-4 speaker window (see --scan)
        "dur": 30.0,
        "note": "EN 3-4 speaker diarization; RTTM is the speaker golden",
    },
    # m4a is derived from an existing clip, not downloaded.
    "en_libri_m4a": {
        "kind": "transcode",
        "out": "en_libri.m4a",
        "src": "en_libri.wav",
        "codec": "aac",
        "note": "ffmpeg decode path (M4A/AAC)",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_trim(src: Path, dst: Path, start: float, dur: float, codec: str = "pcm_s16le") -> None:
    """Trim [start, start+dur] from src, resample to 16k mono, write dst."""
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if start:
        cmd += ["-ss", str(start)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", codec, str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


def write_wav(samples, sr: int, dst: Path) -> None:
    import soundfile as sf

    sf.write(str(dst), samples, sr, subtype="PCM_16")


def fetch_hf(spec: dict, dst: Path) -> dict:
    """Stream one example from an HF audio dataset, resample, trim, write wav."""
    from datasets import Audio, load_dataset

    token = os.environ.get("HF_TOKEN") if spec.get("gated") else None
    if spec.get("gated") and not token:
        raise SystemExit(f"{spec['repo']} is gated - accept terms on HF and export HF_TOKEN")

    ds = load_dataset(
        spec["repo"], spec["config"], split=spec["split"],
        streaming=True, revision=spec["revision"], token=token,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    it = iter(ds)
    for _ in range(spec["index"]):
        next(it)
    ex = next(it)
    audio = ex["audio"]

    tmp = dst.with_suffix(".full.wav")
    write_wav(audio["array"], audio["sampling_rate"], tmp)
    if spec.get("dur"):
        ffmpeg_trim(tmp, dst, spec.get("start", 0.0), spec["dur"])
        tmp.unlink()
    else:
        tmp.rename(dst)

    meta = {"source_repo": spec["repo"], "config": spec["config"],
            "split": spec["split"], "revision": spec["revision"],
            "index": spec["index"]}
    # surface speaker hints when present (voxpopuli/ami-style configs)
    for k in ("speaker_id", "audio_id", "id", "sentence", "text", "raw_text"):
        if k in ex:
            meta[k] = ex[k] if k in ("speaker_id", "audio_id", "id") else None
    return meta


def _download(url: str, dst: Path) -> None:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, dst.open("wb") as f:  # noqa: S310
        while chunk := r.read(1 << 20):
            f.write(chunk)


def _parse_rttm(text: str) -> list[tuple[float, float, str]]:
    """RTTM SPEAKER lines -> [(start, end, speaker)]."""
    turns = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur, spk = float(p[3]), float(p[4]), p[7]
            turns.append((start, start + dur, spk))
    return turns


def _active_speakers(turns, start: float, end: float) -> set[str]:
    return {spk for s, e, spk in turns if s < end and e > start}


def fetch_ami(spec: dict, dst: Path, scan: bool = False) -> dict:
    raw = CLIPS_DIR / f"{spec['meeting']}.Mix-Headset.wav"
    rttm = CLIPS_DIR / f"{spec['meeting']}.rttm"
    if not raw.exists():
        _download(spec["audio_url"], raw)
    if not rttm.exists():
        _download(spec["rttm_url"], rttm)
    turns = _parse_rttm(rttm.read_text())

    if scan:
        # print active-speaker count for each 30 s window so you can pin `start`
        dur = spec.get("dur", 30.0)
        last = max(e for _, e, _ in turns)
        print(f"  scanning {spec['meeting']} for {int(dur)}s windows with 3-4 speakers:")
        t = 0.0
        while t < last:
            n = _active_speakers(turns, t, t + dur)
            if 3 <= len(n) <= 4:
                print(f"    start={t:7.1f}  speakers={sorted(n)}")
            t += dur
        return {}

    ffmpeg_trim(raw, dst, spec["start"], spec["dur"])
    active = sorted(_active_speakers(turns, spec["start"], spec["start"] + spec["dur"]))
    if not (3 <= len(active) <= 4):
        print(f"  WARNING: window has {len(active)} speakers ({active}); "
              f"want 3-4. Re-run with --scan to pick a window.")
    return {"meeting": spec["meeting"], "window": [spec["start"], spec["dur"]],
            "speakers": active, "audio_url": spec["audio_url"],
            "rttm_url": spec["rttm_url"]}


def fetch_transcode(spec: dict, dst: Path) -> dict:
    src = CLIPS_DIR / spec["src"]
    if not src.exists():
        raise SystemExit(f"{spec['src']} missing - build it first")
    ffmpeg_trim(src, dst, 0.0, 0.0, codec=spec["codec"])
    return {"derived_from": spec["src"], "codec": spec["codec"]}


def main(argv: list[str]) -> int:
    scan = "--scan" in argv
    names = [a for a in argv if not a.startswith("-")] or list(CLIPS)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"clips": {}}

    for name in names:
        if name not in CLIPS:
            print(f"unknown clip {name!r}; known: {', '.join(CLIPS)}")
            return 2
        spec = CLIPS[name]
        dst = CLIPS_DIR / spec["out"]
        print(f"[{name}] {spec['note']}")

        if spec["kind"] == "hf":
            meta = fetch_hf(spec, dst)
        elif spec["kind"] == "ami":
            meta = fetch_ami(spec, dst, scan=scan)
            if scan:
                continue
        elif spec["kind"] == "transcode":
            meta = fetch_transcode(spec, dst)
        else:
            raise SystemExit(f"bad kind {spec['kind']}")

        manifest["clips"][name] = {
            "file": f"clips/{spec['out']}",
            "sha256": sha256(dst),
            "sample_rate": SAMPLE_RATE,
            "note": spec["note"],
            **meta,
        }
        print(f"  -> {dst.relative_to(HERE)}  sha256={manifest['clips'][name]['sha256'][:12]}…")

    if not scan:
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {MANIFEST.relative_to(HERE)} ({len(manifest['clips'])} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
