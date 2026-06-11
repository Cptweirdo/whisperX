"""Golden-set baseline for the current (Python) WhisperX pipeline + per-stage timings.

Runs the live pipeline — **transcribe → align → diarize** — over the committed
golden clips (`golden/clips/`, see `golden/manifest.json`) and records, per clip:
the hypothesis text, WER/CER against the reference (where one exists), structural
counts (segments/words/speakers), and **wall-clock timing + real-time factor (RTF)
per stage**. The aggregate is written to `golden/baseline.json` — the reference the
C++ core is later diffed against (Phase 0 of docs/cpp-core-migration-briefs.md).

This is **heavy and opt-in** (downloads a Whisper model + wav2vec2 align models,
runs CPU inference). It is skipped unless `RUN_BASELINE=1`:

    RUN_BASELINE=1 uv run pytest tests/test_baseline_golden.py -v -s

Knobs (env): WHISPERX_BASELINE_MODEL (default ``tiny``),
WHISPERX_BASELINE_COMPUTE (default ``int8``), WHISPERX_BASELINE_DEVICE
(default ``cpu``). Diarization uses the **vendored** community-1 checkpoint
(`app/diarize_model.py::resolve_local_model`) — no HF token. Set
WHISPERX_BASELINE_NO_DIARIZE=1 to skip the (slow) diarization stage.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

RUN = os.environ.get("RUN_BASELINE") == "1"
pytestmark = [
    pytest.mark.baseline,
    pytest.mark.skipif(not RUN, reason="set RUN_BASELINE=1 to run the golden baseline"),
]

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "golden"
MANIFEST = GOLDEN / "manifest.json"
BASELINE = GOLDEN / "baseline.json"

MODEL = os.environ.get("WHISPERX_BASELINE_MODEL", "tiny")
COMPUTE = os.environ.get("WHISPERX_BASELINE_COMPUTE", "int8")
DEVICE = os.environ.get("WHISPERX_BASELINE_DEVICE", "cpu")
DO_DIARIZE = os.environ.get("WHISPERX_BASELINE_NO_DIARIZE") != "1"


# --- text metrics (no external deps) ----------------------------------------
def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)  # strip punctuation
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two token sequences."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    """Word- and character-error rate of `hypothesis` against `reference`."""
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    rw, hw = ref.split(), hyp.split()
    wer = _edit_distance(rw, hw) / max(len(rw), 1)
    cer = _edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", ""))) / max(
        len(ref.replace(" ", "")), 1)
    return round(wer, 4), round(cer, 4)


# --- clip set ----------------------------------------------------------------
def _lang_of(name: str) -> str:
    return name.split("_", 1)[0]  # en_libri/ru_cv_*/de_dialog -> en/ru/de


def _reference(name: str, meta: dict) -> str:
    """Reference transcript for WER, or '' when none is committed."""
    if meta.get("transcription"):
        return meta["transcription"]
    if meta.get("synthetic"):  # dialogs carry per-turn text in their .json
        side = GOLDEN / meta.get("transcript", f"clips/{name}.json")
        if side.exists():
            turns = json.loads(side.read_text(encoding="utf-8"))["turns"]
            return " ".join(t["text"] for t in turns)
    return ""


def _clips() -> list[dict]:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = []
    for name, meta in sorted(man["clips"].items()):
        if "rttm" in meta and not meta.get("synthetic"):
            pass  # (none today; future real-overlap clips)
        out.append({
            "name": name, "meta": meta, "lang": _lang_of(name),
            "audio": ROOT / "golden" / meta["file"],
            "reference": _reference(name, meta),
            "multispeaker": bool(meta.get("n_speakers", 1) and meta.get("n_speakers", 1) > 1),
            "audio_s": meta.get("duration_s"),
        })
    return out


CLIPS = _clips() if MANIFEST.exists() else []


# --- shared, expensive resources --------------------------------------------
@pytest.fixture(scope="session")
def whisperx():
    import whisperx as wx
    return wx


@pytest.fixture(scope="session")
def pipe(whisperx):
    # silero VAD avoids the HF-gated pyannote segmentation model
    return whisperx.load_model(
        MODEL, device=DEVICE, compute_type=COMPUTE, vad_method="silero")


@pytest.fixture(scope="session")
def align_cache():
    return {}


@pytest.fixture(scope="session")
def diarizer(whisperx):
    """Vendored community-1 checkpoint - offline, no HF token."""
    if not DO_DIARIZE:
        return None
    from app.diarize_model import resolve_local_model
    local = resolve_local_model()
    if local is None:
        pytest.skip("no vendored diarization checkpoint under app/models/")
    from whisperx.diarize import DiarizationPipeline
    return DiarizationPipeline(model_name=str(local), device=DEVICE)


@pytest.fixture(scope="session")
def results() -> dict:
    """Collects per-clip baseline rows; written to baseline.json at session end."""
    return {}


@pytest.fixture(scope="session", autouse=True)
def _write_baseline(results):
    yield
    if not results:
        return
    import faster_whisper  # noqa: F401 - version provenance
    from importlib.metadata import version
    payload = {
        "config": {
            "model": MODEL, "device": DEVICE, "compute_type": COMPUTE,
            "vad_method": "silero", "diarize": DO_DIARIZE,
            "whisperx": version("whisperx"),
            "faster_whisper": version("faster-whisper"),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "clips": dict(sorted(results.items())),
    }
    BASELINE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    # human-readable timing table to stdout (pytest -s)
    print(f"\n=== baseline ({MODEL} / {DEVICE} / {COMPUTE}) ===")
    print(f"{'clip':16} {'audio_s':>7} {'asr_s':>7} {'align_s':>7} "
          f"{'diar_s':>7} {'rtf':>5} {'WER':>6} {'spk':>4}")
    for name, r in payload["clips"].items():
        t = r["timings_s"]
        print(f"{name:16} {r.get('audio_s') or 0:7.1f} {t.get('transcribe',0):7.2f} "
              f"{t.get('align',0):7.2f} {t.get('diarize',0) or 0:7.2f} "
              f"{r.get('rtf_total',0):5.2f} "
              f"{(r['wer'] if r.get('wer') is not None else -1):6.3f} "
              f"{r.get('speakers_pred') or 0:4d}")


# --- the baseline run --------------------------------------------------------
@pytest.mark.parametrize("clip", CLIPS, ids=[c["name"] for c in CLIPS])
def test_baseline(clip, whisperx, pipe, align_cache, diarizer, results):
    name, lang, audio_path = clip["name"], clip["lang"], clip["audio"]
    assert audio_path.exists(), f"missing golden clip {audio_path}"
    audio = whisperx.load_audio(str(audio_path))
    audio_s = len(audio) / 16000
    timings: dict[str, float] = {}

    # 1) transcribe
    t0 = time.perf_counter()
    tr = pipe.transcribe(audio, batch_size=8, language=lang)
    timings["transcribe"] = round(time.perf_counter() - t0, 3)
    segments = tr["segments"]
    hypothesis = " ".join(s["text"].strip() for s in segments).strip()
    assert segments, f"{name}: transcribe produced no segments"
    assert hypothesis, f"{name}: empty hypothesis"

    # 2) align
    if lang not in align_cache:
        align_cache[lang] = whisperx.load_align_model(language_code=lang, device=DEVICE)
    amodel, ameta = align_cache[lang]
    t0 = time.perf_counter()
    aligned = whisperx.align(segments, amodel, ameta, audio, DEVICE,
                             return_char_alignments=False)
    timings["align"] = round(time.perf_counter() - t0, 3)
    words = [w for s in aligned["segments"] for w in s.get("words", [])]
    assert words, f"{name}: alignment produced no words"

    # 3) diarize (multi-speaker clips only; vendored checkpoint)
    speakers_pred = None
    if diarizer is not None and clip["multispeaker"]:
        t0 = time.perf_counter()
        diar_df = diarizer(audio)
        whisperx.assign_word_speakers(diar_df, aligned)
        timings["diarize"] = round(time.perf_counter() - t0, 3)
        speakers_pred = int(diar_df["speaker"].nunique())
        assert speakers_pred >= 1, f"{name}: no speakers detected"

    # metrics
    wer = cer = None
    if clip["reference"]:
        wer, cer = wer_cer(clip["reference"], hypothesis)

    proc = sum(timings.values())
    results[name] = {
        "lang": lang,
        "audio_s": round(audio_s, 3),
        "segments": len(segments),
        "words": len(words),
        "hypothesis": hypothesis,
        "reference": clip["reference"] or None,
        "wer": wer,
        "cer": cer,
        "speakers_true": clip["meta"].get("n_speakers"),
        "speakers_pred": speakers_pred,
        "timings_s": timings,
        "rtf_total": round(proc / audio_s, 3) if audio_s else None,
    }

    # structural invariants (NOT byte-parity: ASR text isn't engine-stable)
    assert timings["transcribe"] > 0 and timings["align"] > 0
    if clip["multispeaker"] and diarizer is not None:
        # synthetic dialogs have 4 speakers; tiny-model diarization should find >1
        assert speakers_pred and speakers_pred >= 2, (
            f"{name}: expected multiple speakers, got {speakers_pred}")
