"""A/B quality + glue gate for the native sherpa-onnx diarization backend
(``diarize`` token, Phase 4 / slice 4b).

This backend is **A/B, not parity** with pyannote ``community-1`` (different
segmentation model + embedding extractor + clustering algorithm — see the handoff),
so there is no turn-for-turn golden. Instead it is gated by **Diarization Error Rate
(DER) vs the synthetic dialogs' CC0 ground-truth RTTM**, plus the wiring contract.

Measured context (both deterministic on the pinned model + committed clips): the
synthetic dialogs are concatenated CommonVoice clips and are *hard* — the incumbent
pyannote ``community-1`` itself scores DER ≈ 32 % (en), 32 % (de), 64 % (ru) with the
wrong speaker count on them. The sherpa backend is comparable (≈ 43/32/45 %), so the
gate is a generous absolute **DER ceiling** (a regression bound, not a quality claim)
+ multi-speaker sanity + the assign-glue contract — not ``count == 4``.

Needs the sherpa diarization ONNX (segmentation + embedding): either
``WHISPERX_DIARIZE_ONNX_DIR`` pointing at a local dir, or the published HF mirror
(``RUN_MIRROR=1``). Build with ``-DWHISPERX_CORE_AUDIO=ON`` on ``PYTHONPATH``; skips
cleanly if the module or assets are unavailable.
"""
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

wc = pytest.importorskip("whisperx_core")
if not hasattr(wc, "SherpaDiarizer"):
    pytest.skip("whisperx_core built without WHISPERX_CORE_AUDIO (no SherpaDiarizer)",
                allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "golden" / "clips"
DIALOGS = ["en_dialog", "de_dialog", "ru_dialog"]

# Generous absolute DER bound. The committed synthetic dialogs are concatenated
# CommonVoice clips on which *both* diarizers score poorly (community-1: 32/32/64 %;
# sherpa: ~43/32/45 %). 0.70 catches a wired-wrong backend (DER → ~1.0) while
# tolerating the genuine A/B difficulty — it is not a quality claim.
DER_CEILING = 0.70


def _assets_or_skip():
    if not os.environ.get("WHISPERX_DIARIZE_ONNX_DIR") and not os.environ.get(
        "RUN_MIRROR"
    ):
        pytest.skip("set WHISPERX_DIARIZE_ONNX_DIR or RUN_MIRROR for the diarize gate")
    from whisperx.diarize_sherpa import _resolve_diarize_assets

    try:
        return _resolve_diarize_assets()
    except Exception as e:  # noqa: BLE001 - missing/unreachable assets -> skip
        pytest.skip(f"sherpa diarization assets unavailable: {e}")


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _read_rttm(path: Path):
    turns = []
    for line in path.read_text().splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            turns.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return turns


def _der(ref, hyp, dur, res=0.01):
    """Frame-level DER (missed + false-alarm + speaker-confusion) / ref speech.

    Each hypothesis speaker is mapped to the ref speaker it overlaps most (a lenient
    many-to-one mapping — adequate for a regression bound). Ref is no-overlap.
    """
    n = int(dur / res)
    rf = [None] * n
    hf = [None] * n
    for s, e, spk in ref:
        for i in range(int(s / res), min(int(e / res), n)):
            rf[i] = spk
    for s, e, spk in hyp:
        for i in range(int(s / res), min(int(e / res), n)):
            hf[i] = spk
    ref_spk = sorted({s for s in rf if s})
    hyp_spk = sorted({s for s in hf if s})
    cooc = {}
    for r, h in zip(rf, hf):
        if r and h:
            cooc[(h, r)] = cooc.get((h, r), 0) + 1
    mapping = {h: max(ref_spk, key=lambda r: cooc.get((h, r), 0)) if ref_spk else None
               for h in hyp_spk}
    miss = fa = conf = total = 0
    for r, h in zip(rf, hf):
        if r:
            total += 1
        if r and not h:
            miss += 1
        elif h and not r:
            fa += 1
        elif r and h and mapping[h] != r:
            conf += 1
    return (miss + fa + conf) / max(total, 1), len(hyp_spk)


@pytest.fixture(scope="module")
def diarizer():
    seg, embed = _assets_or_skip()
    # merge_threshold=0 pins raw FastClustering: this DER ceiling bounds the
    # clusterer itself, independent of the centroid-merge post-pass defaults.
    return wc.SherpaDiarizer(
        segmentation=seg, embedding=embed, num_threads=4, threshold=0.5,
        merge_threshold=0.0
    )


@pytest.mark.parametrize("clip", DIALOGS)
def test_der_within_ceiling(diarizer, clip):
    audio = _load_wav(CLIPS / f"{clip}.wav")
    dur = len(audio) / 16000
    ref = _read_rttm(CLIPS / f"{clip}.rttm")

    segs = diarizer.diarize(audio, 0)  # auto count
    assert segs, "diarizer produced no turns"
    # Native contract shape.
    for s in segs:
        assert set(s) == {"start", "end", "speaker"}
        assert isinstance(s["speaker"], int)
        assert s["end"] >= s["start"]

    hyp = [(s["start"], s["end"], str(s["speaker"])) for s in segs]
    der, n_spk = _der(ref, hyp, dur)
    assert n_spk >= 2, f"{clip}: expected multiple speakers, got {n_spk}"
    assert der <= DER_CEILING, f"{clip}: DER {der:.3f} > ceiling {DER_CEILING}"


def test_speaker_count_control(diarizer):
    """num_clusters is an effective control: targeting 2 yields no more speakers
    than the auto run (sherpa treats it as a clustering target, final count ≤)."""
    audio = _load_wav(CLIPS / "en_dialog.wav")
    auto = {s["speaker"] for s in diarizer.diarize(audio, 0)}
    forced = {s["speaker"] for s in diarizer.diarize(audio, 2)}
    assert len(forced) <= len(auto)
    assert len(forced) >= 1


def test_pipeline_glue_and_embeddings(diarizer):
    """The facade SherpaDiarizationPipeline emits the same DataFrame the pyannote
    path does, it feeds assign_word_speakers (the landed `assign` glue), and
    return_embeddings yields per-speaker vectors of the extractor's dim."""
    from whisperx.diarize import assign_word_speakers
    from whisperx.diarize_sherpa import SherpaDiarizationPipeline

    pipe = SherpaDiarizationPipeline(model=diarizer)
    audio = _load_wav(CLIPS / "en_dialog.wav")
    df, emb = pipe(audio, return_embeddings=True)

    assert list(df.columns) == ["segment", "label", "speaker", "start", "end"]
    assert len(df) > 0
    assert all(str(s).startswith("SPEAKER_") for s in df["speaker"])

    # Glue: a fixed transcript gets speaker labels on segments + words.
    transcript = {
        "segments": [
            {"start": 2.0, "end": 10.0, "text": "a",
             "words": [{"word": "a", "start": 2.0, "end": 2.5}]},
            {"start": 40.0, "end": 47.0, "text": "b",
             "words": [{"word": "b", "start": 40.0, "end": 40.5}]},
        ]
    }
    out = assign_word_speakers(df, transcript)
    assert all("speaker" in s for s in out["segments"])
    assert all("speaker" in w for s in out["segments"] for w in s["words"])

    # Embeddings: one vector per speaker, dim == extractor dim.
    dim = diarizer.embedding_dim()
    assert emb and all(len(v) == dim for v in emb.values())
    assert set(emb) == set(df["speaker"].unique())
