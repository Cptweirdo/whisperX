"""Phase 5 align-driver parity: the **fully native** ``whisperx_core.align_run``
(char-clean -> gather+forward -> emission_post -> align_assemble -> word_segments)
vs the committed ``words.json`` (the Python ``align()`` golden).

This drives the entire align body in C++ — the seam the native orchestrator needs.
The ORT forward reproduces the torch ``words.json`` exactly for the mirror models
(the Phase-3B finding), so word text is exact and timings sit within the manifest
±1-frame / ±0.01 tolerances. Also asserts the per-segment progress callback, the
``return_char_alignments`` path, and the stub paths (no clean chars / start past the
audio end).

**Opt-in**: needs ``RUN_MIRROR=1`` (network) **and** a ``WHISPERX_CORE_AUDIO`` build
(for ``align_run`` — ORT). Skips cleanly otherwise.
"""
import json
import os
from pathlib import Path

import pytest

if os.environ.get("RUN_MIRROR") != "1":
    pytest.skip("set RUN_MIRROR=1 to run the C++ align-driver parity test",
                allow_module_level=True)

np = pytest.importorskip("numpy")
hf = pytest.importorskip("huggingface_hub")
wc = pytest.importorskip("whisperx_core")
if not hasattr(wc, "align_run"):
    pytest.skip("whisperx_core built without WHISPERX_CORE_AUDIO (no align_run)",
                allow_module_level=True)

from whisperx.audio import load_audio  # noqa: E402  (ffmpeg; not the torch model)

ROOT = Path(__file__).resolve().parents[2]
INTER = ROOT / "golden" / "intermediates"
CLIPS_DIR = ROOT / "golden" / "clips"
REPO = os.environ.get("MIRROR_REPO", "KonstantK/wav2vec2-align-onnx")
SAMPLE_RATE = 16000

# en (torchaudio group_norm) + ru (HF layer_norm, batchable) cover both forward paths.
CLIPS = ["en_libri", "en_dialog", "ru_dialog"]
ALIGN_MODEL = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
}
_MODEL: dict = {}


def _model(lang: str):
    folder = ALIGN_MODEL[lang].replace("/", "--")
    if folder not in _MODEL:
        onnx = hf.hf_hub_download(REPO, f"{folder}/model.onnx")
        meta = json.loads(
            Path(hf.hf_hub_download(REPO, f"{folder}/meta.json")).read_text())
        _MODEL[folder] = (wc.Wav2Vec2Onnx(onnx), meta)
    return _MODEL[folder]


def _audio(clip):
    return np.ascontiguousarray(load_audio(str(CLIPS_DIR / f"{clip}.wav")),
                                dtype=np.float32)


def _segs(clip):
    return json.loads((INTER / f"{clip}.transcript.json").read_text())["segments"]


# --- 1. align_run word output reproduces words.json ---------------------------
@pytest.mark.parametrize("clip", CLIPS)
def test_align_run_words_match_golden(clip):
    lang = clip.split("_")[0]
    model, meta = _model(lang)
    golden = json.loads((INTER / f"{clip}.words.json").read_text())["words"]

    calls = []
    res = wc.align_run(_segs(clip), model, meta["dictionary"], _audio(clip), lang,
                       bool(meta["batchable"]), "nearest", False, calls.append)

    assert set(res) == {"segments", "word_segments"}
    got = res["word_segments"]
    assert len(got) == len(golden), f"{clip} word count {len(got)} vs {len(golden)}"
    for g, exp in zip(got, golden):
        assert g["word"] == exp["word"], f"{clip} word {g} vs {exp}"
        for key, tol in (("start", 0.021), ("end", 0.021), ("score", 0.011)):
            assert (key in g) == (key in exp), f"{clip} key {key} {g}/{exp}"
            if key in exp:
                assert abs(g[key] - exp[key]) <= tol, f"{clip} {key} {g} vs {exp}"

    # progress fires once per *input transcript segment* that aligns (not per
    # sentence-level subsegment), monotonically up to 100%. Every golden segment
    # aligns (none stubbed), so the count is the input segment count.
    n_segs = len(_segs(clip))
    assert len(calls) == n_segs and calls == sorted(calls)
    assert calls and abs(calls[-1] - 100.0) < 1e-6


# --- 2. word_segments is exactly the flattened per-segment words --------------
def test_word_segments_is_flattened_segments():
    model, meta = _model("en")
    res = wc.align_run(_segs("en_dialog"), model, meta["dictionary"],
                       _audio("en_dialog"), "en", bool(meta["batchable"]),
                       "nearest", False, None)
    flat = [w for seg in res["segments"] for w in seg["words"]]
    assert res["word_segments"] == flat


# --- 3. return_char_alignments attaches chars ---------------------------------
def test_align_run_char_alignments():
    model, meta = _model("en")
    res = wc.align_run(_segs("en_libri"), model, meta["dictionary"],
                       _audio("en_libri"), "en", bool(meta["batchable"]),
                       "nearest", True, None)
    subs = res["segments"]
    assert subs and all("chars" in s and isinstance(s["chars"], list) for s in subs)
    assert any(s["chars"] for s in subs), "char alignments should be populated"


# --- 4. stub paths: no clean chars / start past audio end ---------------------
def test_align_run_stub_segments():
    model, meta = _model("en")
    audio = _audio("en_libri")
    dur = len(audio) / SAMPLE_RATE
    calls = []
    transcript = [
        {"start": 0.0, "end": 0.4, "text": "   "},          # no clean chars -> stub
        {"start": dur + 5.0, "end": dur + 6.0, "text": "hello"},  # past end -> stub
    ]
    res = wc.align_run(transcript, model, meta["dictionary"], audio, "en",
                       bool(meta["batchable"]), "nearest", False, calls.append)
    segs = res["segments"]
    assert len(segs) == 2
    for s in segs:                       # stubs keep original text, no words
        assert s["words"] == [] and "text" in s
    assert res["word_segments"] == []
    assert calls == []                   # stubs never fire progress (Python parity)
