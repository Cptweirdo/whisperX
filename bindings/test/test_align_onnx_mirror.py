"""Pull-and-parity for the wav2vec2 align ONNX mirror (``KonstantK/wav2vec2-align-onnx``).

Proves the **published** ONNX reproduces the committed torch emissions under ONNX
Runtime — the load-bearing asset Phase 3B's C++ aligner consumes. It runs the model
under onnxruntime (never the torch forward) on the exact waveform slices
``alignment.py`` fed wav2vec2 (reconstructed from the golden clip + per-segment
start/end), applies the same log_softmax + wildcard extension the Python facade does,
and asserts ``max|Δ|`` vs each committed ``seg{i}_emission`` is under
``emission_atol`` (golden/intermediates/manifest.json).

This validates the *round trip* — export → upload → **download** → run — so it catches
upload corruption and non-reproducibility that the exporter's own pre-upload
self-check can't. Covers both loader families: en/de torchaudio bundles + ru HF.

**Opt-in** (network + onnxruntime): set ``RUN_MIRROR=1``. Runs in the project env
(uses ``whisperx.load_audio``); skips cleanly otherwise. The exporter
(``golden/export_align_onnx.py``) is the extensible producer; this test covers the
committed golden languages.
"""
import json
import os
from pathlib import Path

import pytest

if os.environ.get("RUN_MIRROR") != "1":
    pytest.skip("set RUN_MIRROR=1 to run the HF align-mirror parity test",
                allow_module_level=True)

np = pytest.importorskip("numpy")
ort = pytest.importorskip("onnxruntime")
hf = pytest.importorskip("huggingface_hub")

from whisperx.audio import load_audio  # noqa: E402  (ffmpeg; not the torch model)

ROOT = Path(__file__).resolve().parents[2]
INTER = ROOT / "golden" / "intermediates"
CLIPS_DIR = ROOT / "golden" / "clips"
REPO = os.environ.get("MIRROR_REPO", "KonstantK/wav2vec2-align-onnx")
SAMPLE_RATE = 16000
EMISSION_ATOL = 0.006  # manifest.json tolerances.emission_atol

CLIPS = ["en_libri", "en_dialog", "de_dialog", "ru_dialog",
         "ru_cv_71085", "ru_cv_71379", "ru_cv_71606", "ru_cv_79125"]

# lang -> default align model, mirroring whisperx.alignment.DEFAULT_ALIGN_MODELS_*
# (importing alignment.py would drag torch; the exporter is the extensible source of
# truth — add a clip's language here when the golden set grows).
ALIGN_MODEL = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
}


def _sanitize(name: str) -> str:
    return name.replace("/", "--")


def _log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (x - m) - np.log(e.sum(axis=-1, keepdims=True))


def _extend_wildcard(emi, blank_id: int):
    non_blank = np.ones(emi.shape[1], dtype=bool)
    non_blank[blank_id] = False
    col = emi[:, non_blank].max(axis=1, keepdims=True)
    return np.concatenate([emi, col], axis=1)


# Download + ORT-session cache, keyed by model folder (the 5 ru clips share one
# 1.2 GB session; hf_hub_download also disk-caches across runs).
_SESS: dict = {}
_META: dict = {}


def _model(lang: str):
    folder = _sanitize(ALIGN_MODEL[lang])
    if folder not in _SESS:
        onnx = hf.hf_hub_download(REPO, f"{folder}/model.onnx")
        meta = json.loads(
            Path(hf.hf_hub_download(REPO, f"{folder}/meta.json")).read_text())
        _SESS[folder] = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        _META[folder] = meta
    return _SESS[folder], _META[folder]


@pytest.mark.parametrize("clip", CLIPS)
def test_mirror_emissions_match_golden(clip):
    lang = clip.split("_")[0]
    sess, meta = _model(lang)
    blank_id = meta["blank_id"]

    audio = load_audio(str(CLIPS_DIR / f"{clip}.wav"))
    segs = json.loads((INTER / f"{clip}.transcript.json").read_text())["segments"]
    golden = np.load(INTER / f"{clip}.tensors.npz")

    for i, seg in enumerate(segs):
        # the exact waveform align.py feeds the model (alignment.py:264-268)
        f1, f2 = int(seg["start"] * SAMPLE_RATE), int(seg["end"] * SAMPLE_RATE)
        wf = audio[f1:f2][None, :].astype(np.float32)  # (1, N)

        # contract v2: masked graph — feed an all-ones attention_mask (batch 1, no
        # padding) and trim by frame_lengths (a no-op here, exercises the output).
        mask = np.ones_like(wf, dtype=np.int64)
        logits, flen = sess.run(["emissions", "frame_lengths"],
                                {"waveform": wf, "attention_mask": mask})
        emi = _log_softmax(logits[:, :int(flen[0])])[0].astype(np.float32)

        g = golden[f"seg{i}_emission"]
        # committed golden is wildcard-extended iff the segment had an OOV char
        if g.shape[1] == emi.shape[1] + 1:
            emi = _extend_wildcard(emi, blank_id)
        assert emi.shape == g.shape, \
            f"{clip} seg{i} emission shape {emi.shape} vs golden {g.shape}"
        d = float(np.abs(emi - g).max())
        assert d < EMISSION_ATOL, \
            f"{clip} seg{i} mirror drift {d:.2e} >= emission_atol {EMISSION_ATOL}"


def test_mirror_meta_contract():
    """meta.json is the torch-free runtime contract: opset, raw-logits, blank_id +
    dictionary that line up with the committed align goldens (no torch needed)."""
    for lang, clip in (("en", "en_libri"), ("de", "de_dialog"), ("ru", "ru_cv_71085")):
        _, meta = _model(lang)
        assert meta["opset"] == 18
        assert meta["contract_version"] == 2
        assert meta["inputs"] == ["waveform", "attention_mask"]
        assert meta["outputs"] == ["emissions", "frame_lengths"]
        assert isinstance(meta["batchable"], bool)
        assert meta["emits"] == "raw_logits"
        assert meta["input"] == "waveform_16k_mono_f32"
        assert isinstance(meta["dictionary"], dict)
        assert meta["n_labels"] == len(meta["dictionary"])

        align = json.loads((INTER / f"{clip}.align.json").read_text())["segments"][0]
        assert meta["blank_id"] == align["blank_id"], f"{lang} blank_id"
        # golden emission width == model labels, or +1 when wildcard-extended
        g = np.load(INTER / f"{clip}.tensors.npz")["seg0_emission"]
        assert g.shape[1] in (meta["n_labels"], meta["n_labels"] + 1), \
            f"{lang} width {g.shape[1]} vs n_labels {meta['n_labels']}"
