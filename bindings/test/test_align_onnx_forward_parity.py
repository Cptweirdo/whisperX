"""Phase 3B parity: the **C++** wav2vec2 forward + post-processing vs the committed
torch emissions.

Where ``test_align_onnx_mirror.py`` validates the published ONNX under *python*
onnxruntime, this runs it through the native ``whisperx_core.Wav2Vec2Onnx`` (raw
``Ort::Session``, sherpa's vendored ORT) — the actual Phase-3B runtime path — then
``whisperx_core.align_emission_post`` (the C++ log_softmax + wildcard + tokenize).
For every golden segment it asserts:
  * the post-processed emission matches the committed ``seg{i}_emission`` within
    ``emission_atol`` (the forward + log_softmax + wildcard are all correct in C++), and
  * the tokens match the committed ``align.json`` tokens exactly (tokenization parity).
Plus the **mask parity gate**: for a batchable (layer_norm) model the padded batched
forward reproduces the per-segment emissions; group_norm models are asserted
non-batchable (they're run per-segment, where there's no padding to corrupt).

**Opt-in**: needs ``RUN_MIRROR=1`` (network) **and** a ``WHISPERX_CORE_AUDIO`` build
(for ``Wav2Vec2Onnx`` — ORT). Skips cleanly otherwise.
"""
import json
import os
from pathlib import Path

import pytest

if os.environ.get("RUN_MIRROR") != "1":
    pytest.skip("set RUN_MIRROR=1 to run the C++ align-forward parity test",
                allow_module_level=True)

np = pytest.importorskip("numpy")
hf = pytest.importorskip("huggingface_hub")
wc = pytest.importorskip("whisperx_core")
if not hasattr(wc, "Wav2Vec2Onnx") or not hasattr(wc, "align_emission_post"):
    pytest.skip("whisperx_core built without WHISPERX_CORE_AUDIO (no Wav2Vec2Onnx)",
                allow_module_level=True)

from whisperx.audio import load_audio  # noqa: E402  (ffmpeg; not the torch model)

ROOT = Path(__file__).resolve().parents[2]
INTER = ROOT / "golden" / "intermediates"
CLIPS_DIR = ROOT / "golden" / "clips"
REPO = os.environ.get("MIRROR_REPO", "KonstantK/wav2vec2-align-onnx")
SAMPLE_RATE = 16000
EMISSION_ATOL = 0.006  # manifest.json tolerances.emission_atol

CLIPS = ["en_libri", "en_dialog", "de_dialog", "ru_dialog",
         "ru_cv_71085", "ru_cv_71379", "ru_cv_71606", "ru_cv_79125"]

ALIGN_MODEL = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
}

# Download + Wav2Vec2Onnx-session cache, keyed by model folder (the 5 ru clips share
# one 1.2 GB session; hf_hub_download also disk-caches across runs).
_MODEL: dict = {}


def _model(lang: str):
    folder = ALIGN_MODEL[lang].replace("/", "--")
    if folder not in _MODEL:
        onnx = hf.hf_hub_download(REPO, f"{folder}/model.onnx")
        meta = json.loads(
            Path(hf.hf_hub_download(REPO, f"{folder}/meta.json")).read_text())
        _MODEL[folder] = (wc.Wav2Vec2Onnx(onnx), meta)
    return _MODEL[folder]


def _waveforms(clip: str):
    audio = load_audio(str(CLIPS_DIR / f"{clip}.wav"))
    segs = json.loads((INTER / f"{clip}.transcript.json").read_text())["segments"]
    return [audio[int(s["start"] * SAMPLE_RATE):int(s["end"] * SAMPLE_RATE)]
            .astype(np.float32) for s in segs]


@pytest.mark.parametrize("clip", CLIPS)
def test_cpp_forward_post_matches_golden(clip):
    lang = clip.split("_")[0]
    model, meta = _model(lang)
    dictionary, blank_id = meta["dictionary"], int(meta["blank_id"])
    align = json.loads((INTER / f"{clip}.align.json").read_text())["segments"]
    golden = np.load(INTER / f"{clip}.tensors.npz")

    # per-segment forward (valid for every model; the only path for group_norm).
    raws = model.forward(_waveforms(clip), batched=False)
    for i, raw in enumerate(raws):
        text_clean = align[i]["text_clean"]
        emi, tokens = wc.align_emission_post(raw, blank_id, text_clean, dictionary)
        g = golden[f"seg{i}_emission"]
        assert emi.shape == g.shape, \
            f"{clip} seg{i} emission {emi.shape} vs golden {g.shape}"
        d = float(np.abs(emi - g).max())
        assert d < EMISSION_ATOL, \
            f"{clip} seg{i} C++ forward+post drift {d:.2e} >= {EMISSION_ATOL}"
        assert list(tokens) == list(align[i]["tokens"]), f"{clip} seg{i} tokens"


def test_mask_gate_batched_matches_per_segment():
    """The runtime mask-parity gate. Batchable (layer_norm) models must reproduce
    per-segment emissions when padded+masked into one batch; group_norm models are
    non-batchable by design (padding corrupts them) and run per-segment."""
    _, en_meta = _model("en")
    _, de_meta = _model("de")
    ru_model, ru_meta = _model("ru")
    assert en_meta["batchable"] is False  # torchaudio group_norm
    assert de_meta["batchable"] is False
    assert ru_meta["batchable"] is True   # HF xls-r layer_norm

    waveforms = _waveforms("ru_dialog")  # multi-segment -> real padding in the batch
    assert len(waveforms) > 1
    per = ru_model.forward(waveforms, batched=False)
    bat = ru_model.forward(waveforms, batched=True)
    assert len(per) == len(bat)
    for i in range(len(per)):
        assert per[i].shape == bat[i].shape, f"ru_dialog seg{i} batched shape"
        d = float(np.abs(per[i] - bat[i]).max())
        assert d < EMISSION_ATOL, f"ru_dialog seg{i} batch-vs-single drift {d:.2e}"


def test_meta_contract_v2():
    """meta.json is the torch-free runtime contract for the masked v2 graph."""
    for lang in ("en", "de", "ru"):
        _, meta = _model(lang)
        assert meta["opset"] == 18
        assert meta["contract_version"] == 2
        assert meta["emits"] == "raw_logits"
        assert meta["inputs"] == ["waveform", "attention_mask"]
        assert meta["outputs"] == ["emissions", "frame_lengths"]
        assert isinstance(meta["batchable"], bool)
        assert meta["n_labels"] == len(meta["dictionary"])
