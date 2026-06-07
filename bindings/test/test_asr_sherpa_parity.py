"""WER/CER gate for the native sherpa-onnx Whisper ASR backend (`asr` token, 4a).

Per the **decoupled-goldens** decision (fact 3) Whisper text is *not* byte-stable
across decoders (CTranslate2 vs sherpa-onnx), so this backend is judged by **WER/CER
relative to the faster-whisper tiny baseline** (`golden/baseline.json`), not exact
text. The gate runs the native ``whisperx_core.WhisperSherpa`` over each golden clip's
**committed VAD spans** (the fixed ``*.vad.json`` merged_chunks — model-free) and
asserts the resulting WER/CER stay within a documented margin of the baseline, that
language detection matches, and that the facade pipeline glue
(``SherpaWhisperPipeline.transcribe``) returns the right contract shape.

Needs the sherpa Whisper ONNX assets — either ``WHISPERX_SHERPA_WHISPER_DIR`` pointing
at a local sherpa ``sherpa-onnx-whisper-tiny`` dir, or the published HF mirror
(``RUN_MIRROR=1``). Build the module with ``-DWHISPERX_CORE_AUDIO=ON`` and put it on
``PYTHONPATH``; skips cleanly if the module or the assets are unavailable.
"""
import json
import os
import sys
from pathlib import Path

import pytest

wc = pytest.importorskip("whisperx_core")
if not hasattr(wc, "WhisperSherpa"):
    pytest.skip("whisperx_core built without WHISPERX_CORE_AUDIO (no WhisperSherpa)",
                allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_baseline_golden import wer_cer  # noqa: E402  (the exact baseline metric)

from whisperx.audio import load_audio, SAMPLE_RATE  # noqa: E402
from whisperx.asr_sherpa import _resolve_sherpa_assets, SherpaWhisperPipeline  # noqa: E402
from whisperx.vads import Vad  # noqa: E402

INTER = ROOT / "golden" / "intermediates"
BASELINE = ROOT / "golden" / "baseline.json"

# Decoupled regression margins: greedy sherpa vs faster-whisper beam search drift a
# few points either way (measured worst-case ≈ +0.18 WER on a short CV clip); these
# margins catch gross breakage (a wired-wrong backend blows WER toward 1.0) while
# tolerating the decoder difference. Not a parity gate — see the module docstring.
WER_MARGIN = 0.30
CER_MARGIN = 0.20
WER_CEILING = 0.70  # absolute sanity bound even where the baseline is already high


def _assets_or_skip():
    if not os.environ.get("WHISPERX_SHERPA_WHISPER_DIR") and not os.environ.get(
        "RUN_MIRROR"
    ):
        pytest.skip("set WHISPERX_SHERPA_WHISPER_DIR or RUN_MIRROR for the asr gate")
    try:
        return _resolve_sherpa_assets("tiny")
    except Exception as e:  # noqa: BLE001 - missing/unreachable assets -> skip
        pytest.skip(f"sherpa Whisper tiny assets unavailable: {e}")


def _clips_with_reference():
    base = json.loads(BASELINE.read_text())["clips"]
    return [(name, clip) for name, clip in base.items() if clip.get("reference")]


@pytest.fixture(scope="module")
def model():
    enc, dec, tok, feat = _assets_or_skip()
    return wc.WhisperSherpa(
        encoder=enc, decoder=dec, tokens=tok, num_threads=4, feature_dim=feat
    )


@pytest.mark.parametrize("name,clip", _clips_with_reference(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_wer_cer_within_baseline_margin(model, name, clip):
    vad = json.loads((INTER / f"{name}.vad.json").read_text())
    spans = [(c["start"], c["end"]) for c in vad["merged_chunks"]]
    audio = load_audio(str(ROOT / "golden" / "clips" / f"{name}.wav"))

    outs = model.transcribe(audio, spans, clip["lang"], "transcribe")
    assert len(outs) == len(spans)
    for o in outs:  # C++ contract: each chunk is {text:str, avg_logprob:float}
        assert isinstance(o["text"], str)
        assert isinstance(o["avg_logprob"], float)

    hyp = " ".join(o["text"].strip() for o in outs).strip()
    assert hyp, f"{name}: empty sherpa hypothesis"

    wer, cer = wer_cer(clip["reference"], hyp)
    base_wer, base_cer = wer_cer(clip["reference"], clip["hypothesis"])
    assert wer <= min(base_wer + WER_MARGIN, WER_CEILING), (
        f"{name}: sherpa WER {wer:.3f} > baseline {base_wer:.3f} + {WER_MARGIN}")
    assert cer <= base_cer + CER_MARGIN, (
        f"{name}: sherpa CER {cer:.3f} > baseline {base_cer:.3f} + {CER_MARGIN}")


@pytest.mark.parametrize("name,clip", _clips_with_reference(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_detect_language_matches(model, name, clip):
    audio = load_audio(str(ROOT / "golden" / "clips" / f"{name}.wav"))
    detected = model.detect_language(audio[: SAMPLE_RATE * 30])
    assert detected == clip["lang"], f"{name}: detected {detected!r} != {clip['lang']!r}"


class _StubVad(Vad):
    """Replays a clip's committed raw VAD segments — exercises the real
    SherpaWhisperPipeline.transcribe loop (merge_chunks + glue) torch-free."""

    class _Seg:
        def __init__(self, s, e, sp):
            self.start, self.end, self.speaker = s, e, sp

    def __init__(self, raw_segments):
        super().__init__(vad_onset=0.5)
        self._raw = [self._Seg(r["start"], r["end"], r.get("speaker")) for r in raw_segments]

    @staticmethod
    def preprocess_audio(audio):
        return audio

    def __call__(self, _inputs):
        return self._raw


def test_pipeline_glue_contract_shape(model):
    """The facade pipeline returns {segments:[{text,start,end,avg_logprob}], language}
    with rounded float timings — the SingleSegment contract asr.py produces."""
    name = "en_dialog"
    clip = json.loads(BASELINE.read_text())["clips"][name]
    vad = json.loads((INTER / f"{name}.vad.json").read_text())
    audio = load_audio(str(ROOT / "golden" / "clips" / f"{name}.wav"))

    pipe = SherpaWhisperPipeline(
        model=model,
        vad=_StubVad(vad["segments"]),
        vad_params={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
        language=clip["lang"],
    )
    result = pipe.transcribe(audio, batch_size=8, language=clip["lang"])

    assert result["language"] == clip["lang"]
    assert result["segments"]
    for seg in result["segments"]:
        assert set(("text", "start", "end", "avg_logprob")) <= set(seg)
        assert isinstance(seg["text"], str)
        assert isinstance(seg["start"], float) and isinstance(seg["end"], float)
        assert round(seg["start"], 3) == seg["start"]  # timings rounded to 3 dp
