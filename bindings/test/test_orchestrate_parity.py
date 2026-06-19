"""Phase 5 orchestrate parity: the **fully native** ``whisperx_core.run_job``
(decode → silero VAD → merge_chunks → WhisperSherpa → align_run → SherpaDiarizer +
assign_word_speakers, all over one decode-once AudioBuffer) vs **calling the same
native stages individually**.

Both paths use the identical native engines, so the orchestrator is correct iff it
reproduces the staged assembly: word text + structure + speaker labels **exact**,
numeric timings/scores within the manifest fp tolerances (the same ±1-frame / ±0.01
the align goldens use — silero/ASR/align/diarize are all deterministic, but Python
``round`` vs the C++ banker's-round and re-running the ORT forward can drift by
sub-ulp). Also asserts the stage-progress sequence, ``on_duration`` fires once, the
align-resolver fires once at loading_align, and the diarizer-absent path.

**Opt-in**: needs ``RUN_MIRROR=1`` (network: sherpa Whisper + align + diarize ONNX)
**and** a ``WHISPERX_CORE_AUDIO`` build (for ``run_job`` / the engines). Skips cleanly
otherwise. A local sherpa Whisper dir (``WHISPERX_SHERPA_WHISPER_DIR``) / diarize dir
(``WHISPERX_DIARIZE_ONNX_DIR``) is used when set (dev/CI before the mirrors publish).
"""
import json
import os
from pathlib import Path

import pytest

if os.environ.get("RUN_MIRROR") != "1":
    pytest.skip("set RUN_MIRROR=1 to run the C++ orchestrate parity test",
                allow_module_level=True)

np = pytest.importorskip("numpy")
wc = pytest.importorskip("whisperx_core")
if not hasattr(wc, "run_job"):
    pytest.skip("whisperx_core built without WHISPERX_CORE_AUDIO (no run_job)",
                allow_module_level=True)

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio  # noqa: E402
from whisperx.asr_sherpa import _resolve_sherpa_assets  # noqa: E402
from whisperx.diarize_sherpa import _resolve_diarize_assets  # noqa: E402
from whisperx.vads.silero import _silero_model_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = ROOT / "golden" / "clips"
INTER = ROOT / "golden" / "intermediates"
ALIGN_REPO = os.environ.get("MIRROR_REPO", "KonstantK/wav2vec2-align-onnx")

ONSET, OFFSET, CHUNK = 0.5, 0.363, 30.0

ALIGN_MODEL = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
}

_ASR = []
_DIARIZER = []
_ALIGN: dict = {}


def _audio(clip):
    return np.ascontiguousarray(load_audio(str(CLIPS_DIR / f"{clip}.wav")),
                                dtype=np.float32)


def _asr():
    if not _ASR:
        enc, dec, tok, feat = _resolve_sherpa_assets("tiny")
        _ASR.append(wc.WhisperSherpa(encoder=enc, decoder=dec, tokens=tok,
                                     num_threads=4, feature_dim=feat))
    return _ASR[0]


def _diarizer():
    if not _DIARIZER:
        seg, embed = _resolve_diarize_assets()
        _DIARIZER.append(wc.SherpaDiarizer(segmentation=seg, embedding=embed,
                                           num_threads=4, provider="cpu"))
    return _DIARIZER[0]


def _align(lang):
    import huggingface_hub as hf
    if lang not in _ALIGN:
        folder = ALIGN_MODEL[lang].replace("/", "--")
        onnx = hf.hf_hub_download(ALIGN_REPO, f"{folder}/model.onnx")
        meta = json.loads(
            Path(hf.hf_hub_download(ALIGN_REPO, f"{folder}/meta.json")).read_text())
        _ALIGN[lang] = (wc.Wav2Vec2Onnx(onnx), meta)
    return _ALIGN[lang]


def _staged(audio, *, with_diarizer, num_clusters=0):
    """The native staged path: call each C++ stage individually, assembling the
    transcript exactly as the orchestrator does (Python round / strip / label)."""
    asr = _asr()
    segs = wc.silero_segments(audio, _silero_model_path(), SAMPLE_RATE, ONSET, CHUNK)
    merged = wc.merge_chunks(segs, CHUNK, ONSET, OFFSET)
    spans = [(m["start"], m["end"]) for m in merged]
    lang = asr.detect_language(audio[:N_SAMPLES]) if spans else ""
    chunks = asr.transcribe(audio, spans, lang, "transcribe")
    transcript = [
        {"text": c["text"].strip(), "start": round(s[0], 3), "end": round(s[1], 3),
         "avg_logprob": float(c["avg_logprob"])}
        for s, c in zip(spans, chunks)
    ]
    model, meta = _align(lang or "en")
    result = wc.align_run(transcript, model, meta["dictionary"], audio, lang or "en",
                          bool(meta["batchable"]), "nearest", False, None)
    if with_diarizer:
        raw = _diarizer().diarize(audio, num_clusters)
        turns = [(r["start"], r["end"], f"SPEAKER_{r['speaker']:02d}") for r in raw]
        result["segments"] = wc.assign_word_speakers(turns, result["segments"], False)
        result["diarized"] = True
    else:
        result["diarized"] = False
    result["language"] = lang or "en"
    return result


def _run_native(clip, *, with_diarizer, num_clusters=0):
    """Invoke the native orchestrator, recording the progress sequence + callbacks."""
    stages, durations, resolve_calls = [], [], []

    def resolve_align(lang):
        resolve_calls.append(lang)
        model, meta = _align(lang)
        return (model, meta["dictionary"], bool(meta["batchable"]))

    result = wc.run_job(
        str(CLIPS_DIR / f"{clip}.wav"), _asr(), _silero_model_path(),
        ONSET, OFFSET, CHUNK, "", "transcribe", resolve_align, "nearest", False,
        _diarizer() if with_diarizer else None, num_clusters,
        stages.append, durations.append,
    )
    return result, stages, durations, resolve_calls


def _assert_words_close(got, exp):
    assert len(got) == len(exp), f"word count {len(got)} vs {len(exp)}"
    for g, e in zip(got, exp):
        assert g["word"] == e["word"], f"word {g} vs {e}"
        for key, tol in (("start", 0.021), ("end", 0.021), ("score", 0.011)):
            assert (key in g) == (key in e), f"key {key}: {g} / {e}"
            if key in e:
                assert abs(g[key] - e[key]) <= tol, f"{key} {g} vs {e}"
        assert g.get("speaker") == e.get("speaker"), f"speaker {g} vs {e}"


# --- 1. native run_job == the staged native path (with diarization) -----------
@pytest.mark.parametrize("clip", ["en_dialog", "ru_dialog"])
def test_run_job_matches_staged(clip):
    audio = _audio(clip)
    expected = _staged(audio, with_diarizer=True)
    got, stages, durations, resolve_calls = _run_native(clip, with_diarizer=True)

    assert got["language"] == expected["language"]
    assert got["diarized"] is True
    _assert_words_close(got["word_segments"], expected["word_segments"])

    # segment-level speaker labels match exactly.
    assert len(got["segments"]) == len(expected["segments"])
    for gs, es in zip(got["segments"], expected["segments"]):
        assert gs.get("speaker") == es.get("speaker")

    # progress sequence + the once-firing callbacks.
    assert stages == ["decoding", "transcribing", "loading_align", "aligning",
                      "diarizing"]
    assert len(durations) == 1
    assert abs(durations[0] - len(audio) / SAMPLE_RATE) < 1e-6
    assert resolve_calls == [expected["language"]]  # resolved once, for the lang


# --- 2. diarizer-absent path: no diarizing stage, diarized=false --------------
def test_run_job_without_diarizer():
    audio = _audio("en_dialog")
    expected = _staged(audio, with_diarizer=False)
    got, stages, durations, resolve_calls = _run_native("en_dialog",
                                                        with_diarizer=False)

    assert got["diarized"] is False
    assert stages == ["decoding", "transcribing", "loading_align", "aligning"]
    assert len(resolve_calls) == 1
    _assert_words_close(got["word_segments"], expected["word_segments"])
    # no speaker labels anywhere (no diarizer ran).
    assert all("speaker" not in w for w in got["word_segments"])
    assert all("speaker" not in s for s in got["segments"])
