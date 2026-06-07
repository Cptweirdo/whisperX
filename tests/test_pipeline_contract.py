"""Contract tests for the server <-> transcription-engine seam (``app/pipeline.py``).

This is the boundary the C++ core migration strangles stage by stage: the server
(``app/server.py::run_session``) calls ``pipeline.run_job``, which in turn calls
into ``whisperx`` (load_audio / asr.transcribe / load_align_model / align /
diarize / assign_word_speakers) and writes the download artifacts. The migration
plan (docs/cpp-core-migration-plan.md §1) commits to keeping ``run_job``'s shape
exact while its internals go native — so these tests *pin that shape*: the stage
progress sequence, the calls run_job makes into the engine (and with what args),
the result dict it returns, the artifacts it writes, and cancellation semantics.

Deliberately torch-free: ``whisperx`` is import-lazy and ``whisperx.utils`` (the
writers) is stdlib-only, so we stub ``whisperx.audio`` (for SAMPLE_RATE) and
monkeypatch the engine functions. The *real* writers run, so the artifact
contract is exercised end-to-end. No model, no GPU, no web deps required.
"""

from __future__ import annotations

import json
import sys
import threading
import types

import pytest

import whisperx  # cheap: __init__ is pure-lazy (importlib only)
from app import pipeline
from app.jobs import Cancelled
from app.pipeline import ModelBundle


# --- fakes that record what run_job demands of the engine ---------------------

class FakeASR:
    """Stand-in for FasterWhisperPipeline. Records transcribe() args and returns a
    fixed result whose ``language`` is the *detected* one (may differ from the
    requested language) — pinning that run_job trusts the returned language."""

    def __init__(self, detected_language="de", segments=None):
        self.detected_language = detected_language
        self.segments = segments or [{"start": 0.0, "end": 1.5, "text": " Hello there friend."}]
        self.calls: list[dict] = []

    def transcribe(self, audio, batch_size=None, language=None):
        self.calls.append({"audio": audio, "batch_size": batch_size, "language": language})
        return {"segments": self.segments, "language": self.detected_language}


class FakeDiarize:
    """Stand-in for DiarizationPipeline. Records the speaker-bound args and returns
    a sentinel 'turns' object that run_job must hand to assign_word_speakers."""

    SENTINEL = object()

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, audio, min_speakers=None, max_speakers=None):
        self.calls.append({"audio": audio, "min_speakers": min_speakers, "max_speakers": max_speakers})
        return self.SENTINEL


ALIGNED_SEGMENTS = [
    {"start": 0.0, "end": 1.5, "text": " Hello there friend.",
     "words": [
         {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.9},
         {"word": "there", "start": 0.5, "end": 0.9, "score": 0.9},
         {"word": "friend.", "start": 0.9, "end": 1.5, "score": 0.9},
     ]},
]


@pytest.fixture
def engine(monkeypatch):
    """Stub the whisperx engine surface run_job calls. Returns a recorder dict so a
    test can assert exactly how run_job drove the engine."""
    rec: dict = {"load_audio_calls": 0, "align_calls": [], "load_align_calls": [],
                 "assign_calls": []}

    # SAMPLE_RATE is the only thing run_job needs from the (torch-heavy) audio module.
    audio_stub = types.ModuleType("whisperx.audio")
    audio_stub.SAMPLE_RATE = 16000
    monkeypatch.setitem(sys.modules, "whisperx.audio", audio_stub)

    def fake_load_audio(path):
        rec["load_audio_calls"] += 1
        rec["audio"] = [0.0] * 16000  # len/16000 == 1.0s duration, deterministic
        return rec["audio"]

    def fake_load_align_model(language_code=None, device=None):
        rec["load_align_calls"].append({"language_code": language_code, "device": device})
        return ("ALIGN_MODEL", {"language": language_code})

    def fake_align(segments, align_model, align_meta, audio, device):
        rec["align_calls"].append({"segments": segments, "model": align_model,
                                   "meta": align_meta, "audio": audio, "device": device})
        return {"segments": [dict(s) for s in ALIGNED_SEGMENTS],
                "word_segments": [w for s in ALIGNED_SEGMENTS for w in s["words"]]}

    def fake_assign(diarize_turns, result):
        rec["assign_calls"].append({"turns": diarize_turns, "result": result})
        out = dict(result)
        out["segments"] = [{**s, "speaker": "SPEAKER_00"} for s in result["segments"]]
        return out

    monkeypatch.setattr(whisperx, "load_audio", fake_load_audio)
    monkeypatch.setattr(whisperx, "load_align_model", fake_load_align_model)
    monkeypatch.setattr(whisperx, "align", fake_align)
    monkeypatch.setattr(whisperx, "assign_word_speakers", fake_assign)
    return rec


def _bundle(*, device="cpu", diarize=None):
    return ModelBundle(asr=FakeASR(), device=device, diarize=diarize)


# =============================================================================
# run_job — stage progress sequence (the SSE/`mark_stage` contract)
# =============================================================================

def test_run_job_stage_sequence_with_diarization(engine, tmp_path):
    stages: list[str] = []
    bundle = _bundle(diarize=FakeDiarize())
    pipeline.run_job(bundle, "audio.wav", str(tmp_path), progress=stages.append)
    assert stages == ["decoding", "transcribing", "loading_align", "aligning", "diarizing"]


def test_run_job_stage_sequence_without_diarization(engine, tmp_path):
    """No diarizer -> the 'diarizing' stage is skipped and assign_word_speakers is
    never called; result records diarized=False."""
    stages: list[str] = []
    bundle = _bundle(diarize=None)
    result = pipeline.run_job(bundle, "audio.wav", str(tmp_path), progress=stages.append)
    assert stages == ["decoding", "transcribing", "loading_align", "aligning"]
    assert engine["assign_calls"] == []
    assert result["diarized"] is False


# =============================================================================
# run_job — what it demands of the engine, and with which arguments
# =============================================================================

def test_run_job_forwards_batch_size_and_requested_language_to_asr(engine, tmp_path):
    bundle = _bundle()
    pipeline.run_job(bundle, "a.wav", str(tmp_path), language="es")
    (call,) = bundle.asr.calls
    assert call["batch_size"] == pipeline.BATCH_SIZE
    assert call["language"] == "es"
    assert call["audio"] is engine["audio"]  # the once-decoded buffer is reused


def test_run_job_trusts_asr_detected_language_downstream(engine, tmp_path):
    """The requested language is a hint to the ASR; everything after transcription
    (align model selection + final result['language']) uses the *detected* language
    the ASR returned."""
    bundle = ModelBundle(asr=FakeASR(detected_language="de"), device="cpu", diarize=None)
    result = pipeline.run_job(bundle, "a.wav", str(tmp_path), language="es")
    assert engine["load_align_calls"][0]["language_code"] == "de"
    assert result["language"] == "de"


def test_run_job_aligns_with_align_device_not_compute_device(engine, tmp_path):
    """wav2vec2 alignment is forced to CPU on the Apple-Silicon backends (MPS conv1d
    limit). run_job must pass _align_device(bundle.device), not bundle.device, to
    both load_align_model and align."""
    bundle = _bundle(device="whispercpp", diarize=None)
    pipeline.run_job(bundle, "a.wav", str(tmp_path))
    assert pipeline._align_device("whispercpp") == "cpu"
    assert engine["load_align_calls"][0]["device"] == "cpu"
    assert engine["align_calls"][0]["device"] == "cpu"


def test_run_job_align_receives_asr_segments_and_loaded_model(engine, tmp_path):
    bundle = _bundle(diarize=None)
    pipeline.run_job(bundle, "a.wav", str(tmp_path))
    call = engine["align_calls"][0]
    assert call["segments"] is bundle.asr.segments      # ASR output feeds the aligner
    assert call["model"] == "ALIGN_MODEL"
    assert call["meta"] == {"language": "de"}
    assert call["audio"] is engine["audio"]


def test_run_job_diarize_receives_speaker_bounds_and_feeds_assign(engine, tmp_path):
    diar = FakeDiarize()
    bundle = _bundle(diarize=diar)
    pipeline.run_job(bundle, "a.wav", str(tmp_path), min_speakers=2, max_speakers=5)
    (dcall,) = diar.calls
    assert (dcall["min_speakers"], dcall["max_speakers"]) == (2, 5)
    assert dcall["audio"] is engine["audio"]
    # The diarizer's turns and the *aligned* result are what assign_word_speakers gets.
    (acall,) = engine["assign_calls"]
    assert acall["turns"] is FakeDiarize.SENTINEL


# =============================================================================
# run_job — the result dict the server (run_session -> mark_done) consumes
# =============================================================================

def test_run_job_result_contract(engine, tmp_path):
    bundle = _bundle(diarize=FakeDiarize())
    result = pipeline.run_job(bundle, "a.wav", str(tmp_path))
    # Keys run_session + the store + the SPA depend on:
    for key in ("segments", "word_segments", "language", "duration", "num_segments",
                "diarized", "artifacts"):
        assert key in result, f"run_job result missing '{key}'"
    assert result["language"] == "de"
    assert result["duration"] == pytest.approx(1.0)          # len(audio)/SAMPLE_RATE
    assert result["num_segments"] == len(result["segments"])
    assert result["diarized"] is True
    # Diarization annotated the segments with a speaker (assign_word_speakers).
    assert result["segments"][0]["speaker"] == "SPEAKER_00"


def test_run_job_reports_duration_once_after_decode(engine, tmp_path):
    durations: list[float] = []
    order: list[str] = []
    bundle = _bundle(diarize=None)
    pipeline.run_job(
        bundle, "a.wav", str(tmp_path),
        progress=order.append,
        on_duration=lambda d: (durations.append(d), order.append(f"dur={d}")),
    )
    assert durations == [pytest.approx(1.0)]
    # duration is reported after decode, before transcription, so later stages ETA live.
    assert order[0] == "decoding"
    assert order[1] == "dur=1.0"
    assert order[2] == "transcribing"


# =============================================================================
# run_job — artifact writing (the files served by /sessions/<id>/download/<fmt>)
# =============================================================================

def test_run_job_writes_every_output_format_with_deterministic_names(engine, tmp_path):
    bundle = _bundle(diarize=None)
    result = pipeline.run_job(bundle, "a.wav", str(tmp_path), artifact_basename="transcript")
    assert set(result["artifacts"]) == set(pipeline.OUTPUT_FORMATS)
    for fmt in pipeline.OUTPUT_FORMATS:
        path = tmp_path / f"transcript.{fmt}"
        assert path.exists(), f"missing artifact {fmt}"
        assert result["artifacts"][fmt] == str(path)


def test_run_job_artifact_basename_is_honored(engine, tmp_path):
    bundle = _bundle(diarize=None)
    pipeline.run_job(bundle, "/some/where/recording.m4a", str(tmp_path),
                     artifact_basename="transcript")
    # Named off the basename we pass, never off the audio path.
    assert (tmp_path / "transcript.json").exists()
    assert not (tmp_path / "recording.json").exists()


# =============================================================================
# run_job — cancellation (run-to-completion: only at stage boundaries)
# =============================================================================

def test_run_job_cancel_before_start_raises_and_decodes_nothing(engine, tmp_path):
    cancel = threading.Event()
    cancel.set()
    bundle = _bundle(diarize=None)
    with pytest.raises(Cancelled):
        pipeline.run_job(bundle, "a.wav", str(tmp_path), cancel_event=cancel)
    assert engine["load_audio_calls"] == 0   # bailed at the first stage boundary


def test_run_job_cancel_is_honored_at_stage_boundary(engine, tmp_path):
    """Setting cancel during transcription stops the job at the *next* stage
    boundary (no mid-stage abort) — the committed run-to-completion semantics."""
    cancel = threading.Event()

    class CancellingASR(FakeASR):
        def transcribe(self, audio, batch_size=None, language=None):
            cancel.set()  # request cancel mid-pipeline
            return super().transcribe(audio, batch_size=batch_size, language=language)

    bundle = ModelBundle(asr=CancellingASR(), device="cpu", diarize=None)
    with pytest.raises(Cancelled):
        pipeline.run_job(bundle, "a.wav", str(tmp_path), cancel_event=cancel)
    # transcription ran; alignment (the next stage) did not.
    assert bundle.asr.calls
    assert engine["align_calls"] == []


# =============================================================================
# Writer output — byte-level golden (the "writers byte-identical" contract).
# These exact strings are what a C++ writer port must reproduce (plan §3).
# Generated from the real whisperx writers; regenerate deliberately if intended.
# =============================================================================

_GOLDEN_RESULT = {
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 1.5, "text": " Hello there friend.", "speaker": "SPEAKER_00",
         "words": [
             {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.9, "speaker": "SPEAKER_00"},
             {"word": "there", "start": 0.5, "end": 0.9, "score": 0.9, "speaker": "SPEAKER_00"},
             {"word": "friend.", "start": 0.9, "end": 1.5, "score": 0.9, "speaker": "SPEAKER_00"},
         ]},
        {"start": 2.0, "end": 3.0, "text": " Hi.", "speaker": "SPEAKER_01",
         "words": [
             {"word": "Hi.", "start": 2.0, "end": 3.0, "score": 0.9, "speaker": "SPEAKER_01"},
         ]},
    ],
}

_GOLDEN = {
    "srt": ("1\n00:00:00,000 --> 00:00:01,500\n[SPEAKER_00]: Hello there friend.\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\n[SPEAKER_01]: Hi.\n\n"),
    "vtt": ("WEBVTT\n\n00:00.000 --> 00:01.500\n[SPEAKER_00]: Hello there friend.\n\n"
            "00:02.000 --> 00:03.000\n[SPEAKER_01]: Hi.\n\n"),
    "txt": "[SPEAKER_00]: Hello there friend.\n[SPEAKER_01]: Hi.\n",
}


@pytest.mark.parametrize("fmt", ["srt", "vtt", "txt"])
def test_writer_output_is_byte_identical_to_golden(fmt, tmp_path):
    from whisperx.utils import get_writer

    get_writer(fmt, str(tmp_path))(_GOLDEN_RESULT, str(tmp_path / "transcript"),
                                   pipeline.WRITER_OPTIONS)
    assert (tmp_path / f"transcript.{fmt}").read_text(encoding="utf-8") == _GOLDEN[fmt]


def test_writer_json_roundtrips_result_verbatim(tmp_path):
    """The json writer dumps the result dict unchanged (ensure_ascii=False) — the
    canonical transcript.json the store reads back. Compare structurally (key order
    is an implementation detail the store does not rely on)."""
    from whisperx.utils import get_writer

    get_writer("json", str(tmp_path))(_GOLDEN_RESULT, str(tmp_path / "transcript"),
                                      pipeline.WRITER_OPTIONS)
    loaded = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert loaded == _GOLDEN_RESULT


# =============================================================================
# Constants + helpers the server depends on (model/device validation, ETA)
# =============================================================================

def test_whisper_model_whitelist_accepts_listed_rejects_unknown():
    for name in pipeline.WhisperModel.values():
        assert pipeline.WhisperModel(name).value == name  # round-trips
    with pytest.raises(ValueError):
        pipeline.WhisperModel("totally-not-a-model")     # how the API rejects junk
    assert pipeline.WhisperModel.coerce("nope", pipeline.WhisperModel.SMALL) \
        is pipeline.WhisperModel.SMALL


def test_device_and_format_constants_are_pinned():
    assert pipeline.DEVICES == ("cpu", "cuda", "mlx", "whispercpp")
    assert pipeline.OUTPUT_FORMATS == ("srt", "vtt", "txt", "json")
    assert pipeline.WRITER_OPTIONS == {
        "max_line_width": None, "max_line_count": None, "highlight_words": False,
    }


def test_eta_seconds_uses_stage_rtf_and_degrades_gracefully():
    assert pipeline.eta_seconds("transcribing", 100) == pytest.approx(22.0)
    assert pipeline.eta_seconds("aligning", 100) == pytest.approx(19.0)
    assert pipeline.eta_seconds("diarizing", 100) == pytest.approx(51.0)
    # Stages without an RTF, or with no/zero duration, yield no estimate.
    assert pipeline.eta_seconds("decoding", 100) is None
    assert pipeline.eta_seconds("loading_align", 100) is None
    assert pipeline.eta_seconds("transcribing", None) is None
    assert pipeline.eta_seconds("transcribing", 0) is None


# =============================================================================
# ModelManager — the bits the server's model/device endpoints rely on
# =============================================================================

@pytest.fixture
def light_manager(monkeypatch):
    """Patch hardware/keyring probes so ModelManager construction + status() are
    deterministic and free of torch/keyring/network."""
    from app import diarize_model

    monkeypatch.setattr(pipeline, "cuda_available", lambda: False)
    monkeypatch.setattr(pipeline, "mlx_available", lambda: False)
    monkeypatch.setattr(pipeline, "whispercpp_available", lambda: False)
    monkeypatch.setattr(pipeline, "_resolve_hf_token", lambda: None)
    monkeypatch.setattr(diarize_model, "resolve_local_model", lambda: None)
    monkeypatch.setattr(diarize_model, "derive_version", lambda _l: None)


def test_resolve_device_falls_back_to_cpu(light_manager):
    assert pipeline.ModelManager(device="cuda").device == "cpu"   # no GPU here
    assert pipeline.ModelManager(device="bogus").device == "cpu"  # unknown
    assert pipeline.ModelManager(device="cpu").device == "cpu"


def test_status_shape_is_pinned(light_manager):
    mgr = pipeline.ModelManager(active="small", device="cpu")
    st = mgr.status()
    assert set(st) == {
        "active", "device", "cuda_available", "mlx_available", "whispercpp_available",
        "diarize", "diarize_error", "diarize_available", "diarize_source",
        "diarize_version", "diarize_token", "models",
    }
    assert st["active"] == "small"
    assert st["device"] == "cpu"
    # One entry per whitelisted model, each with the badge fields the SPA renders.
    assert [m["name"] for m in st["models"]] == pipeline.WhisperModel.values()
    for m in st["models"]:
        assert set(m) == {"name", "loaded", "loading", "error"}


def test_bundle_for_shares_the_align_cache_and_current_device(light_manager, monkeypatch):
    """bundle_for wires the requested ASR with the *shared* align cache + diarizer so
    align models load once across jobs (the streamlining invariant)."""
    asr_sentinel = object()
    diar_sentinel = object()
    monkeypatch.setattr(pipeline.ModelManager, "load_asr", lambda self, name: asr_sentinel)
    monkeypatch.setattr(pipeline.ModelManager, "ensure_diarize", lambda self: diar_sentinel)

    mgr = pipeline.ModelManager(active="small", device="cpu")
    bundle = mgr.bundle_for("small")
    assert bundle.asr is asr_sentinel
    assert bundle.diarize is diar_sentinel
    assert bundle.device == "cpu"
    assert bundle._align_cache is mgr._align_cache  # identity: one shared cache


def test_set_active_validates_and_updates(light_manager, monkeypatch):
    monkeypatch.setattr(pipeline.ModelManager, "warm", lambda self, name: None)  # no thread
    mgr = pipeline.ModelManager(active="small", device="cpu")
    status = mgr.set_active("base")
    assert mgr.active == "base"
    assert status["active"] == "base"
    with pytest.raises(ValueError):
        mgr.set_active("not-a-model")
