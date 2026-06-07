"""sherpa-onnx Whisper ASR backend (Phase 4 / slice 4a).

A drop-in alternative to the faster-whisper path in :mod:`whisperx.asr`, reached
when ``asr`` is in ``WHISPERX_CORE_STAGES`` (the strangler token — faster-whisper
stays the default + WER/CER oracle). Same ``transcribe`` return shape
(``{"segments", "language"}``) so the downstream align/diarize stages are untouched.

Structurally a sibling of :mod:`whisperx.asr_mlx` / :mod:`whisperx.asr_whispercpp`:
the identical VAD-segment serial loop, swapping the decode engine for the native
``whisperx_core.WhisperSherpa`` (sherpa-onnx Whisper on ONNX Runtime). Per the
**decoupled-goldens** decision the text is judged by WER/CER, not byte-parity with
CTranslate2 — greedy vs beam search is a non-issue.

The Whisper ONNX assets (encoder/decoder/tokens) are pulled from our **sha-pinned
mirror** via ``huggingface_hub`` (the 3B mechanism), or from a local directory when
``WHISPERX_SHERPA_WHISPER_DIR`` is set (dev/CI before the mirror is published).
``whisperx_core`` is imported lazily so ``import whisperx`` stays cheap and
cross-platform — nothing here touches it until a sherpa model is loaded/run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio
from whisperx.log_utils import get_logger
from whisperx.schema import ProgressCallback, SingleSegment, TranscriptionResult
from whisperx.vads import Pyannote, Silero, Vad

logger = get_logger(__name__)

# HF repo holding the parity-pinned sherpa Whisper ONNX exports (re-hosted from
# the sherpa-onnx model zoo; see golden/mirror_whisper_onnx.py). One folder per
# model name, each with encoder/decoder/tokens (+ meta.json). Common models are
# mirrored here for pin control; anything not on the mirror falls back to sherpa's
# official release tarball at runtime (cached), so every model still loads.
MIRROR_REPO = "KonstantK/whisper-onnx-sherpa"

# sherpa-onnx hosts its pre-exported Whisper models as release tarballs here — the
# runtime fallback for models we haven't mirrored.
SHERPA_RELEASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-whisper-{model}.tar.bz2"
)

# Mel bin count by Whisper family (the C++ feat_config.feature_dim): large-v3 and
# the turbo derived from it use 128 mels; everything else uses 80.
FEATURE_DIM = {"large-v3": 128, "large-v3-turbo": 128}

# Short WhisperX names -> mirror folder keys. Only `tiny` is published today; the
# rest map to their natural folder name and lazy-download once mirrored.
SHERPA_MODEL_MAP = {
    "tiny": "tiny",
    "tiny.en": "tiny.en",
    "base": "base",
    "base.en": "base.en",
    "small": "small",
    "small.en": "small.en",
    "medium": "medium",
    "medium.en": "medium.en",
    "large": "large-v3",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "turbo": "large-v3-turbo",
}


def _resolve_model_name(whisper_arch: str) -> str:
    if whisper_arch in SHERPA_MODEL_MAP:
        mapped = SHERPA_MODEL_MAP[whisper_arch]
        if mapped != whisper_arch:
            logger.info("Mapping model '%s' -> sherpa Whisper '%s'", whisper_arch, mapped)
        return mapped
    logger.warning(
        "Unknown model '%s' for the sherpa Whisper backend; trying it verbatim. "
        "Known: %s", whisper_arch, list(SHERPA_MODEL_MAP),
    )
    return whisper_arch


def _pick(files: list[Path], needle: str) -> Path:
    """The asset whose name contains `needle`, preferring the fp32 (non-int8) one
    for parity stability (sherpa publishes both `X.onnx` and `X.int8.onnx`)."""
    matches = [f for f in files if needle in f.name and f.suffix in (".onnx", ".txt")]
    if not matches:
        raise FileNotFoundError(f"no '{needle}' asset among {[f.name for f in files]}")
    matches.sort(key=lambda f: ("int8" in f.name, len(f.name)))
    return matches[0]


def _paths_from_dir(d: Path) -> tuple[str, str, str]:
    """(encoder, decoder, tokens) from a directory of sherpa Whisper assets."""
    files = [p for p in d.iterdir() if p.is_file()]
    enc = _pick([f for f in files if f.suffix == ".onnx"], "encoder")
    dec = _pick([f for f in files if f.suffix == ".onnx"], "decoder")
    tok = _pick([f for f in files if f.suffix == ".txt"], "tokens")
    return str(enc), str(dec), str(tok)


def _resolve_sherpa_assets(
    whisper_arch: str, download_root: Optional[str] = None
) -> tuple[str, str, str, int]:
    """Resolve a Whisper name to (encoder, decoder, tokens, feature_dim).

    Order: (1) an explicit local directory (``WHISPERX_SHERPA_WHISPER_DIR``, for
    dev/CI), (2) our sha-pinned HF mirror, (3) sherpa-onnx's official release
    tarball (cached) for anything not on the mirror — so every model still loads.
    """
    local = os.environ.get("WHISPERX_SHERPA_WHISPER_DIR")
    if local:
        d = Path(local)
        logger.info("Using local sherpa Whisper assets: %s", d)
        enc, dec, tok = _paths_from_dir(d)
        return enc, dec, tok, _feature_dim(d)

    name = _resolve_model_name(whisper_arch)
    try:
        return _from_mirror(name, download_root)
    except Exception as e:  # noqa: BLE001 - not mirrored / unreachable -> fall back
        logger.info(
            "sherpa Whisper '%s' not on mirror %s (%s); using sherpa-onnx release",
            name, MIRROR_REPO, e,
        )
    d = _fetch_sherpa_release(name, download_root)
    enc, dec, tok = _paths_from_dir(d)
    return enc, dec, tok, FEATURE_DIM.get(name, 80)


def _from_mirror(name: str, download_root: Optional[str]) -> tuple[str, str, str, int]:
    """Pull a model from our sha-pinned HF mirror; meta.json names the assets."""
    import json

    from huggingface_hub import hf_hub_download  # lazy

    def fetch(fname: str) -> str:
        return hf_hub_download(
            MIRROR_REPO, f"{name}/{fname}", repo_type="model", cache_dir=download_root
        )

    logger.info("Pulling sherpa Whisper '%s' from mirror %s", name, MIRROR_REPO)
    meta = json.loads(Path(fetch("meta.json")).read_text())
    return (fetch(meta["encoder"]), fetch(meta["decoder"]), fetch(meta["tokens"]),
            int(meta.get("feature_dim", 80)))


def _fetch_sherpa_release(name: str, download_root: Optional[str] = None) -> Path:
    """Download + extract sherpa-onnx's official Whisper release tarball into a cache
    dir (skips if already present); returns the extracted model directory."""
    import shutil
    import tarfile
    import tempfile
    import urllib.request

    cache = Path(
        download_root
        or os.environ.get("WHISPERX_SHERPA_CACHE")
        or (Path.home() / ".cache" / "whisperx-sherpa")
    )
    out = cache / f"sherpa-onnx-whisper-{name}"
    if out.is_dir() and any(out.glob("*encoder*.onnx")):
        return out

    cache.mkdir(parents=True, exist_ok=True)
    url = SHERPA_RELEASE.format(model=name)
    logger.info("Downloading sherpa-onnx Whisper '%s' from %s", name, url)
    with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as tf:
        tmp = tf.name
        with urllib.request.urlopen(url) as r:  # noqa: S310 - pinned github release
            shutil.copyfileobj(r, tf)
    try:
        with tarfile.open(tmp, "r:bz2") as t:
            t.extractall(cache)
    finally:
        os.unlink(tmp)
    if not out.is_dir():
        raise FileNotFoundError(
            f"sherpa release for '{name}' did not extract to {out}")
    return out


def _feature_dim(d: Path) -> int:
    meta = d / "meta.json"
    if meta.exists():
        import json

        return int(json.loads(meta.read_text()).get("feature_dim", 80))
    return 80


class SherpaWhisperPipeline:
    """WhisperX transcription pipeline using sherpa-onnx Whisper (ONNX Runtime).

    Mirrors the public surface of :class:`whisperx.asr.FasterWhisperPipeline` the
    rest of the pipeline relies on: a ``transcribe`` returning a
    :class:`~whisperx.schema.TranscriptionResult`. No batching — VAD segments are
    transcribed serially (Whisper pads each to a fixed 30 s mel, so there is no
    cross-segment batch-norm hazard).
    """

    def __init__(
        self,
        model,  # whisperx_core.WhisperSherpa
        vad,
        vad_params: dict,
        language: Optional[str] = None,
        task: str = "transcribe",
    ):
        self.model = model
        self.vad_model = vad
        self._vad_params = vad_params
        self.preset_language = language
        self.task = task

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size: int = 30,
        print_progress: bool = False,
        combined_progress: bool = False,
        verbose: bool = False,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = np.asarray(audio, dtype=np.float32)

        effective_task = task or self.task

        # VAD + merge_chunks — identical to asr.py:222-235 (covers Silero/Pyannote).
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks

        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        # Detect once over the first 30 s of the whole clip (faster-whisper does the
        # same — asr.py:300-312), then force that language on every chunk.
        effective_language = language or self.preset_language
        if effective_language is None and vad_segments:
            effective_language = self.model.detect_language(audio[:N_SAMPLES]) or None
            if effective_language:
                logger.info("sherpa Whisper detected language: %s", effective_language)

        segments: list[SingleSegment] = []
        total_segments = len(vad_segments)
        lang = effective_language or ""
        for idx, vad_seg in enumerate(vad_segments):
            f1 = int(vad_seg["start"] * SAMPLE_RATE)
            f2 = int(vad_seg["end"] * SAMPLE_RATE)
            chunk = audio[f1:f2]
            (out,) = self.model.transcribe(
                chunk, [(0.0, len(chunk) / SAMPLE_RATE)], lang, effective_task
            )
            text = out["text"].strip()

            if verbose:
                print(
                    f"Transcript: [{round(vad_seg['start'], 3)} --> "
                    f"{round(vad_seg['end'], 3)}] {text}"
                )
            if print_progress:
                base = ((idx + 1) / total_segments) * 100
                pct = base / 2 if combined_progress else base
                print(f"Progress: {pct:.2f}%...")
            if progress_callback is not None:
                progress_callback(((idx + 1) / total_segments) * 100)

            segments.append(
                {
                    "text": text,
                    "start": round(vad_seg["start"], 3),
                    "end": round(vad_seg["end"], 3),
                    "avg_logprob": float(out["avg_logprob"]),
                }
            )

        return {"segments": segments, "language": effective_language or "en"}

    def detect_language(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        return self.model.detect_language(audio[:N_SAMPLES]) or "en"


def load_sherpa_model(
    whisper_arch: str,
    *,
    device: str = "cpu",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad] = None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    task: str = "transcribe",
    download_root: Optional[str] = None,
    threads: int = 4,
) -> SherpaWhisperPipeline:
    """Build a :class:`SherpaWhisperPipeline` (sherpa-onnx Whisper on ORT).

    The VAD torch model runs on ``device`` (cpu/cuda); the ASR runs on ORT/CPU.
    """
    import whisperx_core  # lazy — only when the asr token actually loads a model

    enc, dec, tok, feature_dim = _resolve_sherpa_assets(whisper_arch, download_root)
    logger.info("Loading sherpa Whisper model: %s", whisper_arch)
    model = whisperx_core.WhisperSherpa(
        encoder=enc,
        decoder=dec,
        tokens=tok,
        num_threads=threads,
        feature_dim=feature_dim,
        language=language or "",
        task=task,
    )

    default_vad_options = {"chunk_size": 30, "vad_onset": 0.500, "vad_offset": 0.363}
    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_model is not None:
        logger.info("Use manually assigned vad_model. vad_method is ignored.")
        resolved_vad = vad_model
    elif vad_method == "silero":
        resolved_vad = Silero(**default_vad_options)
    elif vad_method == "pyannote":
        import torch

        device_vad = "cuda:0" if device == "cuda" else device
        resolved_vad = Pyannote(torch.device(device_vad), token=None, **default_vad_options)
    else:
        raise ValueError(f"Invalid vad_method: {vad_method}")

    return SherpaWhisperPipeline(
        model=model,
        vad=resolved_vad,
        vad_params=default_vad_options,
        language=language,
        task=task,
    )
