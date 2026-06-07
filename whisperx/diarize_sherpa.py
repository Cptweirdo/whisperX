"""sherpa-onnx speaker-diarization backend (Phase 4 / slice 4b).

A drop-in alternative to the pyannote ``community-1`` path in
:mod:`whisperx.diarize`, reached when ``diarize`` is in ``WHISPERX_CORE_STAGES``
(the strangler token — pyannote stays the default oracle). Same ``__call__``
contract: a pandas DataFrame ``[segment, label, speaker, start, end]`` (so
:func:`whisperx.diarize.assign_word_speakers` is untouched), plus the optional
``return_embeddings`` second value.

This is **A/B, not parity** by construction: sherpa uses pyannote-segmentation-3.0
+ a speaker-embedding extractor (wespeaker_en_voxceleb CAM++) + cosine
FastClustering, whereas community-1 is pyannote.audio 4.0 with VBxClustering/PLDA.
Different models ⇒ different turns; quality is judged by speaker-count + DER vs
ground-truth RTTM, and the model-independent assignment glue is parity-tested
separately.

The two ONNX assets (segmentation + embedding) are pulled from our **sha-pinned
mirror** via ``huggingface_hub`` (the 3B mechanism), from a local directory when
``WHISPERX_DIARIZE_ONNX_DIR`` is set (dev/CI), or from sherpa-onnx's official model
releases as a fallback. ``whisperx_core`` is imported lazily so ``import whisperx``
stays cheap and cross-platform.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.log_utils import get_logger
from whisperx.schema import ProgressCallback

logger = get_logger(__name__)

# HF repo holding the parity-pinned sherpa diarization ONNX (re-hosted from the
# sherpa-onnx model zoo; see golden/mirror_diarize_onnx.py): segmentation +
# embedding + meta.json naming them.
MIRROR_REPO = "KonstantK/diarize-onnx-sherpa"

# sherpa-onnx hosts these as release assets — the runtime fallback. (The
# "recongition" misspelling is upstream's actual release tag.)
SEG_RELEASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBED_RELEASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx"
)


def _speaker_label(cluster_id: int) -> str:
    """Cluster id -> pyannote-style label, matching the DataFrame the oracle emits."""
    return f"SPEAKER_{cluster_id:02d}"


def _pick_seg(files: list[Path]) -> Path:
    for f in files:
        if f.suffix == ".onnx" and ("seg" in f.name.lower()):
            return f
    raise FileNotFoundError(f"no segmentation .onnx among {[f.name for f in files]}")


def _pick_embed(files: list[Path], seg: Path) -> Path:
    onnx = [f for f in files if f.suffix == ".onnx" and f != seg]
    if not onnx:
        raise FileNotFoundError(f"no embedding .onnx among {[f.name for f in files]}")
    return onnx[0]


def _paths_from_dir(d: Path) -> tuple[str, str]:
    """(segmentation, embedding) ONNX paths from a directory of sherpa assets."""
    files = [p for p in d.iterdir() if p.is_file()]
    seg = _pick_seg(files)
    embed = _pick_embed(files, seg)
    return str(seg), str(embed)


def _resolve_diarize_assets(
    download_root: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve (segmentation_onnx, embedding_onnx).

    Order: (1) an explicit local directory (``WHISPERX_DIARIZE_ONNX_DIR``, for
    dev/CI), (2) our sha-pinned HF mirror, (3) sherpa-onnx's official release
    assets (cached) — so the diarizer always loads.
    """
    local = os.environ.get("WHISPERX_DIARIZE_ONNX_DIR")
    if local:
        d = Path(local)
        logger.info("Using local sherpa diarization assets: %s", d)
        return _paths_from_dir(d)

    try:
        return _from_mirror(download_root)
    except Exception as e:  # noqa: BLE001 - not mirrored / unreachable -> fall back
        logger.info(
            "sherpa diarization assets not on mirror %s (%s); using sherpa-onnx release",
            MIRROR_REPO, e,
        )
    return _fetch_releases(download_root)


def _from_mirror(download_root: Optional[str]) -> tuple[str, str]:
    """Pull the assets from our sha-pinned HF mirror; meta.json names them."""
    import json

    from huggingface_hub import hf_hub_download  # lazy

    def fetch(fname: str) -> str:
        return hf_hub_download(
            MIRROR_REPO, fname, repo_type="model", cache_dir=download_root
        )

    logger.info("Pulling sherpa diarization assets from mirror %s", MIRROR_REPO)
    meta = json.loads(Path(fetch("meta.json")).read_text())
    return fetch(meta["segmentation"]), fetch(meta["embedding"])


def _fetch_releases(download_root: Optional[str] = None) -> tuple[str, str]:
    """Download sherpa-onnx's official segmentation tarball + embedding ONNX into a
    cache dir (skips what's already present); returns (segmentation, embedding)."""
    import shutil
    import tarfile
    import tempfile
    import urllib.request

    cache = Path(
        download_root
        or os.environ.get("WHISPERX_SHERPA_CACHE")
        or (Path.home() / ".cache" / "whisperx-sherpa")
    )
    cache.mkdir(parents=True, exist_ok=True)

    seg_dir = cache / "sherpa-onnx-pyannote-segmentation-3-0"
    if not seg_dir.is_dir():
        logger.info("Downloading sherpa segmentation from %s", SEG_RELEASE)
        with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as tf:
            tmp = tf.name
            with urllib.request.urlopen(SEG_RELEASE) as r:  # noqa: S310 - pinned release
                shutil.copyfileobj(r, tf)
        try:
            with tarfile.open(tmp, "r:bz2") as t:
                t.extractall(cache)
        finally:
            os.unlink(tmp)
    seg = seg_dir / "model.onnx"
    if not seg.exists():
        # some tarballs ship a different inner name — pick the first onnx
        seg = next(seg_dir.glob("*.onnx"))

    embed = cache / "wespeaker_en_voxceleb_CAM++.onnx"
    if not embed.exists():
        logger.info("Downloading sherpa embedding from %s", EMBED_RELEASE)
        with urllib.request.urlopen(EMBED_RELEASE) as r:  # noqa: S310 - pinned release
            embed.write_bytes(r.read())
    return str(seg), str(embed)


class SherpaDiarizationPipeline:
    """Speaker diarization via sherpa-onnx (ONNX Runtime), mirroring the public
    surface of :class:`whisperx.diarize.DiarizationPipeline`: a callable returning
    the same ``[segment, label, speaker, start, end]`` DataFrame (+ optional
    per-speaker embeddings)."""

    def __init__(self, model, threshold: float = 0.5):
        self.model = model  # whisperx_core.SherpaDiarizer
        self.threshold = threshold

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
        progress_callback: ProgressCallback = None,
    ) -> Union[
        tuple[pd.DataFrame, Optional[dict[str, list[float]]]], pd.DataFrame
    ]:
        from whisperx.diarize import Segment  # reuse the start/end holder

        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = np.asarray(audio, dtype=np.float32)

        # Speaker-count controls onto FastClustering's single num_clusters
        # (target) XOR auto-threshold: num -> max -> min -> auto. A lone max/min is
        # fed as num_clusters (sherpa has no native min/max range). num_clusters is
        # a clustering *target* — the pyannote impl's frame-level finalization can
        # yield fewer speakers — not a hard guarantee. 0 = auto (count from the
        # cosine threshold).
        if num_speakers:
            num_clusters = int(num_speakers)
        elif max_speakers:
            num_clusters = int(max_speakers)
        elif min_speakers:
            num_clusters = int(min_speakers)
        else:
            num_clusters = 0
        if num_clusters and (min_speakers or max_speakers) and not num_speakers:
            logger.info(
                "sherpa diarization has no min/max range; targeting num_clusters=%d "
                "(final count may be fewer).", num_clusters,
            )

        if progress_callback is not None:
            progress_callback(0.0)
        raw = self.model.diarize(audio, num_clusters)
        if progress_callback is not None:
            progress_callback(99.0)

        rows = []
        for s in raw:
            label = _speaker_label(s["speaker"])
            rows.append({
                "segment": Segment(s["start"], s["end"], label),
                "label": label,
                "speaker": label,
                "start": s["start"],
                "end": s["end"],
            })
        diarize_df = pd.DataFrame(
            rows, columns=["segment", "label", "speaker", "start", "end"])

        if progress_callback is not None:
            progress_callback(100.0)

        if return_embeddings:
            emb = self.model.embeddings(audio, raw) if raw else {}
            speaker_embeddings = {
                _speaker_label(k): list(v) for k, v in emb.items()
            }
            return diarize_df, speaker_embeddings or None
        return diarize_df


def load_sherpa_diarize_model(
    *,
    threads: int = 1,
    threshold: float = 0.5,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
    download_root: Optional[str] = None,
) -> SherpaDiarizationPipeline:
    """Build a :class:`SherpaDiarizationPipeline` (sherpa-onnx diarization on ORT)."""
    import whisperx_core  # lazy — only when the diarize token actually loads a model

    seg, embed = _resolve_diarize_assets(download_root)
    logger.info("Loading sherpa diarization (seg=%s, embed=%s)",
                Path(seg).name, Path(embed).name)
    model = whisperx_core.SherpaDiarizer(
        segmentation=seg,
        embedding=embed,
        num_threads=threads,
        provider="cpu",
        threshold=threshold,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    return SherpaDiarizationPipeline(model=model, threshold=threshold)
