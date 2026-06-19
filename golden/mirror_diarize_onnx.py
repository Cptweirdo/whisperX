#!/usr/bin/env python3
"""Mirror + sha-pin sherpa-onnx's pre-exported diarization ONNX (Phase 4 / slice 4b).

**Re-host, don't hot-link.** sherpa-onnx publishes the pre-exported
``pyannote-segmentation-3.0`` segmentation model and the ``wespeaker_en_voxceleb
CAM++`` speaker-embedding extractor as ONNX in its model zoo. Per the 3B mirror
decision we **re-host the two we pin to a repo we control** (default
``KonstantK/diarize-onnx-sherpa``), sha-pinned, and pull them at runtime via
``huggingface_hub`` (the same cache/sha path ``load_align_model`` uses) — never
hot-linking an upstream URL that can move. No torch/export step: sherpa's files are
already ONNX, so this is download → ``meta.json`` (filenames + ``embedding_dim`` +
sha256 + source + license) → optional self-check → upload.

Both upstream toolkits are **Apache-2.0** (3D-Speaker / wespeaker); pyannote
segmentation-3.0 is MIT. **Runtime consumes only paths** to the two assets; the
producer (this script) is swappable without touching ``whisperx/diarize_sherpa.py``
or the C++ ``SherpaDiarizer``.

Run (heavy deps ephemeral, same convention as ``mirror_whisper_onnx.py``)::

    # publish (self-check is on by default; needs a built whisperx_core for it):
    uv run --no-project --with huggingface_hub --with requests --with numpy \
        python golden/mirror_diarize_onnx.py

    # validate locally without publishing:
    PYTHONPATH=build python golden/mirror_diarize_onnx.py --no-upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO_DEFAULT = "KonstantK/diarize-onnx-sherpa"
CONTRACT_VERSION = 1

SEG_RELEASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBED_RELEASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx"
)
# Names the assets land under in the mirror repo (meta.json points at these).
SEG_NAME = "pyannote-segmentation-3-0.onnx"
EMBED_NAME = "wespeaker_en_voxceleb_CAM++.onnx"
LICENSE = "segmentation: MIT (pyannote-3.0); embedding: Apache-2.0 (wespeaker)"

CHECK_CLIP = "en_dialog"  # a CC0 synthetic dialog; proves the assets diarize >=2 spk


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    import requests

    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def _fetch_assets(tmp: Path) -> tuple[Path, Path]:
    """Download + stage (segmentation.onnx, embedding.onnx) into `tmp`."""
    tar = tmp / "seg.tar.bz2"
    _download(SEG_RELEASE, tar)
    with tarfile.open(tar, "r:bz2") as tf:
        tf.extractall(tmp)
    seg_src = next((tmp).glob("sherpa-onnx-pyannote-segmentation-3-0/*.onnx"))
    seg = tmp / SEG_NAME
    shutil.copyfile(seg_src, seg)

    embed = tmp / EMBED_NAME
    _download(EMBED_RELEASE, embed)
    return seg, embed


def _self_check(seg: Path, embed: Path) -> tuple[bool, int]:
    """Diarize a CC0 dialog through the native SherpaDiarizer; assert >=2 speakers +
    a sane embedding dim. Skips (returns ok) if whisperx_core isn't built with audio."""
    try:
        import whisperx_core as wc
    except ImportError:
        print("  self-check: whisperx_core not importable — skipped")
        return True, 0
    if not hasattr(wc, "SherpaDiarizer"):
        print("  self-check: module built without WHISPERX_CORE_AUDIO — skipped")
        return True, 0
    wav = ROOT / "golden" / "clips" / f"{CHECK_CLIP}.wav"
    if not wav.exists():
        print(f"  self-check: no clip {CHECK_CLIP} — skipped")
        return True, 0

    import numpy as np

    with wave.open(str(wav), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0

    m = wc.SherpaDiarizer(segmentation=str(seg), embedding=str(embed), num_threads=4)
    segs = m.diarize(audio, 0)
    n_spk = len({s["speaker"] for s in segs})
    dim = m.embedding_dim()
    ok = n_spk >= 2 and dim > 0
    print(f"  self-check {CHECK_CLIP}: {len(segs)} turns, {n_spk} speakers, "
          f"embed_dim={dim} ({'ok' if ok else 'FAIL'})")
    return ok, dim


def _remote_meta(repo: str):
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(repo, "meta.json", repo_type="model")
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def mirror(repo: str, *, do_upload: bool, do_check: bool, force: bool) -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        seg, embed = _fetch_assets(tmp)
        seg_sha = _sha256(seg)

        if do_upload and not force:
            rmeta = _remote_meta(repo)
            if rmeta and rmeta.get("contract_version") == CONTRACT_VERSION and \
                    rmeta.get("segmentation_sha256") == seg_sha:
                print(f"  already published on {repo} (seg sha {seg_sha[:12]}) — "
                      f"skip (use --force)")
                return

        embed_dim = 0
        if do_check:
            ok, embed_dim = _self_check(seg, embed)
            if not ok:
                raise SystemExit("self-check FAILED — refusing to upload")

        meta = {
            "contract_version": CONTRACT_VERSION,
            "segmentation": SEG_NAME,
            "embedding": EMBED_NAME,
            "embedding_dim": embed_dim,
            "source": {"segmentation": SEG_RELEASE, "embedding": EMBED_RELEASE},
            "license": LICENSE,
            "segmentation_sha256": seg_sha,
            "embedding_sha256": _sha256(embed),
        }
        (tmp / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        print(f"  assets: {SEG_NAME} / {EMBED_NAME}  embed_dim={embed_dim}")

        if not do_upload:
            print("  --no-upload: kept local only")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(
            repo_id=repo, repo_type="model", folder_path=str(tmp),
            allow_patterns=[SEG_NAME, EMBED_NAME, "meta.json"],
            commit_message=f"mirror sherpa diarization ONNX (seg sha {seg_sha[:12]})",
        )
        print(f"  uploaded -> {repo}/")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO_DEFAULT, help="HF model repo to publish to")
    ap.add_argument("--no-upload", action="store_true",
                    help="download + meta + self-check only, do not push")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the SherpaDiarizer self-check before upload")
    ap.add_argument("--force", action="store_true",
                    help="upload even if the remote sha already matches")
    args = ap.parse_args()

    print(f"repo={args.repo}  upload={not args.no_upload}  check={not args.no_check}")
    mirror(args.repo, do_upload=not args.no_upload, do_check=not args.no_check,
           force=args.force)
    print("done.")


if __name__ == "__main__":
    main()
