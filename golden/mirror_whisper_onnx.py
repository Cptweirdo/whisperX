#!/usr/bin/env python3
"""Mirror + sha-pin sherpa-onnx's pre-exported Whisper ONNX (Phase 4 / slice 4a).

**Re-host, don't hot-link.** sherpa-onnx publishes pre-exported Whisper ONNX
(``<model>-encoder.onnx`` / ``<model>-decoder.onnx`` / ``<model>-tokens.txt``) in its
release assets. Per the 3B mirror decision we **re-host the ones we pin to a repo we
control** (default ``KonstantK/whisper-onnx-sherpa``), sha-pinned, and pull them at
runtime via ``huggingface_hub`` (the same cache/sha path ``load_align_model`` uses) —
never hot-linking an upstream URL that can move. No torch/export step: sherpa's files
are already ONNX, so this is download → ``meta.json`` (filenames + ``feature_dim`` +
sha256 + source) → optional self-check → upload.

**Runtime consumes only a path** to the three assets; the producer (this script) is
swappable without touching ``whisperx/asr_sherpa.py`` or the C++ ``WhisperSherpa``.

Run (heavy deps ephemeral, same convention as ``export_align_onnx.py``)::

    # publish the golden/CI model (self-check is on by default):
    uv run --no-project --with huggingface_hub --with requests \
        python golden/mirror_whisper_onnx.py --model tiny

    # validate locally without publishing (needs a built whisperx_core for the check):
    WHISPERX_SHERPA_WHISPER_DIR=/path/to/sherpa-onnx-whisper-tiny \
        PYTHONPATH=build python golden/mirror_whisper_onnx.py --model tiny --no-upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO_DEFAULT = "KonstantK/whisper-onnx-sherpa"
# sherpa-onnx hosts the pre-exported Whisper models as release tarballs here.
SHERPA_RELEASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-whisper-{model}.tar.bz2"
)
CONTRACT_VERSION = 1  # bumped if the published layout (meta.json keys) changes.
# Mel bin count by Whisper family (the C++ feat_config.feature_dim). large-v3 (and
# turbo, derived from it) use 128 mels; everything else uses 80.
FEATURE_DIM = {"large-v3": 128, "large-v3-turbo": 128, "turbo": 128}

# sherpa's release names the turbo export plain "turbo"; we publish it under its
# canonical name. Download-side alias only — the mirror folder key stays canonical.
SHERPA_RELEASE_NAME = {"large-v3-turbo": "turbo"}

# A golden clip for the optional self-check (proves the mirrored model loads through
# WhisperSherpa and produces non-empty English text before upload). en_libri works
# for any multilingual model; .en-only models would need an English clip too.
CHECK_CLIP = {
    "tiny": "en_libri", "base": "en_libri", "small": "en_libri",
    "medium": "en_libri", "large-v1": "en_libri", "large-v2": "en_libri",
    "large-v3": "en_libri", "large-v3-turbo": "en_libri",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _feature_dim(model: str) -> int:
    return FEATURE_DIM.get(model, 80)


def _pick(names: list[str], needle: str, suffix: str) -> str:
    """The fp32 (non-int8) asset whose name contains `needle` and ends with `suffix`."""
    cand = [n for n in names if needle in n and n.endswith(suffix)]
    if not cand:
        raise FileNotFoundError(f"no '{needle}{suffix}' among {names}")
    cand.sort(key=lambda n: ("int8" in n, len(n)))  # prefer fp32, then shortest
    return cand[0]


def _versions() -> dict:
    import importlib.metadata as md

    out = {}
    for pkg in ("huggingface_hub",):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def _download_sherpa(model: str, dest: Path) -> Path:
    """Download + extract the sherpa release tarball into `dest`/sherpa-onnx-whisper-<rel>."""
    import requests

    rel = SHERPA_RELEASE_NAME.get(model, model)
    url = SHERPA_RELEASE.format(model=rel)
    tar_path = dest / f"whisper-{rel}.tar.bz2"
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with tar_path.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with tarfile.open(tar_path, "r:bz2") as tf:
        tf.extractall(dest)
    out = dest / f"sherpa-onnx-whisper-{rel}"
    if not out.is_dir():
        # some tarballs use a flat layout; fall back to the extract root
        out = dest
    return out


def _self_check(folder: Path, encoder: str, decoder: str, tokens: str,
                feature_dim: int, model: str) -> bool:
    """Load the mirrored assets through the native WhisperSherpa and assert a golden
    clip transcribes to non-empty text + the right language. Skips if whisperx_core
    isn't built with WHISPERX_CORE_AUDIO or the clip is absent."""
    try:
        import whisperx_core as wc
    except ImportError:
        print("  self-check: whisperx_core not importable — skipped")
        return True
    if not hasattr(wc, "WhisperSherpa"):
        print("  self-check: module built without WHISPERX_CORE_AUDIO — skipped")
        return True
    clip = CHECK_CLIP.get(model)
    wav = ROOT / "golden" / "clips" / f"{clip}.wav" if clip else None
    if not (clip and wav.exists()):
        print(f"  self-check: no golden clip for {model!r} — skipped")
        return True

    from whisperx.audio import load_audio, SAMPLE_RATE

    m = wc.WhisperSherpa(
        encoder=str(folder / encoder), decoder=str(folder / decoder),
        tokens=str(folder / tokens), num_threads=4, feature_dim=feature_dim,
    )
    audio = load_audio(str(wav))
    lang = m.detect_language(audio[: SAMPLE_RATE * 30])
    out = m.transcribe(audio, [(0.0, len(audio) / SAMPLE_RATE)], "", "transcribe")
    text = out[0]["text"].strip() if out else ""
    ok = bool(text) and lang == clip.split("_", 1)[0]
    print(f"  self-check {clip}: lang={lang!r} text={text[:60]!r} "
          f"({'ok' if ok else 'FAIL'})")
    return ok


def _remote_meta(repo: str, model: str):
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(repo, f"{model}/meta.json", repo_type="model")
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _already_published(repo: str, model: str, src_sha: str) -> bool:
    rmeta = _remote_meta(repo, model)
    return bool(rmeta) and rmeta.get("contract_version") == CONTRACT_VERSION and \
        rmeta.get("encoder_sha256") == src_sha


def mirror_one(model: str, *, repo: str, do_upload: bool, do_check: bool,
               force: bool) -> None:
    print(f"\n=== {model} ===")
    feature_dim = _feature_dim(model)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Prefer a local sherpa dir (dev/offline) over re-downloading the tarball.
        import os

        local = os.environ.get("WHISPERX_SHERPA_WHISPER_DIR")
        src = Path(local) if local else _download_sherpa(model, tmp)
        names = [p.name for p in src.iterdir() if p.is_file()]
        encoder = _pick(names, "encoder", ".onnx")
        decoder = _pick(names, "decoder", ".onnx")
        tokens = _pick(names, "tokens", ".txt")
        enc_sha = _sha256(src / encoder)

        if do_upload and not force and _already_published(repo, model, enc_sha):
            print(f"  already published on {repo}/{model}/ (sha {enc_sha[:12]}) — "
                  f"skip (use --force)")
            return

        meta = {
            "model": model,
            "contract_version": CONTRACT_VERSION,
            "encoder": encoder,
            "decoder": decoder,
            "tokens": tokens,
            "feature_dim": feature_dim,
            "source": SHERPA_RELEASE.format(
                model=SHERPA_RELEASE_NAME.get(model, model)),
            "encoder_sha256": enc_sha,
            "decoder_sha256": _sha256(src / decoder),
            "tokens_sha256": _sha256(src / tokens),
            "versions": _versions(),
        }
        (src / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        print(f"  assets: {encoder} / {decoder} / {tokens}  feature_dim={feature_dim}")

        if do_check and not _self_check(src, encoder, decoder, tokens, feature_dim,
                                        model):
            raise SystemExit(f"self-check FAILED for {model} — refusing to upload")

        if not do_upload:
            print("  --no-upload: kept local only")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        # Public repo (free CDN, pulled token-free at runtime — same as the wav2vec2
        # align mirror); idempotent so re-runs / multi-model batches are safe.
        api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(
            repo_id=repo, repo_type="model", folder_path=str(src),
            path_in_repo=model,
            allow_patterns=[encoder, decoder, tokens, "meta.json"],
            commit_message=f"mirror sherpa whisper {model} (sha {enc_sha[:12]})",
        )
        print(f"  uploaded -> {repo}/{model}/")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", default=["tiny"],
                    help="Whisper model name(s) to mirror (default: tiny)")
    ap.add_argument("--repo", default=REPO_DEFAULT, help="HF model repo to publish to")
    ap.add_argument("--no-upload", action="store_true",
                    help="download + meta + self-check only, do not push")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the WhisperSherpa self-check before upload")
    ap.add_argument("--force", action="store_true",
                    help="upload even if the remote sha already matches")
    args = ap.parse_args()

    print(f"repo={args.repo}  upload={not args.no_upload}  "
          f"check={not args.no_check}  models={args.model}")
    for model in args.model:
        mirror_one(model, repo=args.repo, do_upload=not args.no_upload,
                   do_check=not args.no_check, force=args.force)
    print("\ndone.")


if __name__ == "__main__":
    main()
