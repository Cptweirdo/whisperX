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
# v1: flat encoder/decoder/tokens keys — what mirror_one still publishes.
# v2 (additive; the flat v1 keys stay and keep naming the CPU-safe variant):
# + "variants" {fp32|fp16|int8: {encoder, decoder, files[], source}} and a
# filename-keyed "sha256" map. `files` lists extra assets a variant needs
# alongside (e.g. the fp32 encoder's external-data .weights — fp32 large
# models exceed the 2 GB protobuf limit). The C++ downloader
# (adapters/server/assets/downloader.cpp) fetches the variant matching
# WHISPERX_ASR_PRECISION; v1 readers keep using the flat keys.
CONTRACT_VERSION = 1
VARIANT_CONTRACT_VERSION = 2  # written by mirror_variant (additive upgrade)
# Where each variant's files originate (recorded per-variant in meta.json).
VARIANT_SOURCE = {
    "fp32": "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-turbo",
    "fp16": "converted from the fp32 variant by golden/convert_whisper_fp16.py "
            "(onnxconverter-common, keep_io_types=True)",
}
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
                feature_dim: int, model: str, provider: str = "cpu") -> bool:
    """Run _self_check_impl in a fresh subprocess. In-process, huggingface_hub/
    requests (imported earlier for the remote meta) load Python's own
    libcrypto/ssl DLLs first and the whisperx_core pyd's dependency chain then
    binds the wrong copies (0xc0000139 entry-point-not-found on Windows); a
    child process imports the pyd before anything else."""
    import subprocess
    import sys

    payload = json.dumps({
        "folder": str(folder), "encoder": encoder, "decoder": decoder,
        "tokens": tokens, "feature_dim": feature_dim, "model": model,
        "provider": provider,
    })
    code = (
        "import json,sys; from pathlib import Path; "
        "g = Path(sys.argv[1]); sys.path.insert(0, str(g)); "
        "sys.path.insert(0, str(g.parent / 'build')); "  # whisperx_core pyd
        "from mirror_whisper_onnx import _self_check_impl; "
        "sys.exit(0 if _self_check_impl(**json.loads(sys.argv[2])) else 1)"
    )
    r = subprocess.run([sys.executable, "-c", code,
                        str(Path(__file__).resolve().parent), payload])
    return r.returncode == 0


def _self_check_impl(folder: str, encoder: str, decoder: str, tokens: str,
                     feature_dim: int, model: str, provider: str = "cpu") -> bool:
    """Load the mirrored assets through the native WhisperSherpa and assert a golden
    clip transcribes to non-empty text + the right language. Skips if whisperx_core
    isn't built with WHISPERX_CORE_AUDIO or the clip is absent. provider="cuda"
    is for the big fp32/fp16 variants (turbo fp32 on CPU would crawl)."""
    folder = Path(folder)
    _dll_setup()
    try:
        import whisperx_core as wc
    except ImportError as e:
        print(f"  self-check: whisperx_core not importable ({e}) — skipped")
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
        provider=provider,
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
    # >=: a v2 (variants) meta still carries the flat v1 keys, so a matching
    # encoder sha means the legacy publish is present — don't clobber it.
    return bool(rmeta) and \
        rmeta.get("contract_version", 0) >= CONTRACT_VERSION and \
        rmeta.get("encoder_sha256") == src_sha


def _ensure_hf_token() -> None:
    """HfApi reads HF_TOKEN from the env; fall back to app/.env (never print it)."""
    import os

    if os.environ.get("HF_TOKEN"):
        return
    dotenv = ROOT / "app" / ".env"
    if not dotenv.exists():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN=") and len(line) > len("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
            print("  HF_TOKEN loaded from app/.env")
            return


def _dll_setup() -> None:
    """Windows: make the pyd's CUDA deps loadable (mirrors scripts/bench_cuda_decode.py
    — ORT's CUDA EP needs cudnn 9, bundled in the venv's torch wheel)."""
    import os

    if os.name != "nt":
        return
    build = ROOT / "build"
    torch_lib = ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    for d in (build, build / "bin", torch_lib):
        if d.is_dir():
            os.add_dll_directory(str(d))
    if torch_lib.is_dir():
        import ctypes

        for name in sorted(torch_lib.glob("cudnn*_9.dll"),
                           key=lambda p: "graph" not in p.name):
            ctypes.WinDLL(str(name))


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


def _pick_variant_file(names: list[str], needle: str, variant: str) -> str:
    """The `variant`'s asset among `names`: fp16 wants the *.fp16.* file, fp32
    the plain (non-int8, non-fp16) one."""
    cand = [n for n in names if needle in n and n.endswith(".onnx")]
    if variant == "fp16":
        cand = [n for n in cand if "fp16" in n]
    else:
        cand = [n for n in cand if "int8" not in n and "fp16" not in n]
    if not cand:
        raise FileNotFoundError(f"no {variant} '{needle}.onnx' in {names}")
    cand.sort(key=len)
    return cand[0]


def mirror_variant(model: str, variant: str, src: Path, *, repo: str,
                   do_upload: bool, do_check: bool, force: bool,
                   check_provider: str) -> None:
    """Publish a precision variant (fp32/fp16) of an already-mirrored model:
    merge a `variants` block + filename-keyed sha256 map into the remote
    meta.json (contract v2 — flat v1 keys are left untouched, so v1 readers
    and the CPU path keep resolving the original int8/fp32 files)."""
    print(f"\n=== {model} [{variant}] from {src} ===")
    feature_dim = _feature_dim(model)
    names = [p.name for p in src.iterdir() if p.is_file()]
    encoder = _pick_variant_file(names, "encoder", variant)
    decoder = _pick_variant_file(names, "decoder", variant)
    tokens = _pick(names, "tokens", ".txt")  # needed for the self-check only
    # External-data files (fp32 encoders >2 GB store weights outside the graph;
    # ORT resolves them relative to the .onnx, so they must ship alongside).
    extras = sorted(n for n in names if n.endswith(".weights"))

    rmeta = _remote_meta(repo, model)
    if not rmeta:
        raise SystemExit(f"{repo}/{model}/meta.json not found — publish the "
                         f"base model first (mirror_one)")

    enc_sha = _sha256(src / encoder)
    published = (rmeta.get("variants") or {}).get(variant)
    if do_upload and not force and published and \
            (rmeta.get("sha256") or {}).get(encoder) == enc_sha:
        print(f"  variant already published (sha {enc_sha[:12]}) — skip "
              f"(use --force)")
        return

    if do_check and not _self_check(src, encoder, decoder, tokens, feature_dim,
                                    model, provider=check_provider):
        raise SystemExit(f"self-check FAILED for {model} [{variant}] — "
                         f"refusing to upload")

    meta = dict(rmeta)
    meta["contract_version"] = VARIANT_CONTRACT_VERSION
    variants = dict(meta.get("variants") or {})
    # Record the flat v1 keys as their own variant so consumers can enumerate.
    base_kind = "int8" if "int8" in meta["encoder"] else "fp32"
    variants.setdefault(base_kind, {
        "encoder": meta["encoder"], "decoder": meta["decoder"], "files": [],
        "source": meta.get("source", ""),
    })
    variants[variant] = {
        "encoder": encoder, "decoder": decoder, "files": extras,
        "source": VARIANT_SOURCE.get(variant, ""),
    }
    meta["variants"] = variants
    sha = dict(meta.get("sha256") or {})
    for f in (encoder, decoder, *extras):
        sha[f] = enc_sha if f == encoder else _sha256(src / f)
    # Seed the v1 files into the map too (one place to look things up).
    for key, legacy in (("encoder", "encoder_sha256"),
                        ("decoder", "decoder_sha256"),
                        ("tokens", "tokens_sha256")):
        if meta.get(legacy):
            sha.setdefault(meta[key], meta[legacy])
    meta["sha256"] = sha
    meta["versions"] = _versions()
    (src / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    up = [encoder, decoder, *extras]
    print(f"  variant assets: {' / '.join(up)}  (+meta.json)")

    if not do_upload:
        print("  --no-upload: kept local only")
        return

    from huggingface_hub import HfApi

    _ensure_hf_token()
    api = HfApi()
    api.upload_folder(
        repo_id=repo, repo_type="model", folder_path=str(src),
        path_in_repo=model, allow_patterns=up + ["meta.json"],
        commit_message=f"add {variant} variant for {model} (sha {enc_sha[:12]})",
    )
    print(f"  uploaded -> {repo}/{model}/ [{variant}]")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", default=["tiny"],
                    help="Whisper model name(s) to mirror (default: tiny)")
    ap.add_argument("--repo", default=REPO_DEFAULT, help="HF model repo to publish to")
    ap.add_argument("--variant", choices=("fp32", "fp16"),
                    help="publish a precision variant of one --model from --src "
                         "(meta.json contract v2) instead of the base mirror")
    ap.add_argument("--src", type=Path,
                    help="local dir holding the variant's encoder/decoder "
                         "(+ .weights, + tokens for the self-check)")
    ap.add_argument("--check-provider", default="cpu", choices=("cpu", "cuda"),
                    help="EP for the self-check (use cuda for the big variants)")
    ap.add_argument("--no-upload", action="store_true",
                    help="download + meta + self-check only, do not push")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the WhisperSherpa self-check before upload")
    ap.add_argument("--force", action="store_true",
                    help="upload even if the remote sha already matches")
    args = ap.parse_args()

    if not args.no_upload:
        _ensure_hf_token()
    if args.variant:
        if len(args.model) != 1 or not args.src:
            ap.error("--variant needs exactly one --model and --src")
        mirror_variant(args.model[0], args.variant, args.src, repo=args.repo,
                       do_upload=not args.no_upload, do_check=not args.no_check,
                       force=args.force, check_provider=args.check_provider)
        print("\ndone.")
        return

    print(f"repo={args.repo}  upload={not args.no_upload}  "
          f"check={not args.no_check}  models={args.model}")
    for model in args.model:
        mirror_one(model, repo=args.repo, do_upload=not args.no_upload,
                   do_check=not args.no_check, force=args.force)
    print("\ndone.")


if __name__ == "__main__":
    main()
