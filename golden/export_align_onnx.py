#!/usr/bin/env python3
"""Export WhisperX wav2vec2 align models to ONNX and publish the HF mirror.

**Run vs. produce.** ONNX is a portable, language-agnostic format: the C++ core
(Phase 3B) runs wav2vec2 under ONNX Runtime with *no* PyTorch and *no* Python. The
only step that needs torch is this **one-time conversion**, which is offline tooling
(dev/CI) — never the shipped runtime. The engine only ever consumes a *path to a
parity-valid ``model.onnx``*; where that file comes from is swappable (torch export
now → CI-hosted later) without touching the C++ seam.

**We are our own mirror.** Public ONNX re-hosts cover only the popular base model
and aren't pinned/validated against our committed emissions, so we publish our own
parity-checked export to a repo *we* control (default
``KonstantK/wav2vec2-align-onnx``). The runtime later pulls from it exactly like the
original weights are pulled today via ``huggingface_hub`` (free, CDN, sha-pinned).

**Extensible by design — adding a language is data, not code.** The model registry
is ``whisperx.alignment``'s own ``DEFAULT_ALIGN_MODELS_TORCH`` (5) +
``DEFAULT_ALIGN_MODELS_HF`` (38) tables, imported here. ``--lang <code>`` exports any
of those 43 with zero new logic; the only per-architecture knowledge is the tiny
``loader-type -> wrapper`` registry below (torchaudio vs huggingface). A brand-new
architecture = one wrapper entry.

The exported ``model.onnx`` emits **raw logits** (NOT log_softmax, NOT the wildcard
column — those are whisperX post-processing the consumer applies, matching
``alignment.py:285,295-302``); input is the **raw 16 kHz mono waveform** ``(B, N)``
(the HF path feeds raw waveform too — ``alignment.py:282`` — so one input contract).
``meta.json`` ships the char->id ``dictionary`` + ``blank_id`` so the eventual
torch-free runtime can tokenize without loading a torch model.

Run (heavy deps ephemeral — not added to the project, same as
``measure_ort_tolerance.py``)::

    uv run --no-project --with torch --with torchaudio --with transformers \
        --with onnx --with onnxruntime --with huggingface_hub \
        python golden/export_align_onnx.py --all-golden

    # any supported language, no code change:
    python golden/export_align_onnx.py --lang es it          # voxpopuli ES/IT
    python golden/export_align_onnx.py --all                 # all 43 (the tail)
    python golden/export_align_onnx.py --lang en --no-upload # export + self-check only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
INTER = ROOT / "golden" / "intermediates"
CLIPS = ROOT / "golden" / "clips"

REPO_DEFAULT = "KonstantK/wav2vec2-align-onnx"
OPSET = 17
SAMPLE_RATE = 16000
GOLDEN_LANGS = ["en", "de", "ru"]
EMISSION_ATOL = 0.006  # golden/intermediates/manifest.json

# One golden clip per language for the pre-upload self-check (proves the export
# reproduces the committed torch emissions before it leaves this machine).
CHECK_CLIP = {"en": "en_libri", "de": "de_dialog", "ru": "ru_cv_71085"}


def sanitize(model_name: str) -> str:
    """HF path-safe folder key for a model name (``org/name`` -> ``org--name``)."""
    return model_name.replace("/", "--")


def resolve_model_name(lang: str) -> str:
    """lang -> default align model name, exactly as ``load_align_model`` resolves."""
    from whisperx.alignment import (
        DEFAULT_ALIGN_MODELS_HF,
        DEFAULT_ALIGN_MODELS_TORCH,
    )
    if lang in DEFAULT_ALIGN_MODELS_TORCH:
        return DEFAULT_ALIGN_MODELS_TORCH[lang]
    if lang in DEFAULT_ALIGN_MODELS_HF:
        return DEFAULT_ALIGN_MODELS_HF[lang]
    raise ValueError(f"no default align model for language {lang!r}")


def all_langs() -> list[str]:
    from whisperx.alignment import (
        DEFAULT_ALIGN_MODELS_HF,
        DEFAULT_ALIGN_MODELS_TORCH,
    )
    return list(DEFAULT_ALIGN_MODELS_TORCH) + list(DEFAULT_ALIGN_MODELS_HF)


def _wrap_emission(model, pipeline_type: str):
    """Wrap a loaded align model so ``forward(waveform) -> raw logits (B, T, V)``.

    The whole per-architecture surface lives here: a new loader type adds one entry.
    """
    import torch

    class TorchaudioEmission(torch.nn.Module):
        # torchaudio bundle forward returns ``(emissions, lengths)`` — take [0].
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[0]

    class HFEmission(torch.nn.Module):
        # Wav2Vec2ForCTC forward returns a ModelOutput with ``.logits``.
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).logits

    registry = {"torchaudio": TorchaudioEmission, "huggingface": HFEmission}
    if pipeline_type not in registry:
        raise ValueError(
            f"no ONNX wrapper for align pipeline type {pipeline_type!r} "
            f"(known: {sorted(registry)})"
        )
    return registry[pipeline_type](model).eval()


def _blank_id(dictionary: dict) -> int:
    """Reproduce ``alignment.py:289-292``'s blank/pad id resolution."""
    blank_id = 0
    for char, code in dictionary.items():
        if char in ("[pad]", "<pad>"):
            blank_id = code
    return blank_id


def _log_softmax(x):
    import numpy as np

    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (x - m) - np.log(e.sum(axis=-1, keepdims=True))


def _extend_wildcard(emi, blank_id: int):
    """Append the max-non-blank column align.py adds for OOV chars (on log-softmax)."""
    import numpy as np

    non_blank = np.ones(emi.shape[1], dtype=bool)
    non_blank[blank_id] = False
    col = emi[:, non_blank].max(axis=1, keepdims=True)
    return np.concatenate([emi, col], axis=1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_revision(model_name: str, pipeline_type: str):
    """Best-effort HF commit hash for provenance (torchaudio bundles have none)."""
    if pipeline_type != "huggingface":
        return None
    try:
        from huggingface_hub import model_info

        return model_info(model_name).sha
    except Exception:
        return None


def _versions() -> dict:
    import importlib.metadata as md

    out = {}
    for pkg in ("torch", "torchaudio", "transformers", "onnx", "onnxruntime"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def _self_check(onnx_path: Path, lang: str, blank_id: int) -> bool:
    """Run the fresh ONNX over a golden clip's segments; compare to committed
    emissions within ``emission_atol``. Skips cleanly if the clip/golden is absent."""
    import numpy as np
    import onnxruntime as ort

    clip = CHECK_CLIP.get(lang)
    wav = CLIPS / f"{clip}.wav" if clip else None
    npz = INTER / f"{clip}.tensors.npz" if clip else None
    tj = INTER / f"{clip}.transcript.json" if clip else None
    if not (clip and wav.exists() and npz.exists() and tj.exists()):
        print(f"  self-check: no golden for {lang!r} — skipped")
        return True

    from whisperx.audio import load_audio  # ffmpeg subprocess, no torch

    audio = load_audio(str(wav))
    segs = json.loads(tj.read_text())["segments"]
    golden = np.load(npz)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    ok = True
    for i, seg in enumerate(segs):
        f1, f2 = int(seg["start"] * SAMPLE_RATE), int(seg["end"] * SAMPLE_RATE)
        wf = audio[f1:f2][None, :].astype(np.float32)  # (1, N)
        logits = sess.run(["emissions"], {"waveform": wf})[0]
        emi = _log_softmax(logits)[0].astype(np.float32)
        g = golden[f"seg{i}_emission"]
        if g.shape[1] == emi.shape[1] + 1:
            emi = _extend_wildcard(emi, blank_id)
        d = float(np.abs(emi - g).max()) if emi.shape == g.shape else float("inf")
        status = "ok" if d < EMISSION_ATOL else "FAIL"
        ok = ok and d < EMISSION_ATOL
        print(f"  self-check {clip} seg{i}: max|Δ|={d:.2e}  ({status})")
    return ok


def export_one(lang: str, model_name: str | None, *, repo: str, do_upload: bool,
               do_check: bool, force: bool) -> None:
    import torch

    from whisperx.alignment import load_align_model

    resolved = model_name or resolve_model_name(lang)
    print(f"\n=== {lang}  ->  {resolved} ===")

    # Skip early (before the expensive load+export) when already on the mirror at
    # this opset + upstream revision, unless we're forcing or only exporting locally.
    source_revision = _hf_revision(resolved, "huggingface")  # None for torchaudio
    if do_upload and not force and _already_published(repo, resolved,
                                                      source_revision):
        print(f"  already published on {repo}/{sanitize(resolved)}/ "
              f"(opset{OPSET}, rev={source_revision}) — skip (use --force)")
        return

    model, meta = load_align_model(lang, "cpu", model_name=model_name)
    pipeline_type = meta["type"]
    dictionary = meta["dictionary"]
    blank_id = _blank_id(dictionary)
    wrapped = _wrap_emission(model, pipeline_type)

    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        onnx_path = folder / "model.onnx"
        example = torch.zeros(1, SAMPLE_RATE, dtype=torch.float32)  # 1 s dummy
        with torch.inference_mode():
            torch.onnx.export(
                wrapped, example, str(onnx_path),
                input_names=["waveform"], output_names=["emissions"],
                dynamic_axes={"waveform": {0: "batch", 1: "samples"},
                              "emissions": {0: "batch", 1: "time"}},
                opset_version=OPSET,
            )
        sha = _sha256(onnx_path)
        size_mb = onnx_path.stat().st_size / 1e6
        print(f"  exported {size_mb:.0f} MB  sha256={sha[:12]}…  "
              f"labels={len(dictionary)}  blank_id={blank_id}  type={pipeline_type}")

        meta_json = {
            "model_name": resolved,
            "language": lang,
            "pipeline_type": pipeline_type,
            "opset": OPSET,
            "blank_id": blank_id,
            "n_labels": len(dictionary),
            "dictionary": dictionary,
            "source_revision": source_revision,
            "emits": "raw_logits",  # consumer applies log_softmax + wildcard
            "input": "waveform_16k_mono_f32",
            "onnx_sha256": sha,
            "versions": _versions(),
        }
        (folder / "meta.json").write_text(
            json.dumps(meta_json, ensure_ascii=False, indent=2) + "\n")

        if do_check and not _self_check(onnx_path, lang, blank_id):
            raise SystemExit(
                f"self-check FAILED for {resolved} — refusing to upload "
                f"(emissions drift > {EMISSION_ATOL})")

        if not do_upload:
            print("  --no-upload: kept local only")
            return

        path_in_repo = sanitize(resolved)
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id=repo, repo_type="model", folder_path=str(folder),
            path_in_repo=path_in_repo,
            commit_message=f"export {resolved} ({lang}) opset{OPSET} sha {sha[:12]}",
        )
        print(f"  uploaded -> {repo}/{path_in_repo}/")


def _remote_meta(repo: str, path_in_repo: str):
    """The remote ``meta.json`` for a model folder, or None if not yet published."""
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(repo, f"{path_in_repo}/meta.json", repo_type="model")
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _already_published(repo: str, resolved: str, source_revision) -> bool:
    """True when this model is already on the mirror at the same opset + upstream
    revision — so a benign re-run skips it. Keyed on **stable identity**, not the
    freshly-exported onnx bytes (``torch.onnx.export`` isn't guaranteed
    byte-reproducible), so it never needlessly re-uploads. ``--force`` overrides."""
    rmeta = _remote_meta(repo, sanitize(resolved))
    return bool(rmeta) and rmeta.get("opset") == OPSET and \
        rmeta.get("source_revision") == source_revision


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", nargs="+", metavar="CODE",
                    help="language code(s) from the alignment.py tables")
    ap.add_argument("--model", metavar="NAME",
                    help="explicit align model (torchaudio bundle or HF id); "
                         "pair with a single --lang to tag the language")
    ap.add_argument("--all-golden", action="store_true",
                    help=f"export the golden set: {GOLDEN_LANGS}")
    ap.add_argument("--all", action="store_true",
                    help="export every language in both tables (the 43-model tail)")
    ap.add_argument("--repo", default=REPO_DEFAULT, help="HF model repo to publish to")
    ap.add_argument("--no-upload", action="store_true",
                    help="export + self-check only, do not push")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the golden self-check before upload")
    ap.add_argument("--force", action="store_true",
                    help="upload even if the remote sha already matches")
    args = ap.parse_args()

    # build the (lang, model_name) target list
    if args.model:
        if not args.lang or len(args.lang) != 1:
            ap.error("--model requires exactly one --lang to tag the language")
        targets = [(args.lang[0], args.model)]
    elif args.all:
        targets = [(l, None) for l in all_langs()]
    elif args.all_golden:
        targets = [(l, None) for l in GOLDEN_LANGS]
    elif args.lang:
        targets = [(l, None) for l in args.lang]
    else:
        ap.error("nothing to do: pass --all-golden, --all, --lang, or --model")

    print(f"repo={args.repo}  upload={not args.no_upload}  "
          f"check={not args.no_check}  targets={[t[0] for t in targets]}")
    for lang, model_name in targets:
        export_one(lang, model_name, repo=args.repo, do_upload=not args.no_upload,
                   do_check=not args.no_check, force=args.force)
    print("\ndone.")


if __name__ == "__main__":
    main()
