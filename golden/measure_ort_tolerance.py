#!/usr/bin/env python3
"""Measure torch-vs-ORT wav2vec2 emission drift → the Phase-0 tolerance budget.

The committed emission goldens (`golden/intermediates/*.tensors.npz`) come from
the **torch** wav2vec2 (CTranslate2-era pipeline). The C++ core runs the same
model under **ONNX Runtime**; the two are numerically close but not identical.
This script exports the torchaudio align model to ONNX, runs it on the *exact*
golden segment, and reports the max/mean absolute drift of the log-softmax
emissions vs the committed golden — the number the Phase-0 `emission_atol` is set
just above (briefs "How to read", tolerances).

It doubles as the **front-loaded Phase-3 risk probe**: if wav2vec2 won't export
to ONNX or the drift is large, we find out now.

Run (deps are ephemeral — not added to the project):

    uv run --with onnx --with onnxruntime python golden/measure_ort_tolerance.py

Writes `golden/intermediates/tolerance_report.json` and updates the measured
`emission_atol` in `golden/intermediates/manifest.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
INTER = ROOT / "golden" / "intermediates"
SAMPLE_RATE = 16000

# The clip + segment to probe: en_libri seg0 (clean English, no wildcard column).
CLIP = "en_libri"
SEG = 0
T_START, T_END = 0.578, 3.505  # from golden/intermediates/en_libri.transcript.json


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (x - m) - np.log(e.sum(axis=-1, keepdims=True))


def main():
    import torch
    import torchaudio
    import onnxruntime as ort
    import whisperx as wx

    golden = np.load(INTER / f"{CLIP}.tensors.npz")[f"seg{SEG}_emission"]
    print(f"golden emission {golden.shape} dtype={golden.dtype}")

    # reconstruct the exact waveform segment align.py fed the model
    audio = wx.load_audio(str(ROOT / "golden" / "clips" / f"{CLIP}.wav"))
    f1, f2 = int(T_START * SAMPLE_RATE), int(T_END * SAMPLE_RATE)
    wf = torch.from_numpy(audio[f1:f2]).unsqueeze(0)  # (1, N)
    print(f"waveform segment {tuple(wf.shape)}  ({f2 - f1} samples)")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval()

    class EmissionOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            emissions, _ = self.m(x)
            return emissions

    wrapped = EmissionOnly(model)

    # The committed golden has a deterministic **wildcard column** appended by
    # align.py (model emits 29 labels; golden is 30). Reproduce it so we can both
    # (a) confirm the torch path regenerates the golden, and (b) measure pure
    # engine drift on the raw log-softmax (pre-extension), where the only
    # difference is torch-vs-ORT numerics, not the identical numpy extension.
    def _extend_wildcard(emi: np.ndarray) -> np.ndarray:
        non_blank = np.ones(emi.shape[1], dtype=bool)
        non_blank[0] = False  # blank_id == 0
        col = emi[:, non_blank].max(axis=1, keepdims=True)
        return np.concatenate([emi, col], axis=1)

    with torch.inference_mode():
        torch_raw = wrapped(wf)
        torch_emi = torch.log_softmax(torch_raw, dim=-1)[0].numpy().astype(np.float32)

    extended = (golden.shape[1] == torch_emi.shape[1] + 1)
    torch_golden = _extend_wildcard(torch_emi) if extended else torch_emi
    torch_vs_golden = float(np.abs(torch_golden - golden).max())
    print(f"model labels={torch_emi.shape[1]}  golden width={golden.shape[1]}  "
          f"wildcard-extended={extended}")
    print(f"torch vs golden  max|Δ| = {torch_vs_golden:.2e}")

    # export to ONNX + run under ORT on the same input
    onnx_path = INTER / "wav2vec2_en.onnx"
    torch.onnx.export(
        wrapped, wf, str(onnx_path),
        input_names=["waveform"], output_names=["emissions"],
        dynamic_axes={"waveform": {1: "n"}, "emissions": {1: "t"}},
        opset_version=17)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_raw = sess.run(["emissions"], {"waveform": wf.numpy()})[0]
    ort_emi = _log_softmax(ort_raw)[0].astype(np.float32)
    onnx_path.unlink()  # the export itself isn't a committed golden

    # pure engine drift on the raw model emission (29-wide log-softmax)
    ort_vs_torch = np.abs(ort_emi - torch_emi)
    # tie to the committed artifact via the same wildcard extension
    ort_golden = _extend_wildcard(ort_emi) if extended else ort_emi
    ort_vs_golden = np.abs(ort_golden - golden)
    report = {
        "clip": CLIP, "segment": SEG, "emission_shape": list(golden.shape),
        "model": "WAV2VEC2_ASR_BASE_960H", "onnx_opset": 17,
        "torch_vs_golden_max": torch_vs_golden,
        "ort_vs_golden_max": float(ort_vs_golden.max()),
        "ort_vs_golden_mean": float(ort_vs_golden.mean()),
        "ort_vs_torch_max": float(ort_vs_torch.max()),
        "ort_vs_torch_mean": float(ort_vs_torch.mean()),
        "torch": __import__("importlib.metadata", fromlist=["version"]).version("torch"),
        "onnxruntime": ort.__version__,
    }
    print(json.dumps(report, indent=2))

    # set emission_atol just above observed ORT-vs-torch drift (round up to a
    # clean value); keep >= the starting 1e-3 floor.
    drift = max(report["ort_vs_torch_max"], report["ort_vs_golden_max"])
    atol = max(1e-3, float(f"{drift * 2:.1g}"))  # 1 sig-fig headroom, 2x drift
    report["recommended_emission_atol"] = atol
    print(f"\nrecommended emission_atol = {atol:g}  (max drift {drift:.2e})")

    (INTER / "tolerance_report.json").write_text(
        json.dumps(report, indent=2) + "\n")

    man_path = INTER / "manifest.json"
    man = json.loads(man_path.read_text())
    man["tolerances"]["emission_atol"] = atol
    man["tolerances"]["trellis_atol"] = atol
    man["tolerances"]["_measured"] = {
        "ort_vs_torch_max": report["ort_vs_torch_max"],
        "source": "golden/measure_ort_tolerance.py",
        "onnxruntime": ort.__version__,
    }
    man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    print(f"updated {man_path.name}: emission_atol -> {atol:g}")


if __name__ == "__main__":
    main()
