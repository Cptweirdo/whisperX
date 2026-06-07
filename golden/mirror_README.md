---
license: mit
library_name: onnxruntime
tags:
  - wav2vec2
  - forced-alignment
  - whisperx
  - onnx
---

# wav2vec2 forced-alignment models — ONNX mirror for WhisperX

Parity-validated ONNX exports of the wav2vec2-CTC models WhisperX uses for
**word-level forced alignment**, so the C++ engine core can run them under ONNX
Runtime with **no PyTorch and no Python** at runtime.

These are produced by [`golden/export_align_onnx.py`](https://github.com/Cptweirdo/whisperX)
in the WhisperX repo — **our own export, not a re-host** — because parity to the
committed torch emission goldens (`emission_atol = 0.006`) is only guaranteed by a
pinned `opset 17` / raw-logits export. Each model is self-checked against the golden
emissions *before* upload.

## Layout

```
<model_name>/            # '/' in HF ids sanitized to '--'
  model.onnx             # masked graph (contract v2), opset 18, dynamic {batch,time}
  meta.json              # dictionary, blank_id, contract, provenance, sha256
```

## Artifact contract (v2 — masked, batchable)

- **`model.onnx`** takes **two inputs** — `waveform (B,N) float32` (raw 16 kHz mono, no
  feature normalization) + `attention_mask (B,N) int64` (1 = real sample, 0 = padding)
  — and emits **two outputs**: `emissions (B,T,V)` **raw logits** and
  `frame_lengths (B,)` (valid output frames per row, to trim padded batches). The
  consumer applies `log_softmax` and the OOV **wildcard column** (max non-blank per
  frame), matching `whisperx/alignment.py`. Exported via the PyTorch **dynamo** path.
- **Batched inference** (pad to a common length + mask) is parity-safe **only for
  `layer_norm` feature extractors** — `meta["batchable"]`. A `group_norm` extractor
  (the torchaudio base bundles) normalizes each channel **over time**, so right-padding
  shifts every valid frame's statistics and no mask recovers it; those models must be
  run **per-segment** (batch 1, all-ones mask). The HF xls-r models are batchable.
- **`meta.json`**: `{model_name, language, pipeline_type (torchaudio|huggingface),
  opset, contract_version, blank_id, n_labels, dictionary (char→id), batchable,
  inputs, outputs, source_revision, onnx_sha256, versions}` — everything needed to run
  + tokenize without a torch model.

## Published models

| Folder | Lang | Source | Loader |
|---|---|---|---|
| `WAV2VEC2_ASR_BASE_960H` | en | torchaudio bundle | torchaudio |
| `VOXPOPULI_ASR_BASE_10K_DE` | de | torchaudio bundle | torchaudio |
| `jonatasgrosman--wav2vec2-large-xlsr-53-russian` | ru | HF | huggingface |

More languages are added on demand — the exporter resolves any code in WhisperX's
`DEFAULT_ALIGN_MODELS_{TORCH,HF}` tables (43 total) with no code change:
`python golden/export_align_onnx.py --lang <code>`.

## Usage (ONNX Runtime)

```python
import onnxruntime as ort, numpy as np, json
from huggingface_hub import hf_hub_download

folder = "WAV2VEC2_ASR_BASE_960H"
meta = json.load(open(hf_hub_download("KonstantK/wav2vec2-align-onnx", f"{folder}/meta.json")))
sess = ort.InferenceSession(hf_hub_download("KonstantK/wav2vec2-align-onnx", f"{folder}/model.onnx"))

wav = waveform_16k_mono[None].astype("float32")           # (1, N)
mask = np.ones_like(wav, dtype=np.int64)                   # all real (no padding)
logits, frame_lengths = sess.run(["emissions", "frame_lengths"],
                                 {"waveform": wav, "attention_mask": mask})
logits = logits[0, :int(frame_lengths[0])]                 # trim to valid frames
# then: log_softmax over the last axis, optional wildcard extension, Viterbi forced-align.
# To batch: pad rows to a common length, set mask=0 on padding — but only when
# meta["batchable"] (layer_norm); run group_norm models one segment at a time.
```
