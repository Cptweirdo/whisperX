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
  model.onnx             # raw logits (B,N waveform) -> (B,T,V); opset 17, dynamic axes
  meta.json              # dictionary, blank_id, opset, provenance, sha256
```

## Artifact contract

- **`model.onnx`** emits **raw logits** — the consumer applies `log_softmax` and the
  OOV **wildcard column** (max non-blank per frame), matching `whisperx/alignment.py`.
  Input is the **raw 16 kHz mono float32 waveform** `(batch, samples)` (no feature
  normalization — the HF path feeds raw audio too).
- **`meta.json`**: `{model_name, language, pipeline_type (torchaudio|huggingface),
  opset, blank_id, n_labels, dictionary (char→id), source_revision, onnx_sha256,
  versions}` — everything needed to tokenize without a torch model.

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
logits = sess.run(["emissions"], {"waveform": waveform_16k_mono[None].astype("float32")})[0]
# then: log_softmax over the last axis, optional wildcard extension, Viterbi forced-align
```
