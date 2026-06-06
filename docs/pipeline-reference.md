# WhisperX Transcription Pipeline — Reference

> What each stage of the pipeline does, the exact code responsible for it, and
> the dependencies it pulls in. This is the **spec** the port work targets (see
> [`android-port-options.md`](./android-port-options.md),
> [`ios-port-options.md`](./ios-port-options.md),
> [`windows-port-options.md`](./windows-port-options.md),
> [`packaging-shared-vs-bespoke.md`](./packaging-shared-vs-bespoke.md)). Line refs
> are against the tree at the time of writing.

## Data flow at a glance

```
audio file
  │  load_audio()  ── ffmpeg subprocess → PCM 16 kHz mono float32
  ▼
[Stage 1] Audio          whisperx/audio.py
  │  waveform (np.float32, 16 kHz)
  ▼
[Stage 2] VAD            whisperx/vads/     → voiced regions, merged into ≤chunk_size windows
  │  vad_segments: [{start, end, segments:[…]}]
  ▼
[Stage 3] Transcribe     whisperx/asr.py    → batched Whisper decode per VAD window
  │  TranscriptionResult {segments:[{start,end,text,avg_logprob}], language}
  ▼
[Stage 4] Align          whisperx/alignment.py → wav2vec2 CTC + Viterbi → word/char timestamps
  │  AlignedTranscriptionResult {segments:[…words…], word_segments:[…]}
  ▼
[Stage 5] Diarize        whisperx/diarize.py → pyannote turns + speaker assignment
  │  result segments/words gain "speaker"
  ▼
[Stage 6] Write          whisperx/utils.py  → srt/vtt/txt/tsv/json/aud
```

Stages are **sequential and independently loaded**: each loads its model, runs,
then is freed (`del model; gc.collect(); torch.cuda.empty_cache()`) before the
next — see the orchestration in `transcribe.py:124-238`. Audio constants
(`SAMPLE_RATE=16000`, `N_FFT=400`, `HOP_LENGTH=160`, `N_SAMPLES=480000`,
`N_FRAMES=3000`, `FRAMES_PER_SECOND=100`, `TOKENS_PER_SECOND=50`) are defined once
in `audio.py:13-22` and reused across stages.

---

## Stage 0 — Orchestration & public API

**What it does.** Drives the five stages from CLI args; exposes a lazy-import
public API so `import whisperx` stays light.

| Concern | Code | Notes |
|---|---|---|
| CLI entry / every flag | `whisperx/__main__.py::cli()` (`__main__.py:12`) | `whisperx` console script (`pyproject.toml:46`) |
| Pipeline driver | `whisperx/transcribe.py::transcribe_task()` (`transcribe.py:20`) | Part 1 VAD+ASR `:124`, Part 2 Align `:165`, Diarize `:208`, Write `:235` |
| Lazy public API | `whisperx/__init__.py` | `load_model`, `load_align_model`, `align`, `load_audio`, `assign_word_speakers` via `_lazy_import` (`__init__.py:4`) — heavy deps load only when called |
| Result shapes | `whisperx/schema.py` | TypedDicts (below) — the contract between stages |
| Logging | `whisperx/log_utils.py` | `get_logger(__name__)` |

**Deps:** stdlib `argparse`, `torch` (only for `torch.cuda.is_available()` device
default + `set_num_threads`), `numpy`.

---

## Stage 1 — Audio loading & features (`whisperx/audio.py`)

**What it does.** Decodes any audio file to a mono 16 kHz float32 waveform, and
computes the log-Mel spectrogram Whisper's encoder expects.

| Function | Line | Role |
|---|---|---|
| `load_audio(file, sr=16000)` | `audio.py:25` | **Shells out to the `ffmpeg` CLI** (`subprocess.run`, `:44-61`) → `s16le` PCM → `np.float32 / 32768`. **No pure-Python fallback.** |
| `log_mel_spectrogram(audio, n_mels, …)` | `audio.py:112` | `torch.hann_window` + **`torch.stft`** (`:150`) → magnitudes → Mel filterbank matmul → log10/clamp/scale. n_mels ∈ {80,128}. |
| `mel_filters(device, n_mels)` | `audio.py:94` | Loads precomputed filterbank from `assets/mel_filters.npz` (`@lru_cache`) — decouples the librosa dependency. |
| `pad_or_trim(array, length)` | `audio.py:68` | Pad/trim to `N_SAMPLES` (30 s) for the encoder. |

**Deps:** `ffmpeg` (external CLI, **native**), `torch`/`torch.nn.functional`
(STFT — **native**), `numpy`. *Portability note:* the ffmpeg subprocess and
`torch.stft` are the two pieces every non-desktop port must replace (AVFoundation /
AudioRecord + a native/Dart FFT).

---

## Stage 2 — Voice Activity Detection & chunk merging (`whisperx/vads/`)

**What it does.** Finds voiced regions, then **merges them into windows of
≤`chunk_size` (default 30 s)** so Whisper runs on batched, speech-only audio
(WhisperX's batched-inference trick). `--vad_method` selects the backend.

| Component | Code | Role |
|---|---|---|
| Common interface | `vads/vad.py::Vad` (`vad.py:7`) | Base class; **`Vad.merge_chunks` (`vad.py:19-53`) is the pure-Python window-merge** used by all backends. |
| Pyannote (default) | `vads/pyannote.py::Pyannote` (`pyannote.py:233`) | `load_vad_model` (`:21`) loads `assets/pytorch_model.bin` via `pyannote.audio.Model`; `Binarize` (`:51`) does hysteresis thresholding + a **min-cut** so no segment exceeds `max_duration` (`:140-149`). |
| Silero | `vads/silero.py::Silero` (`silero.py:18`) | Loads JIT model via **`torch.hub.load('snakers4/silero-vad')`** (`:26`); `get_speech_timestamps` → segments. |
| Selection | `asr.py::load_model` (`asr.py:465-478`) | `vad_method` → `Silero(...)` or `Pyannote(...)`; a manually passed `vad_model` wins. Defaults: `vad_onset=0.500`, `vad_offset=0.363`, `chunk_size=30` (`asr.py:455-459`). |

**Deps:** `torch` (both); `pyannote.audio` + `pyannote.core` (pyannote path —
**native/gated model**); `torch.hub` download (silero). `merge_chunks` itself is
**pure Python/numpy**.

---

## Stage 3 — Transcription / ASR (`whisperx/asr.py` + backends)

**What it does.** Runs Whisper over the VAD windows in batches (timestamp-free,
fixed prompt) and returns text segments + language.

| Component | Code | Role |
|---|---|---|
| `WhisperModel` | `asr.py:31` | Subclass of `faster_whisper.WhisperModel`. `generate_segment_batched` (`:37`) — batched encode+decode via **CTranslate2** (`self.model.encode` `:95`, `self.model.generate` `:62`). |
| `FasterWhisperPipeline` | `asr.py:106` | Subclass of HF `transformers.Pipeline`. `preprocess`=`log_mel_spectrogram` (`:159`), `_forward`=`generate_segment_batched` (`:169`), `transcribe` (`:197`) = run VAD → `merge_chunks` → batched decode → assemble `SingleSegment`s. `detect_language` (`:300`). |
| `load_model(...)` | `asr.py:325` | Builds the pipeline; holds `default_asr_options` (`:417-445`, beam_size 5, temperatures, thresholds, `without_timestamps=True`) and `default_vad_options`. Picks compute type (`float16` GPU / `float32` CPU, `:396`). |
| MLX backend | `asr_mlx.py::load_mlx_model` (via `asr.py:360`) | Apple-Silicon GPU ASR (`device="mlx"`); same `transcribe()` contract so downstream stages are unchanged. |
| whisper.cpp backend | `asr_whispercpp.py::load_whispercpp_model` (via `asr.py:377`) | ggml/Metal/CPU ASR (`device="whispercpp"`); same contract. |

**Output:** `TranscriptionResult = {segments:[{start,end,text,avg_logprob}], language}`.

**Deps:** `ctranslate2` (**native C++ inference engine**), `faster_whisper`,
`torch` (device handling, the DataLoader/mel), `transformers` (Pipeline base),
`numpy`. Optional backends: `mlx-whisper` (Darwin arm64), `pywhispercpp`
(cross-platform).

---

## Stage 4 — Forced alignment (`whisperx/alignment.py`)

**What it does.** This is WhisperX's headline feature: force-aligns the
transcript text to audio with a per-language **wav2vec2 CTC** model to produce
**word-level (and optional char-level) timestamps**. Skipped for
`--task translate` (`transcribe.py:47`).

| Component | Code | Role |
|---|---|---|
| `load_align_model(lang, …)` | `alignment.py:80` | Resolves a model from `DEFAULT_ALIGN_MODELS_TORCH` (5 langs, torchaudio bundles, `:32`) or `DEFAULT_ALIGN_MODELS_HF` (38 langs, HF `Wav2Vec2ForCTC`/`Wav2Vec2Processor`, `:40`). Returns model + `{dictionary, language, type}`. |
| `align(...)` | `alignment.py:117` | The orchestrator. **Step 1** (`:146-203`): clean text to dict chars, OOV→wildcard, sentence spans via **NLTK punkt** (`:189-196`). **Step 2** (`:207-411`): per segment, slice waveform, run the model → `torch.log_softmax` emissions (`:256-263`), build tokens (+wildcard column for OOV, `:272-283`), then DP align. |
| `get_trellis(emission, tokens)` | `alignment.py:425` | **Viterbi trellis** (pure `torch` DP over frames × tokens). |
| `backtrack(trellis, …)` | `alignment.py:455` | Backtrace the optimal path (pure Python + `.exp()`). Returns `None` on failure → segment kept unaligned. |
| `merge_repeats` / `merge_words` | `alignment.py:508` / `:526` | Collapse repeated CTC labels into char then word `Segment`s. |
| char→word→sentence assembly | `alignment.py:298-411` | **pandas** DataFrame ops + `interpolate_nans` (`utils.py:470`) to fill missing word timings; group by sentence/timestamps. |

**Output:** `AlignedTranscriptionResult = {segments:[…+words[],chars?], word_segments:[]}`.

**Deps:** `torch` + `torchaudio` (bundle models — **native**), `transformers`
(`Wav2Vec2ForCTC`/`Processor` — **native** + HF download), `nltk` (punkt, auto-
downloads), `pandas`, `numpy`. *Portability note:* only the **model forward pass**
needs an engine; `get_trellis`/`backtrack`/`merge_*`/interpolation are
**pure algorithm** (the part the Dart/native ports reimplement; the trellis is
the same as `torchaudio.functional.forced_align`).

---

## Stage 5 — Speaker diarization (`whisperx/diarize.py`)

**What it does.** Labels *who spoke when* with pyannote, then maps those speaker
turns onto the aligned segments/words.

| Component | Code | Role |
|---|---|---|
| `DiarizationPipeline` | `diarize.py:91` | Wraps **`pyannote.audio.Pipeline.from_pretrained`** (`:103`); default model `pyannote/speaker-diarization-community-1` (`:101`) — **gated, needs HF token**. `__call__` (`:105`) runs it (segmentation→embeddings, with optional progress hook `:138-162`) → DataFrame `[segment,label,speaker,start,end]` (`:170-172`); optional speaker embeddings. |
| `IntervalTree` | `diarize.py:14` | **Pure-Python/numpy** sorted-array + binary-search structure for O(log n) overlap queries (`query` `:41`, `find_nearest` `:72`). |
| `assign_word_speakers(...)` | `diarize.py:185` | Builds the tree from diarization turns, assigns each segment/word the speaker with **max intersection duration** (`:222-257`); optional `fill_nearest`; attaches embeddings. **Pure Python.** |

**Output:** the transcript result with `speaker` added to segments/words (+
optional `speaker_embeddings`).

**Deps:** `pyannote.audio` + `pyannote.core` (**native, gated model + HF
download**), `torch`, `pandas`, `numpy`. *Portability note:* the model is the hard
part (sherpa-onnx ships an ONNX equivalent for mobile); `IntervalTree` +
`assign_word_speakers` are **pure algorithm**.

---

## Stage 6 — Output writing (`whisperx/utils.py` + helpers)

**What it does.** Serializes the result to subtitle/transcript files.

| Component | Code | Role |
|---|---|---|
| `get_writer(format, dir)` | `utils.py:443` | Maps `txt/vtt/srt/tsv/json` → writer classes (`:446`), `aud` (Audacity) optional (`:453`), or `all`. |
| `ResultWriter` + subclasses | `utils.py:215-441` | `WriteTXT`, `WriteVTT`, `WriteSRT`, `WriteTSV`, `WriteAudacity`, `WriteJSON`; `SubtitlesWriter` (`:248`) handles `highlight_words`/`max_line_*`. |
| `SubtitlesProcessor` | `whisperx/SubtitlesProcessor.py` | Subtitle line splitting/formatting. |
| `conjunctions` | `whisperx/conjunctions.py` | Per-language conjunction lists for sentence/segment splitting. |
| `format_timestamp`, `interpolate_nans`, `LANGUAGES`/`TO_LANGUAGE_CODE` | `utils.py:194` / `:470` / `:8` | Shared helpers. |

**Deps:** **pure Python stdlib** (`json`, `re`, `zlib`, file I/O). Fully portable.

---

## Result schema (`whisperx/schema.py`)

The contract every stage reads/writes — the cross-language data shapes the ports
must match:

- `SingleSegment` = `{start, end, text, avg_logprob?}` — Stage 3 output unit.
- `SingleWordSegment` = `{word, start, end, score}`; `SingleCharSegment` = `{char, start, end, score}` — Stage 4 units.
- `SingleAlignedSegment` = `SingleSegment` + `{words:[SingleWordSegment], chars?}`.
- `TranscriptionResult` = `{segments:[SingleSegment], language}` — Stage 3 → 4.
- `AlignedTranscriptionResult` = `{segments:[SingleAlignedSegment], word_segments:[SingleWordSegment]}` — Stage 4 → 5.
- `SegmentData` — internal alignment scratch (clean chars/indices, sentence spans).

---

## Dependency & portability summary

| Stage | Native / heavy deps | Pure-Python (portable) parts | Replace off-desktop with |
|---|---|---|---|
| 1 Audio | **ffmpeg CLI**, `torch.stft` | constants, `pad_or_trim` | AVFoundation/AudioRecord + native/Dart FFT |
| 2 VAD | `pyannote.audio` / `torch.hub` silero | **`Vad.merge_chunks`** | silero ONNX (sherpa) + port merge logic |
| 3 ASR | **`ctranslate2`**, `faster_whisper`, `transformers` | segment assembly | LiteRT / sherpa-onnx Whisper |
| 4 Align | `torch`, `torchaudio`, `transformers`, `nltk`, `pandas` | **`get_trellis`, `backtrack`, `merge_*`, interpolation** | wav2vec2→ONNX/CoreML/LiteRT + port the DP |
| 5 Diarize | **`pyannote.audio`** (gated) | **`IntervalTree`, `assign_word_speakers`** | sherpa-onnx pyannote-seg + CAM++ + port assignment |
| 6 Write | — | **all of it** | trivial reimplement |

**Bottom line:** the model forward passes (Stages 1 STFT, 3, 4, 5) are the
native/engine-bound parts; the **glue** — VAD chunk merge, the alignment
Viterbi, the diarization interval-tree assignment, and all output formatting — is
pure algorithm and is exactly what the cross-platform "headless core" reuses
(shared as code on desktop, ported/validated against golden vectors on mobile).
