"""Phase-2 (slice 2B) VAD smoke: the C++ ORT silero path
(``whisperx_core.silero_segments``, sherpa-onnx) runs end-to-end on a golden
clip and its raw segments feed ``merge_chunks`` without error.

This is the **decoupled** gate (briefs fact 3): torch silero ≠ ORT silero, so
the segments are NOT byte-compared to a golden — only sanity-checked (in-bounds,
ordered, non-empty) and shown to flow into the pure ``merge_chunks``. The exact
``merge_chunks`` parity lives in ``test_vad_parity.py`` against fixed input.

Skipped unless the module was built with the audio stage
(``cmake -DWHISPERX_CORE_AUDIO=ON``) and the silero ONNX is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

for cand in (ROOT / "build", ROOT / "build" / "Release", ROOT / "build" / "Debug"):
    if cand.is_dir():
        sys.path.insert(0, str(cand))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

whisperx_core = pytest.importorskip(
    "whisperx_core", reason="build the C++ module first: cmake --build build")

if not hasattr(whisperx_core, "silero_segments"):
    pytest.skip("module built without WHISPERX_CORE_AUDIO (no silero VAD)",
                allow_module_level=True)

MODEL = ROOT / "models" / "silero_vad.onnx"
if not MODEL.exists():
    pytest.skip(f"silero model missing: {MODEL}", allow_module_level=True)

CLIP = ROOT / "golden" / "clips" / "en_dialog.wav"  # 4-speaker, lots of speech
SAMPLE_RATE = 16000
VAD_ONSET = 0.5
CHUNK_SIZE = 30.0


def test_silero_segments_feed_merge_chunks():
    assert CLIP.exists(), f"missing clip {CLIP}"
    audio = whisperx_core.load_audio(str(CLIP), SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE

    segs = whisperx_core.silero_segments(
        audio, str(MODEL), SAMPLE_RATE, VAD_ONSET, CHUNK_SIZE)

    # Non-empty, ordered, in-bounds, "UNKNOWN"-labelled (matches Silero.__call__).
    assert len(segs) > 0, "silero found no speech in a speech-dense clip"
    prev_start = -1.0
    for start, end, spk in segs:
        assert spk == "UNKNOWN"
        assert 0.0 <= start < end <= duration + 0.05, (start, end, duration)
        assert start >= prev_start  # emitted in order
        prev_start = start

    # The raw segments must flow into the pure merge_chunks (slice 2A).
    merged = whisperx_core.merge_chunks(
        list(segs), CHUNK_SIZE, VAD_ONSET, VAD_ONSET)
    assert len(merged) >= 1
    for chunk in merged:
        assert chunk["end"] - chunk["start"] <= CHUNK_SIZE + 1e-6
        assert len(chunk["segments"]) >= 1
