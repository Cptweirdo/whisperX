"""Phase-2 (slice 2A) parity oracle: the C++ ``whisperx_core.merge_chunks`` vs the
pure-Python ``whisperx.vads.vad.Vad._py_merge_chunks`` it replaces (the ``vad``
stage token).

Two things are proven here:

1. **Per-function parity** — the C++ port and the Python oracle pack identical
   chunk boundaries + sub-segment groupings on adversarial segment lists
   (back-to-back, zero-width, spans that force several flushes, the degenerate
   first-segment-over-chunk_size case the ``curr_end-curr_start>0`` guard covers).
2. **Golden replay (the decoupled gate)** — the committed raw VAD segments in
   ``golden/intermediates/*.vad.json`` are fed through the C++ port and must
   reproduce the committed ``merged_chunks`` EXACTLY (start/end + n_segments).
   silero's own timestamps aren't byte-comparable across torch/ORT, but the merge
   is pure, so this fixed input makes any merge drift a real bug.

Skipped unless the C++ module has been built (``cmake --build build``); run with
the build dir on the path (``PYTHONPATH=build``).
"""

from __future__ import annotations

import json
import sys
from collections import namedtuple
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

from whisperx.vads.vad import Vad  # noqa: E402

Seg = namedtuple("Seg", "start end speaker")
INTERMEDIATES = ROOT / "golden" / "intermediates"


def _py(segments, chunk_size, onset, offset):
    """Python oracle, normalized so inner (start,end) tuples compare as lists."""
    out = Vad._py_merge_chunks(
        [Seg(*s) for s in segments], chunk_size, onset, offset)
    return [{"start": m["start"], "end": m["end"],
             "segments": [list(p) for p in m["segments"]]} for m in out]


def _cpp(segments, chunk_size, onset, offset):
    return whisperx_core.merge_chunks(
        [tuple(s) for s in segments], chunk_size, onset, offset)


# (start, end, speaker) segment lists exercising the flush logic.
CASES = [
    [(0.0, 5.0, "UNKNOWN")],                                   # single
    [(0.0, 10.0, "UNKNOWN"), (11.0, 20.0, "UNKNOWN"),
     (21.0, 25.0, "UNKNOWN")],                                 # all under chunk_size
    [(0.0, 10.0, "UNKNOWN"), (15.0, 20.0, "UNKNOWN"),
     (30.0, 35.0, "UNKNOWN")],                                 # one flush
    [(0.0, 40.0, "UNKNOWN")],                                  # degenerate first-flush guard
    [(0.0, 5.0, "UNKNOWN"), (5.0, 5.0, "UNKNOWN"),
     (5.0, 12.0, "UNKNOWN")],                                  # back-to-back + zero-width
    [(i * 4.0, i * 4.0 + 3.0, "UNKNOWN") for i in range(30)],  # many flushes
    [(0.0, 5.0, "A"), (6.0, 9.0, "B"), (40.0, 44.0, "C")],     # mixed speakers
]


@pytest.mark.parametrize("segments", CASES)
def test_merge_chunks_matches_python(segments):
    chunk_size, onset, offset = 30.0, 0.5, 0.363
    assert _cpp(segments, chunk_size, onset, offset) == \
        _py(segments, chunk_size, onset, offset)


def _golden_vads():
    return sorted(INTERMEDIATES.glob("*.vad.json"))


@pytest.mark.parametrize("vad_path", _golden_vads(),
                         ids=lambda p: p.name.replace(".vad.json", ""))
def test_golden_segments_replay_to_merged_chunks(vad_path):
    """C++ merge of the fixed raw segments == committed merged_chunks (exact)."""
    g = json.loads(vad_path.read_text())
    if "segments" not in g:
        pytest.skip(f"{vad_path.name} has no raw segments — rerun "
                    "`dump_goldens.py --vad-only`")
    p = g["params"]
    segs = [(s["start"], s["end"], s.get("speaker")) for s in g["segments"]]
    merged = _cpp(segs, p["chunk_size"], p["onset"], p["offset"])
    got = [{"start": round(m["start"], 4), "end": round(m["end"], 4),
            "n_segments": len(m["segments"])} for m in merged]
    assert got == g["merged_chunks"]
