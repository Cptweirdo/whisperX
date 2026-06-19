"""Parity for the native speaker-assignment glue (`assign` token, Phase 4).

Replays each dialog clip's committed ``*.assign.json`` through the C++
``whisperx_core.assign_word_speakers`` and asserts the speaker labels it sets on
segments **and** words match the Python oracle's ``segments_out`` exactly.

The golden is **model-independent**: the input turns are the synthetic dialogs'
ground-truth RTTM (CC0, ``golden/clips/*.rttm``), fed once through the Python
``assign_word_speakers`` over the pre-diarize aligned segments by
``golden/dump_goldens.py`` — so this runs torch/pandas/model-free (the C++ takes
turns as plain tuples + segments as plain dicts). Build the module with
``-DWHISPERX_CORE_PYMODULE=ON`` and put it on ``PYTHONPATH`` (no WHISPERX_CORE_AUDIO
needed); skips cleanly if it isn't importable or the golden hasn't been generated.
"""
import json
from pathlib import Path

import pytest

wc = pytest.importorskip("whisperx_core")

INTER = Path(__file__).resolve().parents[2] / "golden" / "intermediates"
GOLDENS = sorted(INTER.glob("*_dialog.assign.json"))

if not GOLDENS:
    pytest.skip("no *_dialog.assign.json goldens — run golden/dump_goldens.py",
                allow_module_level=True)


def _speakers(segments):
    """(segment speaker, [word speakers]) per segment — the assigned labels."""
    return [
        (seg.get("speaker"), [w.get("speaker") for w in seg.get("words", [])])
        for seg in segments
    ]


@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.stem)
def test_assign_matches_oracle(path):
    g = json.loads(path.read_text())
    turns = [tuple(t) for t in g["turns"]]  # (start, end, speaker)

    out = wc.assign_word_speakers(turns, g["segments_in"], False)
    expected = g["segments_out"]

    assert len(out) == len(expected)
    # Speaker labels (the parity gate) match exactly on segments and words.
    assert _speakers(out) == _speakers(expected)
    # The C++ preserves every word (only sets 'speaker'); structure is intact.
    for o, e in zip(out, expected):
        assert len(o.get("words", [])) == len(e.get("words", []))
