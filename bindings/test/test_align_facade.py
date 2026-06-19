"""align() facade parity: the C++ aligner path (`WHISPERX_CORE_STAGES=align`)
produces the same aligned result as the pure-Python oracle on the same input.

torch-gated (the align() forward uses a mock torch model), so this runs in the
full project suite / the stage-token matrix, and skips in the dep-free CI lane.
The numerical golden parity vs the committed emissions lives in
test_align_parity.py; this proves the *facade swap* is behaviour-preserving,
including the native-splitter vs punkt path on multi-sentence text.
"""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("whisperx_core")

from whisperx.alignment import align  # noqa: E402

DICTIONARY = {
    "<pad>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8,
    "i": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15, "r": 16,
    "s": 17, "t": 18, "u": 19, "w": 20, "|": 21,
}
METADATA = {"language": "en", "dictionary": DICTIONARY, "type": "torchaudio"}


def _make_emission(num_frames, transcript_chars, blank_id=0):
    vocab = max(DICTIONARY.values()) + 1
    emission = torch.full((num_frames, vocab), -5.0)
    emission[:, blank_id] = -1.0
    in_dict = [(i, c) for i, c in enumerate(transcript_chars) if c.lower() in DICTIONARY]
    if in_dict:
        per = num_frames // (len(transcript_chars) + 1)
        for (ci, ch) in in_dict:
            center = (ci + 1) * per
            for t in range(max(0, center - per // 2), min(num_frames, center + per // 2)):
                emission[t, DICTIONARY[ch.lower()]] = 2.0
                emission[t, blank_id] = -3.0
    return emission


def _run(text, enabled, num_frames=160, duration=5.0):
    model = type("M", (), {})()
    emission = _make_emission(num_frames, list(text))
    model.__call__ = lambda *a, **k: (emission.unsqueeze(0), None)
    # MagicMock-free callable returning the fixed emission
    import unittest.mock as mock
    m = mock.MagicMock()
    m.return_value = (emission.unsqueeze(0), None)
    audio = torch.randn(int(duration * 16000))
    prev = os.environ.get("WHISPERX_CORE_STAGES")
    os.environ["WHISPERX_CORE_STAGES"] = "align" if enabled else ""
    try:
        return align([{"text": text, "start": 0.0, "end": duration}], m, METADATA,
                     audio, "cpu")
    finally:
        if prev is None:
            os.environ.pop("WHISPERX_CORE_STAGES", None)
        else:
            os.environ["WHISPERX_CORE_STAGES"] = prev


def _assert_equal(a, b):
    assert len(a["word_segments"]) == len(b["word_segments"])
    for wa, wb in zip(a["word_segments"], b["word_segments"]):
        assert wa["word"] == wb["word"]
        for k in ("start", "end", "score"):
            assert (k in wa) == (k in wb), f"{k} presence {wa} vs {wb}"
            if k in wa:
                # ±1 frame: the two paths share the emission but differ in float32
                # (torch) vs float64 (C++) ratio arithmetic + round boundaries.
                assert abs(wa[k] - wb[k]) <= 0.021, f"{k} {wa} vs {wb}"
    assert len(a["segments"]) == len(b["segments"])
    for sa, sb in zip(a["segments"], b["segments"]):
        assert sa["text"] == sb["text"]
        assert abs(sa["start"] - sb["start"]) <= 0.021
        assert abs(sa["end"] - sb["end"]) <= 0.021


@pytest.mark.parametrize("text", [
    "the cat sat",
    "cost 43 dollars",
    "has 43k users",
    # multi-sentence with capitalized starts → punkt and the native splitter
    # agree (lowercase-start divergence is intentional, covered in the baseline).
    "The cat sat. The dog ran.",
    "Go home. Stay here. Run fast.",
])
def test_cpp_path_matches_python_oracle(text):
    _assert_equal(_run(text, enabled=False), _run(text, enabled=True))


def test_ignore_method_parity():
    py = _run("the 99 cats", enabled=False)
    cpp = _run("the 99 cats", enabled=True)
    _assert_equal(py, cpp)
