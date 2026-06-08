"""Cross-language parity for the native output writers (`writers` token, Phase 5).

The C++ `write_<fmt>` must reproduce the Python `ResultWriter.write_result` bytes
**exactly** for srt/vtt/txt/tsv/aud (timecodes are integer-ms math; aud floats go
through a Python-repr formatter), and **round-trip equal** for json (semantic
parity, the Phase-1 store-JSON precedent — `json.loads` matches, key order free).

Imports only the built module + the pure-Python `whisperx.utils` writers (no torch),
so it runs in the dep-free lane. The Python writers are driven directly (their
`_core_string` returns None unless the token is on) to get the oracle bytes, then
compared against the native strings.
"""
import io
import json

import pytest

wc = pytest.importorskip("whisperx_core")
if not hasattr(wc, "write_srt"):
    pytest.skip("whisperx_core built without the writers", allow_module_level=True)

from whisperx.utils import (  # noqa: E402
    WriteAudacity,
    WriteJSON,
    WriteSRT,
    WriteTSV,
    WriteTXT,
    WriteVTT,
)

# (class, native fn) pairs; extension drives the on-disk name + the C++ entry.
TEXT_WRITERS = [
    (WriteSRT, wc.write_srt),
    (WriteVTT, wc.write_vtt),
    (WriteTXT, wc.write_txt),
    (WriteTSV, wc.write_tsv),
    (WriteAudacity, wc.write_aud),
]

DEFAULT_OPTS = {"max_line_width": None, "max_line_count": None, "highlight_words": False}


def _py_bytes(cls, result, options):
    """Run the Python writer body (the oracle) to a string buffer."""
    buf = io.StringIO()
    cls("/tmp").write_result(result, file=buf, options=options)
    return buf.getvalue()


# --- result fixtures: the contract golden + adversarial shapes ----------------

GOLDEN = {
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 1.5, "text": " Hello there friend.", "speaker": "SPEAKER_00",
         "words": [
             {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.9, "speaker": "SPEAKER_00"},
             {"word": "there", "start": 0.5, "end": 0.9, "score": 0.9, "speaker": "SPEAKER_00"},
             {"word": "friend.", "start": 0.9, "end": 1.5, "score": 0.9, "speaker": "SPEAKER_00"},
         ]},
        {"start": 2.0, "end": 3.0, "text": " Hi.", "speaker": "SPEAKER_01",
         "words": [{"word": "Hi.", "start": 2.0, "end": 3.0, "score": 0.9, "speaker": "SPEAKER_01"}]},
    ],
}

# No speakers, untimed words (no start/end keys), a "-->" to sanitize, tab in text.
NO_SPEAKER = {
    "language": "de",
    "segments": [
        {"start": 0.0, "end": 4.2, "text": "Am 1.\tJanuar --> los",
         "words": [
             {"word": "Am", "start": 0.0, "end": 0.3},
             {"word": " 1."},  # untimed: neither start nor end
             {"word": " Januar", "start": 1.0, "end": 1.8},
             {"word": " los", "start": 3.9, "end": 4.2},
         ]},
    ],
}

# A segment with no "words" at all -> the segment-level subtitle fallback.
NO_WORDS = {
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 2.0, "text": "a --> b", "speaker": "S0"},
        {"start": 2.0, "end": 5.0, "text": " trailing space "},
    ],
}

# A long multi-word segment that line-splitting + highlight options exercise.
LONG = {
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 6.0,
         "text": "the quick brown fox jumps over the lazy dog again",
         "words": [
             {"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
             for i, w in enumerate(
                 "the quick brown fox jumps over the lazy dog again".split())
         ]},
    ],
}

# No-space language join ("".join) path.
ZH = {
    "language": "zh",
    "segments": [
        {"start": 0.0, "end": 1.0, "text": "你好世界",
         "words": [
             {"word": "你好", "start": 0.0, "end": 0.5},
             {"word": "世界", "start": 0.5, "end": 1.0},
         ]},
    ],
}

RESULTS = {"golden": GOLDEN, "no_speaker": NO_SPEAKER, "no_words": NO_WORDS,
           "long": LONG, "zh": ZH}

OPTION_SETS = {
    "defaults": DEFAULT_OPTS,
    "highlight": {"max_line_width": None, "max_line_count": None, "highlight_words": True},
    "linewidth": {"max_line_width": 20, "max_line_count": 2, "highlight_words": False},
    "linewidth_hl": {"max_line_width": 20, "max_line_count": 2, "highlight_words": True},
}


@pytest.mark.parametrize("rname", list(RESULTS))
@pytest.mark.parametrize("oname", list(OPTION_SETS))
@pytest.mark.parametrize("cls,native", TEXT_WRITERS, ids=lambda x: getattr(x, "extension", ""))
def test_text_writer_byte_identical(cls, native, rname, oname):
    result, options = RESULTS[rname], OPTION_SETS[oname]
    assert native(result, options) == _py_bytes(cls, result, options)


@pytest.mark.parametrize("rname", list(RESULTS))
def test_json_writer_roundtrips(rname):
    result = RESULTS[rname]
    out = wc.write_json(result, DEFAULT_OPTS)
    assert json.loads(out) == result
    # also matches the Python writer semantically
    assert json.loads(out) == json.loads(_py_bytes(WriteJSON, result, DEFAULT_OPTS))


def test_format_timestamp_matches_python():
    from whisperx.utils import format_timestamp
    for s in [0.0, 1.5, 61.25, 3661.0, 0.0025, 0.0035, 599.999]:
        assert wc.format_timestamp(s, False, ".") == format_timestamp(s, False, ".")
        assert wc.format_timestamp(s, True, ",") == format_timestamp(s, True, ",")
