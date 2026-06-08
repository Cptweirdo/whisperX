"""Phase 5 align-driver parity: the native char-cleaner (whisperx_core.clean_segment)
vs the committed goldens + the alignment.py:252-280 Python oracle.

Torch-free. The committed `*.align.json` carries each segment's `text_clean`
(== "".join(clean_char)) + `clean_cdx` + the model `dictionary`; pairing them with
the original `text` in `*.transcript.json` makes this a self-contained CI check.
Skips cleanly if the module lacks the cleaner or numpy is unavailable.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
wc = pytest.importorskip("whisperx_core")

if not hasattr(wc, "clean_segment"):
    pytest.skip("whisperx_core built without clean_segment", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
INTER = ROOT / "golden" / "intermediates"

CLIPS = ["en_libri", "en_dialog", "de_dialog", "ru_dialog",
         "ru_cv_71085", "ru_cv_71379", "ru_cv_71606", "ru_cv_79125"]

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]


def _py_clean(text, lang, dictionary):
    """The alignment.py:252-280 char-cleaning, verbatim — the oracle."""
    num_leading = len(text) - len(text.lstrip())
    num_trailing = len(text) - len(text.rstrip())
    clean_char, clean_cdx = [], []
    for cdx, char in enumerate(text):
        char_ = char.lower()
        if lang not in LANGUAGES_WITHOUT_SPACES:
            char_ = char_.replace(" ", "|")
        if cdx < num_leading:
            pass
        elif cdx > len(text) - num_trailing - 1:
            pass
        elif char_ in dictionary.keys():
            clean_char.append(char_)
            clean_cdx.append(cdx)
        elif char_ not in (" ", "|"):
            clean_char.append(char_)
            clean_cdx.append(cdx)
    return "".join(clean_char), clean_cdx


# --- 1. native cleaner reproduces the committed goldens exactly ---------------
@pytest.mark.parametrize("clip", CLIPS)
def test_clean_segment_matches_golden(clip):
    align = json.loads((INTER / f"{clip}.align.json").read_text())
    transcript = json.loads((INTER / f"{clip}.transcript.json").read_text())["segments"]
    dictionary = align["dictionary"]
    lang = clip.split("_")[0]
    for aseg, tseg in zip(align["segments"], transcript):
        clean_char, clean_cdx = wc.clean_segment(tseg["text"], lang, dictionary)
        assert clean_char == aseg["text_clean"], f"{clip}: {tseg['text']!r}"
        assert list(clean_cdx) == aseg["clean_cdx"], f"{clip}: {tseg['text']!r}"


# --- 2. native cleaner == Python oracle on adversarial inputs -----------------
@pytest.mark.parametrize("text,lang", [
    ("  Leading and trailing.  ", "en"),
    ("Mixed CASE Text", "en"),
    ("digits 123 and $ymbols!", "en"),       # OOV kept as wildcard
    ("\tTabbed\tinterior\t", "en"),
    ("Это русский текст", "ru"),               # Cyrillic lowercase
    ("Ωμέγα ΚΑΙ άλφα", "el"),                  # Greek lowercase
    ("Über Straße fahren", "de"),              # ß / umlaut (no uppercase ß here)
    ("你 好 世 界", "zh"),                       # no-spaces language
    ("", "en"),
    ("    ", "en"),
])
def test_clean_segment_matches_python_oracle(text, lang):
    # Use a permissive dict (every lowered char a key) so the dict-key branch and
    # the wildcard branch are both exercised across two dictionaries.
    full = {c: i for i, c in enumerate(sorted(set(text.lower().replace(" ", "|"))))}
    sparse = {"|": 0}  # forces everything non-space down the wildcard branch
    for d in (full, sparse):
        exp_char, exp_cdx = _py_clean(text, lang, d)
        clean_char, clean_cdx = wc.clean_segment(text, lang, d)
        assert clean_char == exp_char, f"{text!r} lang={lang} dict={'full' if d is full else 'sparse'}"
        assert list(clean_cdx) == exp_cdx, f"{text!r} lang={lang}"
