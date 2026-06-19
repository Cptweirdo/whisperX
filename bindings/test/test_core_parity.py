"""Phase-0 parity oracle: the C++ `whisperx_core` module vs the Python reference.

This is the trivial-but-real first link of the strangler-fig loop — it proves we
can call a C++ function from pytest and get **bit-identical** results to the
Python implementation it was ported from. As later stages land, their parity
tests join this file / `bindings/test/`.

Skipped unless the module has been built. Build it with::

    cmake -S . -B build -G Ninja && cmake --build build

then run with the build dir on the path::

    PYTHONPATH=build uv run pytest bindings/test/test_core_parity.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Make a freshly-built module importable without installing it.
for cand in (ROOT / "build", ROOT / "build" / "Release", ROOT / "build" / "Debug"):
    if cand.is_dir():
        sys.path.insert(0, str(cand))

whisperx_core = pytest.importorskip(
    "whisperx_core", reason="build the C++ module first: cmake --build build")


def _py_edit_distance(a: list[str], b: list[str]) -> int:
    """The Python reference (copied from tests/test_baseline_golden.py)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def test_module_metadata():
    assert isinstance(whisperx_core.__version__, str)
    import json
    info = json.loads(whisperx_core.build_info_json())
    assert info["version"] == whisperx_core.__version__
    assert "nlohmann_json" in info  # the linked dep is reported


CASES = [
    ([], []),
    (["the", "cat"], ["the", "dog"]),
    (["a", "b", "c"], ["a", "x", "c", "d"]),
    (["a", "b", "c"], []),
    ("transcribe the audio".split(), "transcribed the audio".split()),
    # Cyrillic tokens (RU golden path)
    (["привет", "мир"], ["привет", "земля"]),
]


@pytest.mark.parametrize("a,b", CASES, ids=lambda x: " ".join(x) or "<empty>")
def test_edit_distance_parity(a, b):
    assert whisperx_core.edit_distance(a, b) == _py_edit_distance(a, b)


@pytest.mark.parametrize(
    "a,b",
    [("kitten", "sitting"), ("", "abc"), ("да", "да"), ("hello", "h3llo")],
)
def test_char_edit_distance_parity(a, b):
    expect = _py_edit_distance(list(a), list(b))
    assert whisperx_core.char_edit_distance(a, b) == expect
