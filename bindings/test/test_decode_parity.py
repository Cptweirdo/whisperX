"""Phase-2 (slice 2B) parity oracle: the in-process C++ ``whisperx_core.load_audio``
(libav* linked) vs the Python ``whisperx.audio._py_load_audio`` it replaces (the
ffmpeg **subprocess**, the ``decode`` stage token).

Both paths use ffmpeg's own decoders + ``swresample`` (the C++ links the same
``libav*`` the CLI wraps), so the PCM is compared **sample-for-sample**:

* the golden **wav** clips are already 16 kHz mono s16 — pure decode + s16
  conversion → **bit-exact** (``array_equal``).
* the **m4a** clip (44.1 kHz stereo AAC) exercises the AAC decoder + downmix +
  resample → compared within ``atol = 2 / 32768`` (a couple LSB) to absorb only
  legitimate resampler/dither differences.

Skipped unless the C++ module was built with the audio stage
(``cmake -DWHISPERX_CORE_AUDIO=ON``); run with the build dir on the path
(``PYTHONPATH=build``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

for cand in (ROOT / "build", ROOT / "build" / "Release", ROOT / "build" / "Debug"):
    if cand.is_dir():
        sys.path.insert(0, str(cand))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

whisperx_core = pytest.importorskip(
    "whisperx_core", reason="build the C++ module first: cmake --build build")

if not hasattr(whisperx_core, "load_audio"):
    pytest.skip("module built without WHISPERX_CORE_AUDIO (no in-process decode)",
                allow_module_level=True)

SAMPLE_RATE = 16000


def _subprocess_load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """The ffmpeg-subprocess oracle — a verbatim copy of whisperx.audio.load_audio's
    body, inlined so this test stays torch-free (no `import whisperx`)."""
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", path, "-f", "s16le",
           "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


MANIFEST = json.loads((ROOT / "golden" / "manifest.json").read_text())


def _clips():
    """(clip_name, abs_path, exact?) for every decodable golden clip."""
    for name, meta in MANIFEST["clips"].items():
        f = meta.get("file")
        if not f:
            continue
        path = ROOT / "golden" / f
        # wav goldens are 16 kHz mono → bit-exact; compressed/off-rate are atol.
        exact = path.suffix.lower() == ".wav"
        yield pytest.param(name, path, exact, id=name)


@pytest.mark.parametrize("name, path, exact", list(_clips()))
def test_load_audio_matches_subprocess(name, path, exact):
    assert path.exists(), f"missing golden clip {path}"
    cpp = whisperx_core.load_audio(str(path), SAMPLE_RATE)
    ref = _subprocess_load_audio(str(path), SAMPLE_RATE)

    assert cpp.dtype == np.float32
    assert cpp.ndim == 1
    assert cpp.shape == ref.shape, f"{name}: {cpp.shape} vs {ref.shape}"

    if exact:
        assert np.array_equal(cpp, ref), (
            f"{name}: not bit-exact; max|Δ|={np.max(np.abs(cpp - ref)):.3e}")
    else:
        # ≤ 2 LSB of int16 — legitimate resample/dither slack only.
        np.testing.assert_allclose(cpp, ref, atol=2 / 32768.0, rtol=0)
