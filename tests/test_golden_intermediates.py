"""Integrity guard for the committed per-stage golden intermediates.

Fast and torch-free: it does **not** run the pipeline — it checks that the
artifacts dumped by ``golden/dump_goldens.py`` are internally consistent and
un-corrupted (sha256 matches the manifest, npz arrays exist and their shapes
agree with the recorded `emission_shape`/`trellis_shape`, structural fields are
present). Regenerating the goldens is a deliberate act (run the dump script); this
just stops silent drift/corruption from creeping into the committed reference the
C++ core is diffed against.

Skipped when the intermediates haven't been generated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
INTER = ROOT / "golden" / "intermediates"
MANIFEST = INTER / "manifest.json"

if not MANIFEST.exists():
    pytest.skip("golden intermediates not generated (run golden/dump_goldens.py)",
                allow_module_level=True)

MAN = json.loads(MANIFEST.read_text())
CLIPS = sorted(MAN["clips"].items())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_has_tolerance_budget():
    tol = MAN["tolerances"]
    assert tol["emission_atol"] > 0 and tol["time_frames"] >= 1
    for k in ("tokens", "path", "char_segments", "merged_chunks", "speaker_labels"):
        assert k in tol["exact"]


@pytest.mark.parametrize("name,entry", CLIPS, ids=[c[0] for c in CLIPS])
def test_artifacts_sha256_match(name, entry):
    for kind, art in entry["artifacts"].items():
        path = INTER / art["file"]
        assert path.exists(), f"{name}: missing {kind} artifact {art['file']}"
        assert _sha256(path) == art["sha256"], f"{name}: {kind} sha256 drift"


@pytest.mark.parametrize("name,entry", CLIPS, ids=[c[0] for c in CLIPS])
def test_tensor_shapes_agree_with_align_json(name, entry):
    z = np.load(INTER / entry["artifacts"]["tensors"]["file"])
    align = json.loads((INTER / entry["artifacts"]["align"]["file"]).read_text())
    segs = align["segments"]
    assert len(segs) == entry["n_align_segments"]
    for i, seg in enumerate(segs):
        emi = z[f"seg{i}_emission"]
        assert list(emi.shape) == seg["emission_shape"]
        assert emi.dtype == np.float32
        # trellis is recomputed by the core (not stored); its recorded shape must
        # be (frames+1, tokens+1) relative to the stored emission.
        assert seg["trellis_shape"][0] == emi.shape[0] + 1
        assert seg["trellis_shape"][1] == len(seg["tokens"]) + 1
        # emission bytes hash to the recorded value (no numeric corruption)
        assert hashlib.sha256(emi.tobytes()).hexdigest() == seg["emission_sha256"]


@pytest.mark.parametrize("name,entry", CLIPS, ids=[c[0] for c in CLIPS])
def test_align_structural_fields(name, entry):
    align = json.loads((INTER / entry["artifacts"]["align"]["file"]).read_text())
    for seg in align["segments"]:
        assert seg["tokens"], f"{name}: empty tokens"
        assert all(isinstance(t, int) for t in seg["tokens"])
        if seg["path"] is not None:  # successful backtrack
            assert seg["char_segments"], f"{name}: path but no char-segments"
            # path time/token indices are ints; char-segments cover [start,end)
            for tok_i, time_i, score in seg["path"]:
                assert isinstance(tok_i, int) and isinstance(time_i, int)
            for label, start, end, score in seg["char_segments"]:
                assert isinstance(start, int) and end > start


@pytest.mark.parametrize("name,entry", CLIPS, ids=[c[0] for c in CLIPS])
def test_transcript_and_vad_present(name, entry):
    tr = json.loads((INTER / entry["artifacts"]["transcript"]["file"]).read_text())
    assert tr["segments"] and all("text" in s for s in tr["segments"])
    vad = json.loads((INTER / entry["artifacts"]["vad"]["file"]).read_text())
    assert vad["merged_chunks"], f"{name}: no merged VAD chunks"
