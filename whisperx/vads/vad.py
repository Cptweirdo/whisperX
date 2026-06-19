import os
from typing import Optional

import pandas as pd
from pyannote.core import Annotation, Segment


def _core_vad_enabled() -> bool:
    """Whether the C++ ``whisperx_core`` VAD algorithms back this run.

    ``WHISPERX_CORE_STAGES`` carries comma-separated stage tokens (``db`` /
    ``edits`` / ``vad`` / …); ``vad`` routes ``Vad.merge_chunks`` to the C++ port,
    with the pure-Python ``_py_merge_chunks`` kept as the live parity oracle.
    """
    raw = os.environ.get("WHISPERX_CORE_STAGES", "")
    return "vad" in {s.strip() for s in raw.split(",") if s.strip()}


def _core():
    """Import the built C++ module (lazily; opt-in build path)."""
    import whisperx_core
    return whisperx_core


class Vad:
    def __init__(self, vad_onset):
        if not (0 < vad_onset < 1):
            raise ValueError(
                "vad_onset is a decimal value between 0 and 1."
            )

    @staticmethod
    def preprocess_audio(audio):
        pass

    # keep merge_chunks as static so it can be also used by manually assigned vad_model (see 'load_model')
    @staticmethod
    def merge_chunks(segments,
                     chunk_size,
                     onset: float,
                     offset: Optional[float]):
        """Merge operation described in paper (facade: routes to C++ under `vad`)."""
        if _core_vad_enabled():
            tuples = [
                (float(s.start), float(s.end),
                 None if s.speaker is None else str(s.speaker))
                for s in segments
            ]
            merged = _core().merge_chunks(tuples, chunk_size, onset, offset)
            # C++ emits inner [start, end] lists; restore tuples so the result is
            # byte-identical to the Python oracle (seg_idxs are (start, end) tuples).
            for m in merged:
                m["segments"] = [tuple(p) for p in m["segments"]]
            return merged
        return Vad._py_merge_chunks(segments, chunk_size, onset, offset)

    @staticmethod
    def _py_merge_chunks(segments,
                         chunk_size,
                         onset: float,
                         offset: Optional[float]):
        """
         Merge operation described in paper
         """
        curr_end = 0
        merged_segments = []
        seg_idxs: list[tuple]= []
        speaker_idxs: list[Optional[str]] = []

        curr_start = segments[0].start
        for seg in segments:
            if seg.end - curr_start > chunk_size and curr_end - curr_start > 0:
                merged_segments.append({
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                })
                curr_start = seg.start
                seg_idxs = []
                speaker_idxs = []
            curr_end = seg.end
            seg_idxs.append((seg.start, seg.end))
            speaker_idxs.append(seg.speaker)
        # add final
        merged_segments.append({
            "start": curr_start,
            "end": curr_end,
            "segments": seg_idxs,
        })

        return merged_segments

