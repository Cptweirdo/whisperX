"""Seed a finished demo session for the Playwright e2e suite.

Writes a DB row + transcript.json directly via SessionStore (bypassing the
model-bound pipeline) into WHISPERX_DATA_DIR, and marks the app onboarded so the
SPA boots straight to the dashboard. Run by tests/e2e/serve.sh before Flask starts.
"""
import json
import os
import sys

# repo root = app/web/tests/e2e -> ../../../..
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO)

from app.store import SessionStore  # noqa: E402

SID = "e2e-demo"
# Alternating speakers -> three turns; reassigning the middle one collapses to one.
SEGMENTS = [
    {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00", "text": "Welcome everyone to the meeting today.",
     "words": [{"word": w, "start": i * 0.2, "end": i * 0.2 + 0.2}
               for i, w in enumerate("Welcome everyone to the meeting today.".split())]},
    {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01", "text": "Thanks for having me, glad to be here.",
     "words": [{"word": w, "start": 1.5 + i * 0.2, "end": 1.7 + i * 0.2}
               for i, w in enumerate("Thanks for having me, glad to be here.".split())]},
    {"start": 3.0, "end": 4.5, "speaker": "SPEAKER_00", "text": "Let us begin with the quarterly numbers.",
     "words": [{"word": w, "start": 3.0 + i * 0.2, "end": 3.2 + i * 0.2}
               for i, w in enumerate("Let us begin with the quarterly numbers.".split())]},
]


def main() -> None:
    data_dir = os.environ.get("WHISPERX_DATA_DIR")
    if not data_dir:
        raise SystemExit("WHISPERX_DATA_DIR must be set")
    store = SessionStore(data_dir)
    store.set_setting("onboarded", "1")
    os.makedirs(store.session_dir(SID), exist_ok=True)
    with open(os.path.join(store.session_dir(SID), "audio.wav"), "wb") as f:
        f.write(b"RIFF")  # dummy; the player just 404s/empties, transcript still renders
    store.create(SID, filename="Quarterly board meeting", audio_filename="audio.wav",
                 options={}, model="large-v3")
    store.mark_done(SID, language="en", diarized=True, model="large-v3",
                    num_segments=len(SEGMENTS), duration=4.5)
    with open(store.result_path(SID), "w", encoding="utf-8") as f:
        json.dump({"segments": SEGMENTS, "language": "en",
                   "num_segments": len(SEGMENTS), "duration": 4.5}, f)
    print(f"seeded session {SID} into {data_dir}")


if __name__ == "__main__":
    main()
