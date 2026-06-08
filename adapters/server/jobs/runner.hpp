// The transcription run_session — the C++ port of app/server.py::run_session +
// the native branch of app/pipeline.py::run_job (and the pybind run_job Steps
// wiring). Builds the orchestrate::run_job Steps over the resident native engines
// (decode -> silero VAD + merge_chunks -> WhisperSherpa -> align_run -> optional
// SherpaDiarizer + assign_word_speakers), fires stage/duration progress, writes
// the export artifacts via the native writers, and marks the session done.
//
// Returned as a JobQueue::RunSession closure so the queue owns lifecycle/errors.
#pragma once

#include "config.hpp"
#include "jobs/jobs.hpp"

namespace whisperx::db {
class SessionStore;
}
namespace whisperx::server::sse {
class Broker;
}
namespace whisperx::server::models {
class ModelManager;
}

namespace whisperx::server::jobs {

// Build the run_session callback bound to the given collaborators. They must
// outlive the returned closure (they are server-lifetime singletons).
RunSession make_run_session(whisperx::db::SessionStore& store,
                            whisperx::server::models::ModelManager& manager,
                            whisperx::server::sse::Broker& broker,
                            const whisperx::server::Config& cfg);

}  // namespace whisperx::server::jobs
