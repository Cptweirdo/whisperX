// Shared server state handed to the oat++ controllers — the collaborators that
// in app/server.py are module globals (_sessions, _manager, _queue, _broker).
#pragma once

#include <string>

#include "config.hpp"

namespace whisperx::db {
class SessionStore;
}
namespace whisperx::server::models {
class ModelManager;
}
namespace whisperx::server::jobs {
class JobQueue;
}
namespace whisperx::server::sse {
class Broker;
}

namespace whisperx::server {

// Reserved SSE channel for global model-load state (server.py MODELS_CHANNEL).
inline constexpr const char* kModelsChannel = "__models__";

struct AppState {
    whisperx::db::SessionStore& store;
    whisperx::server::models::ModelManager& manager;
    whisperx::server::jobs::JobQueue& queue;
    whisperx::server::sse::Broker& broker;
    Config cfg;
};

}  // namespace whisperx::server
