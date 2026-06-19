// Logging for the native server — spdlog with a console sink + a rotating file
// sink under <data_dir>/logs. Replaces app/server.py's logging.basicConfig and
// the per-module get_logger pattern. The core stays dep-free: it emits via the
// std::function stage/progress callbacks the server wires to these loggers.
#pragma once

#include <memory>
#include <string>

#include <spdlog/spdlog.h>

namespace whisperx::server::log {

// Initialise the global logging: console (stderr) + rotating file sink at
// <data_dir>/logs/whisperx-server.log. `level` is a spdlog level name
// (trace/debug/info/warn/error/critical), from WHISPERX_LOG_LEVEL. Idempotent.
void init(const std::string& data_dir, const std::string& level);

// A named logger (component tag), e.g. get("jobs"), get("models"). All share the
// sinks set up by init(); created on first use.
std::shared_ptr<spdlog::logger> get(const std::string& name);

}  // namespace whisperx::server::log
