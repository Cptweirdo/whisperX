// ISO-8601 UTC-seconds timestamp, matching Python's
// datetime.now(timezone.utc).isoformat(timespec="seconds") -> "...T..+00:00".
// Shared by the SQLite store (row timestamps) and the edits overlay (delta "ts").
#pragma once

#include <chrono>
#include <ctime>
#include <string>

namespace whisperx {

inline std::string now_iso() {
    const auto t = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S+00:00", &tm);
    return buf;
}

}  // namespace whisperx
