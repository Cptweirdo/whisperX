// Typed session row — the C++ shape of one `sessions` table row.
//
// SessionStore reads/writes these instead of raw JSON objects: enum Status /
// Stage, std::optional for nullable columns, a typed translations map. Only
// genuinely opaque payloads stay nlohmann::json (`options` — user-supplied,
// schema evolves — and the file-backed transcript/edits/translation sidecars).
//
// `to_json()` reproduces app.store._row_to_dict exactly (15 keys, JSON null
// for NULL columns, options/translations as parsed objects, diarized bool,
// num_segments int64, duration double, the rest strings) — the pybind facade
// and the parity oracle (bindings/test/test_store_parity.py) depend on it.
//
// Lenient read policy (get() must never throw on a hand-edited/legacy DB):
//   * unknown `status` string  -> Status::Error
//   * unknown / NULL `stage`   -> nullopt
//   * translation entry with a missing/unknown status -> dropped from the map
//     (it survives on disk: the write path is a raw-JSON read-modify-write).
#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace whisperx::db {

using nlohmann::json;

// Session lifecycle AND per-language translation status (the SPA reuses one
// Status union for both — app/web/src/lib/types.ts).
enum class Status { Queued, Running, Done, Error };

// Pipeline stages, in order — the five names orchestrate.cpp publishes.
enum class Stage { Decoding, Transcribing, LoadingAlign, Aligning, Diarizing };

std::string to_string(Status status);
std::string to_string(Stage stage);
std::optional<Status> parse_status(std::string_view s);
std::optional<Stage> parse_stage(std::string_view s);

// One entry of the `translations` JSON column: {"status": ..., "service"?,
// "error"?}. `service` is a free string ("google", "deepl", ...).
struct TranslationEntry {
    Status status = Status::Queued;
    std::optional<std::string> service;
    std::optional<std::string> error;

    json to_json() const;
    // nullopt when `j` is not an object or its status is missing/unknown.
    static std::optional<TranslationEntry> from_json(const json& j);
};

using TranslationMap = std::map<std::string, TranslationEntry>;  // lang -> entry

json translations_to_json(const TranslationMap& m);
// Lenient: non-object input -> empty map; unparseable entries are skipped.
TranslationMap translations_from_json(const json& j);

struct SessionRow {
    std::string id;
    std::optional<std::string> filename;
    std::optional<std::string> audio_filename;
    Status status = Status::Queued;
    std::optional<Stage> stage;
    std::optional<std::string> error;
    // NULL column -> json null; bad stored JSON -> {} (mirrors _row_to_dict).
    json options;
    std::optional<std::string> language;
    std::optional<bool> diarized;
    std::optional<std::string> model;
    std::optional<long long> num_segments;
    std::optional<double> duration;
    std::optional<TranslationMap> translations;  // nullopt = NULL column
    std::string created_at;  // ISO-8601 UTC seconds (display-only)
    std::string updated_at;

    json to_json() const;
};

}  // namespace whisperx::db
