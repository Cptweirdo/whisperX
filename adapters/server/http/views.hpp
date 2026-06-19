// Response shaping — the C++ port of app/server.py's _card / _summary /
// _build_turns / _turn_words / _models_event and app/render.py's resolve_label /
// render_markdown. Pure JSON in/out, reusing whisperx::edits::group_turns /
// distinct_speakers (the same turn-index contract the edit endpoints use) so the
// SPA renders identically whether the host is Flask or oat++.
#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "db/session_row.hpp"

namespace whisperx::server::views {

using nlohmann::json;

// render.py::_speaker_label / resolve_label: SPEAKER_00 -> "Speaker 1"; a stored
// display name wins. `raw` may be a JSON string or null; `names` is {key: name}.
std::string speaker_label(const json& raw);
std::string resolve_label(const json& raw, const json& names);

// server.py::_turn_words — flatten a turn's words to {text,start?,end?,stale?}.
json turn_words(const json& segments, const json& seg_indices);

// server.py::_build_turns — speaker-grouped turns the SPA renders (indices from
// group_turns; empty-text turns skipped without renumbering).
json build_turns(const json& segments, const json& names);

// server.py::_card / _summary — dashboard card + list summary.
json card(const db::SessionRow& row);
json summary(const std::vector<db::SessionRow>& rows);

// server.py::_models_event — the /models/events payload from a ModelManager status.
json models_event(const json& status);

// render.py::render_markdown — Markdown transcript of the (overlaid) segments.
std::string render_markdown(const json& result, const json& names,
                            const std::string& title);

}  // namespace whisperx::server::views
