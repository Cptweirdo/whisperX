#include "db/session_row.hpp"

namespace whisperx::db {

std::string to_string(Status status) {
    switch (status) {
        case Status::Queued: return "queued";
        case Status::Running: return "running";
        case Status::Done: return "done";
        case Status::Error: return "error";
    }
    return "error";  // unreachable
}

std::string to_string(Stage stage) {
    switch (stage) {
        case Stage::Decoding: return "decoding";
        case Stage::Transcribing: return "transcribing";
        case Stage::LoadingAlign: return "loading_align";
        case Stage::Aligning: return "aligning";
        case Stage::Diarizing: return "diarizing";
    }
    return "decoding";  // unreachable
}

std::optional<Status> parse_status(std::string_view s) {
    if (s == "queued") return Status::Queued;
    if (s == "running") return Status::Running;
    if (s == "done") return Status::Done;
    if (s == "error") return Status::Error;
    return std::nullopt;
}

std::optional<Stage> parse_stage(std::string_view s) {
    if (s == "decoding") return Stage::Decoding;
    if (s == "transcribing") return Stage::Transcribing;
    if (s == "loading_align") return Stage::LoadingAlign;
    if (s == "aligning") return Stage::Aligning;
    if (s == "diarizing") return Stage::Diarizing;
    return std::nullopt;
}

json TranslationEntry::to_json() const {
    json j = json::object();
    j["status"] = to_string(status);
    if (service.has_value()) j["service"] = *service;
    if (error.has_value()) j["error"] = *error;
    return j;
}

std::optional<TranslationEntry> TranslationEntry::from_json(const json& j) {
    if (!j.is_object()) return std::nullopt;
    auto st = j.find("status");
    if (st == j.end() || !st->is_string()) return std::nullopt;
    const auto parsed = parse_status(st->get<std::string>());
    if (!parsed.has_value()) return std::nullopt;
    TranslationEntry e;
    e.status = *parsed;
    auto sv = j.find("service");
    if (sv != j.end() && sv->is_string()) e.service = sv->get<std::string>();
    auto er = j.find("error");
    if (er != j.end() && er->is_string()) e.error = er->get<std::string>();
    return e;
}

json translations_to_json(const TranslationMap& m) {
    json out = json::object();
    for (const auto& [lang, entry] : m) {
        out[lang] = entry.to_json();
    }
    return out;
}

TranslationMap translations_from_json(const json& j) {
    TranslationMap out;
    if (!j.is_object()) return out;
    for (const auto& [lang, entry] : j.items()) {
        if (auto e = TranslationEntry::from_json(entry)) {
            out.emplace(lang, std::move(*e));
        }
    }
    return out;
}

json SessionRow::to_json() const {
    auto opt_str = [](const std::optional<std::string>& v) {
        return v.has_value() ? json(*v) : json(nullptr);
    };
    json d = json::object();
    d["id"] = id;
    d["filename"] = opt_str(filename);
    d["audio_filename"] = opt_str(audio_filename);
    d["status"] = to_string(status);
    d["stage"] = stage.has_value() ? json(to_string(*stage)) : json(nullptr);
    d["error"] = opt_str(error);
    d["options"] = options;
    d["language"] = opt_str(language);
    d["diarized"] = diarized.has_value() ? json(*diarized) : json(nullptr);
    d["model"] = opt_str(model);
    d["num_segments"] =
        num_segments.has_value() ? json(*num_segments) : json(nullptr);
    d["duration"] = duration.has_value() ? json(*duration) : json(nullptr);
    d["translations"] = translations.has_value()
                            ? translations_to_json(*translations)
                            : json(nullptr);
    d["created_at"] = created_at;
    d["updated_at"] = updated_at;
    return d;
}

}  // namespace whisperx::db
