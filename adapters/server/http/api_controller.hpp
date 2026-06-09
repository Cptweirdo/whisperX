// The oat++ ApiController reproducing the MVP slice of app/server.py's JSON/SSE
// surface (docs/api-reference.md). Raw-JSON responses (nlohmann -> string), the
// _body() tolerance, multipart upload, the transcript edit/speaker endpoints, the
// model/device + settings/onboarding endpoints, the binary downloads + export.md,
// and the per-session / models SSE streams. Deferred features (translation,
// backup) return SPA-compatible shapes so the client degrades gracefully.
#pragma once

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <regex>
#include <string>

#include <nlohmann/json.hpp>

#include "oatpp/core/macro/codegen.hpp"
#include "oatpp/web/mime/multipart/FileProvider.hpp"
#include "oatpp/web/mime/multipart/InMemoryDataProvider.hpp"
#include "oatpp/web/mime/multipart/PartList.hpp"
#include "oatpp/web/mime/multipart/Reader.hpp"
#include "oatpp/web/server/api/ApiController.hpp"

#include "audio/decode.hpp"
#include "db/session_store.hpp"
#include "edits/edits.hpp"
#include "http/app_state.hpp"
#include "http/http_util.hpp"
#include "http/views.hpp"
#include "jobs/jobs.hpp"
#include "log/log.hpp"
#include "models/model_manager.hpp"
#include "oauth/backup_service.hpp"
#include "oauth/flow.hpp"
#include "secrets/hf_verify.hpp"
#include "secrets/keyring.hpp"
#include "sse/broker.hpp"
#include "sse/sse_response.hpp"
#include "translate/google.hpp"
#include "translate/overlay.hpp"
#include "translate/queue.hpp"
#include "writers/writers.hpp"

namespace whisperx::server {

namespace fs = std::filesystem;
using nlohmann::json;
namespace mpart = oatpp::web::mime::multipart;

#include OATPP_CODEGEN_BEGIN(ApiController)

class ApiController : public oatpp::web::server::api::ApiController {
public:
    ApiController(AppState& app, const std::shared_ptr<ObjectMapper>& om)
        : oatpp::web::server::api::ApiController(om), app_(app) {}

private:
    AppState& app_;

    using Resp = std::shared_ptr<OutgoingResponse>;

    Resp jr(const Status& s, const json& body) {
        return http_util::json_response(s, body);
    }

    // (result, edit-overlaid segments) — server.py::_current_segments.
    json current_segments(const std::string& id) {
        json result = app_.store.load_result(id);
        json orig = (result.is_object() && result.contains("segments"))
                        ? result["segments"]
                        : json::array();
        return app_.store.current_segments(id, orig);
    }

    json transcript_payload(const std::string& id) {
        json segs = current_segments(id);
        json names = app_.store.get_speaker_names(id);
        return {
            {"turns", views::build_turns(segs, names)},
            {"segments", segs},
            {"can_undo", app_.store.edit_history_len(id) > 0},
        };
    }

    bool models_ready() {
        json st = app_.manager.status();
        std::string active = st.value("active", "");
        for (const auto& m : st.value("models", json::array()))
            if (m.value("name", "") == active) return m.value("loaded", false);
        return false;
    }

    // A safe BCP-47-ish language tag (also guards translation file paths) —
    // server.py::_valid_lang.
    static bool valid_lang(const std::string& lang) {
        static const std::regex re(R"(^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})?$)");
        return std::regex_match(lang, re);
    }

    static std::string trim(const std::string& s) {
        auto b = s.find_first_not_of(" \t\r\n");
        return b == std::string::npos
                   ? ""
                   : s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
    }

public:
    // --- Health ----------------------------------------------------------
    ENDPOINT("GET", "/healthz", healthz) {
        return jr(Status::CODE_200,
                  {{"status", "ok"}, {"models_ready", models_ready()}});
    }

    // --- Sessions --------------------------------------------------------
    ENDPOINT("GET", "/api/sessions", list_sessions) {
        json rows = app_.store.list();
        json cards = json::array();
        for (const auto& r : rows) cards.push_back(views::card(r));
        return jr(Status::CODE_200,
                  {{"sessions", cards}, {"summary", views::summary(rows)}});
    }

    ENDPOINT("POST", "/api/sessions", create_session,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        if (!models_ready())
            return jr(Status::CODE_503, {{"error", "loading_models"}});

        // Parse the multipart: audio -> temp file, other fields in-memory.
        auto multipart = std::make_shared<mpart::PartList>(request->getHeaders());
        mpart::Reader reader(multipart.get());
        std::string tmp_audio =
            (fs::temp_directory_path() /
             ("wxupload-" + std::to_string(::rand()) + ".bin"))
                .string();
        reader.setPartReader("audio", mpart::createFilePartReader(tmp_audio));
        reader.setDefaultPartReader(
            mpart::createInMemoryPartReader(64 * 1024));
        request->transferBody(&reader);

        auto field = [&](const char* name) -> std::string {
            auto p = multipart->getNamedPart(name);
            if (!p || !p->getPayload()) return "";
            auto d = p->getPayload()->getInMemoryData();
            return d ? std::string(*d) : "";
        };
        auto audio_part = multipart->getNamedPart("audio");
        std::string orig_name =
            (audio_part && audio_part->getFilename())
                ? std::string(*audio_part->getFilename())
                : "";
        std::error_code ec;
        if (orig_name.empty() || !fs::exists(tmp_audio, ec) ||
            fs::file_size(tmp_audio, ec) == 0) {
            fs::remove(tmp_audio, ec);
            return jr(Status::CODE_400, {{"error", "No audio file uploaded."}});
        }

        std::string requested = field("model");
        std::string model;
        if (!requested.empty()) {
            if (!models::is_known_model(requested)) {
                fs::remove(tmp_audio, ec);
                return jr(Status::CODE_400,
                          {{"error", "Unknown model: " + requested}});
            }
            model = requested;
        } else {
            model = app_.manager.active();
        }

        std::string session_id;
        {  // uuid4-hex-ish
            static const char* hex = "0123456789abcdef";
            for (int i = 0; i < 32; ++i) session_id += hex[std::rand() & 0xF];
        }
        std::string safe = http_util::secure_filename(orig_name);
        std::string ext = fs::path(safe).extension().string();
        if (ext.empty()) ext = ".bin";
        std::string audio_filename = "audio" + ext;
        fs::path session_dir =
            fs::path(app_.cfg.data_dir) / "sessions" / session_id;
        fs::create_directories(session_dir, ec);
        fs::path saved = session_dir / audio_filename;
        fs::rename(tmp_audio, saved, ec);
        if (ec) {  // cross-device rename — fall back to copy
            fs::copy_file(tmp_audio, saved,
                          fs::copy_options::overwrite_existing, ec);
            fs::remove(tmp_audio, ec);
        }

        // Reject over-long audio upfront (header probe, no decode).
        if (app_.cfg.max_audio_hours > 0) {
            double dur = whisperx::audio::probe_duration(saved.string());
            if (dur > 0 && dur > app_.cfg.max_audio_hours * 3600.0) {
                fs::remove_all(session_dir, ec);
                char msg[256];
                std::snprintf(msg, sizeof(msg),
                              "Audio is %.1f h, over the %g h limit. Split it "
                              "into shorter files and upload each.",
                              dur / 3600.0, app_.cfg.max_audio_hours);
                return jr(Status::CODE_413,
                          {{"error", msg},
                           {"duration_hours", dur / 3600.0},
                           {"max_hours", app_.cfg.max_audio_hours}});
            }
        }

        auto int_field = [&](const char* name) -> json {
            std::string v = field(name);
            if (!v.empty() &&
                v.find_first_not_of("0123456789") == std::string::npos)
                return std::stoi(v);
            return nullptr;
        };
        std::string lang = field("language");
        json options = {
            {"language", lang.empty() ? json(nullptr) : json(lang)},
            {"min_speakers", int_field("min_speakers")},
            {"max_speakers", int_field("max_speakers")},
        };
        std::string display = field("name");
        if (display.empty()) display = orig_name;
        app_.store.create(session_id, display, audio_filename, options,
                          std::optional<std::string>(model));
        app_.queue.submit(session_id);
        return jr(Status::CODE_201,
                  {{"id", session_id}, {"status", "queued"}});
    }

    ENDPOINT("GET", "/api/sessions/{id}", get_session, PATH(String, id)) {
        std::string sid = id;
        json row = app_.store.get(sid);
        if (row.is_null()) return jr(Status::CODE_404, json::object());
        json names = app_.store.get_speaker_names(sid);
        json card = views::card(row);
        json result = app_.store.load_result(sid);
        if (result.is_object()) {
            json orig = result.contains("segments") ? result["segments"]
                                                     : json::array();
            json segs = app_.store.current_segments(sid, orig);
            result["segments"] = segs;
            card["turns"] = views::build_turns(segs, names);
        }
        card["result"] = result.is_null() ? json(nullptr) : result;
        card["speaker_names"] = names;
        card["can_undo"] = app_.store.edit_history_len(sid) > 0;
        card["created_at"] = row.value("created_at", json(nullptr));
        card["updated_at"] = row.value("updated_at", json(nullptr));
        card["options"] = (row.contains("options") && row["options"].is_object())
                              ? row["options"]
                              : json::object();
        json formats = json::array();
        for (const char* fmt : {"srt", "vtt", "txt", "json"}) {
            std::error_code ec;
            if (fs::exists(fs::path(app_.cfg.data_dir) / "sessions" / sid /
                               (std::string("transcript.") + fmt),
                           ec))
                formats.push_back(fmt);
        }
        card["formats"] = formats;
        return jr(Status::CODE_200, card);
    }

    ENDPOINT("POST", "/api/sessions/{id}/rename", rename_session,
             PATH(String, id),
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        std::string sid = id;
        if (app_.store.get(sid).is_null())
            return jr(Status::CODE_404, json::object());
        json body = http_util::parse_body(request);
        std::string name = body.value("name", "");
        // trim
        auto b = name.find_first_not_of(" \t\r\n");
        name = (b == std::string::npos)
                   ? ""
                   : name.substr(b, name.find_last_not_of(" \t\r\n") - b + 1);
        if (name.empty())
            return jr(Status::CODE_400, {{"error", "Name cannot be empty."}});
        app_.store.rename(sid, name);
        return jr(Status::CODE_200, {{"id", sid}, {"filename", name}});
    }

    ENDPOINT("POST", "/api/sessions/{id}/delete", delete_session,
             PATH(String, id)) {
        std::string sid = id;
        app_.queue.cancel(sid);
        if (!app_.store.remove(sid)) return jr(Status::CODE_404, json::object());
        return jr(Status::CODE_200, {{"deleted", true}});
    }

    // --- Transcript editing ----------------------------------------------
    ENDPOINT("POST", "/api/sessions/{id}/turns/{idx}", edit_turn,
             PATH(String, id), PATH(String, idx),
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        std::string sid = id;
        if (app_.store.get(sid).is_null())
            return jr(Status::CODE_404, json::object());
        json body = http_util::parse_body(request);
        std::string text = body.value("text", "");
        try {
            app_.store.save_turn_edit(sid, std::stol(idx), text);
        } catch (const std::exception&) {
            return jr(Status::CODE_400, {{"error", "Unknown turn."}});
        }
        return jr(Status::CODE_200, transcript_payload(sid));
    }

    ENDPOINT("POST", "/api/sessions/{id}/undo", undo_edit, PATH(String, id)) {
        std::string sid = id;
        if (app_.store.get(sid).is_null())
            return jr(Status::CODE_404, json::object());
        app_.store.undo_turn_edit(sid);
        return jr(Status::CODE_200, transcript_payload(sid));
    }

    // --- Speakers --------------------------------------------------------
    ENDPOINT("GET", "/api/sessions/{id}/speakers", list_speakers,
             PATH(String, id)) {
        std::string sid = id;
        if (app_.store.get(sid).is_null())
            return jr(Status::CODE_404, json::object());
        json segs = current_segments(sid);
        json names = app_.store.get_speaker_names(sid);
        json out = json::array();
        for (const auto& key : whisperx::edits::distinct_speakers(segs))
            out.push_back({{"key", key},
                           {"label", views::resolve_label(key, names)}});
        return jr(Status::CODE_200, out);
    }

    ENDPOINT("POST", "/api/sessions/{id}/speakers", rename_speaker,
             PATH(String, id),
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        std::string sid = id;
        if (app_.store.get(sid).is_null())
            return jr(Status::CODE_404, json::object());
        json body = http_util::parse_body(request);
        std::string speaker = body.value("speaker", "");
        std::string name = body.value("name", "");
        if (speaker.empty())
            return jr(Status::CODE_400, {{"error", "Missing speaker key."}});
        app_.store.set_speaker_name(sid, speaker, name);
        json names = name.empty() ? json(nullptr) : json{{speaker, name}};
        return jr(Status::CODE_200,
                  {{"key", speaker},
                   {"label", views::resolve_label(speaker, names)}});
    }

    ENDPOINT("POST", "/api/sessions/{id}/turns/{idx}/speaker", reassign_turn,
             PATH(String, id), PATH(String, idx),
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        std::string sid = id;
        if (app_.store.get(sid).is_null())
            return jr(Status::CODE_404, json::object());
        json body = http_util::parse_body(request);
        std::string speaker = body.value("speaker", "");
        std::string name = body.value("name", "");
        if (speaker.empty()) {
            if (name.empty())
                return jr(Status::CODE_400,
                          {{"error",
                            "Provide a speaker key or a name for a new speaker."}});
            json segs = current_segments(sid);
            json names = app_.store.get_speaker_names(sid);
            auto casefold = [](std::string s) {
                std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                return s;
            };
            std::set<std::string> taken;
            for (const auto& k : whisperx::edits::distinct_speakers(segs))
                taken.insert(casefold(views::resolve_label(k, names)));
            for (auto it = names.begin(); it != names.end(); ++it)
                if (it.value().is_string())
                    taken.insert(casefold(it.value().get<std::string>()));
            if (taken.count(casefold(name)))
                return jr(Status::CODE_409,
                          {{"error", "A speaker named '" + name +
                                         "' already exists."}});
            json existing = json::array();
            for (const auto& k : whisperx::edits::distinct_speakers(segs))
                existing.push_back(k);
            for (auto it = names.begin(); it != names.end(); ++it)
                existing.push_back(it.key());
            speaker = whisperx::edits::next_speaker_key(existing);
        }
        if (!name.empty()) app_.store.set_speaker_name(sid, speaker, name);
        try {
            app_.store.save_turn_reassign(sid, std::stol(idx), speaker);
        } catch (const std::exception&) {
            return jr(Status::CODE_400, {{"error", "Unknown turn."}});
        }
        return jr(Status::CODE_200, transcript_payload(sid));
    }

    // Reassign a *selection* inside a turn to another speaker (the edit-mode
    // 3-way split: head + tail keep the original speaker, the [start,end) middle
    // moves to `speaker`/`name`). Offsets are UTF-16 code units into the turn's
    // space-joined words. Speaker resolution mirrors reassign_turn (existing key,
    // or mint one from `name` after a case-insensitive duplicate check).
    ENDPOINT("POST", "/api/sessions/{id}/turns/{idx}/split", split_reassign_turn,
             PATH(String, id), PATH(String, idx),
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        std::string sid = id;
        if (app_.store.get(sid).is_null())
            return jr(Status::CODE_404, json::object());
        json body = http_util::parse_body(request);
        std::string speaker = body.value("speaker", "");
        std::string name = body.value("name", "");
        if (speaker.empty() && name.empty())
            return jr(Status::CODE_400,
                      {{"error",
                        "Provide a speaker key or a name for a new speaker."}});
        const long start = body.value("start", -1L);
        const long end = body.value("end", -1L);
        if (start < 0 || end <= start)
            return jr(Status::CODE_400, {{"error", "Empty or invalid selection."}});
        if (speaker.empty()) {
            json segs = current_segments(sid);
            json names = app_.store.get_speaker_names(sid);
            auto casefold = [](std::string s) {
                std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                return s;
            };
            std::set<std::string> taken;
            for (const auto& k : whisperx::edits::distinct_speakers(segs))
                taken.insert(casefold(views::resolve_label(k, names)));
            for (auto it = names.begin(); it != names.end(); ++it)
                if (it.value().is_string())
                    taken.insert(casefold(it.value().get<std::string>()));
            if (taken.count(casefold(name)))
                return jr(Status::CODE_409,
                          {{"error", "A speaker named '" + name +
                                         "' already exists."}});
            json existing = json::array();
            for (const auto& k : whisperx::edits::distinct_speakers(segs))
                existing.push_back(k);
            for (auto it = names.begin(); it != names.end(); ++it)
                existing.push_back(it.key());
            speaker = whisperx::edits::next_speaker_key(existing);
        }
        if (!name.empty()) app_.store.set_speaker_name(sid, speaker, name);
        try {
            app_.store.save_turn_split(sid, std::stol(idx), start, end, speaker);
        } catch (const std::exception&) {
            return jr(Status::CODE_400, {{"error", "Unknown turn."}});
        }
        return jr(Status::CODE_200, transcript_payload(sid));
    }

    // --- Models / device -------------------------------------------------
    ENDPOINT("GET", "/api/models", get_models) {
        return jr(Status::CODE_200, app_.manager.status());
    }

    ENDPOINT("POST", "/api/models/active", switch_model,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        json body = http_util::parse_body(request);
        std::string model = body.value("model", "");
        if (!models::is_known_model(model))
            return jr(Status::CODE_400, {{"error", "Unknown model: " + model}});
        json status = app_.manager.set_active(model);
        app_.store.set_setting("active_model", model);
        return jr(Status::CODE_200, status);
    }

    ENDPOINT("POST", "/api/device", switch_device,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        json body = http_util::parse_body(request);
        std::string device_str = body.value("device", "");
        auto dev = parse_device(device_str);
        if (!dev)
            return jr(Status::CODE_400,
                      {{"error", "Unknown device: " + device_str}});
        // Reject cuda when the binary wasn't built with the GPU ORT / no device is
        // present, rather than constructing a session that silently falls back to CPU.
        if (*dev == Device::Cuda &&
            !app_.manager.status().value("cuda_available", false)) {
            json st = app_.manager.status();
            st["error"] = "CUDA device not available in this build.";
            return jr(Status::CODE_400, st);
        }
        // UX gate: a running job blocks the switch (in-flight engine borrows are
        // shared_ptrs, so this is courtesy, not the memory-safety mechanism).
        if (app_.store.has_active_jobs()) {
            json st = app_.manager.status();
            st["error"] = "busy";
            return jr(Status::CODE_409, st);
        }
        try {
            json st = app_.manager.set_device(*dev);
            app_.store.set_setting("device", to_string(*dev));  // persist on success
            return jr(Status::CODE_200, st);
        } catch (const std::exception& exc) {
            // set_device rolled back to the previous device; report, don't crash
            // (an uncaught exception on the request thread is fatal — no signal handler).
            json st = app_.manager.status();
            st["error"] = exc.what();
            return jr(Status::CODE_400, st);
        }
    }

    // --- Settings / onboarding (token storage + translation deferred) ----
    json settings_payload() {
        return {
            {"default_language", app_.store.get_setting("default_language", "")
                                     .value_or("")},
            {"languages", transcribe_languages()},
            {"models", app_.manager.status()},
            {"translation_service",
             app_.store.get_setting("translation_service", "google")
                 .value_or("google")},
            {"translation_services", translation_services()},
            {"translation_languages", translation_languages()},
            {"google_key",
             {{"key_set",
               secrets::resolve_google_api_key().has_value()}}},
            {"diarize",
             {{"version", nullptr},
              {"model_name", "pyannote/speaker-diarization-community-1"},
              {"token_set", secrets::resolve_hf_token().has_value()}}},
            {"backup", app_.backup.status_json()},
            {"onboarded",
             app_.store.get_setting("onboarded", "").value_or("") == "1"},
        };
    }

    ENDPOINT("GET", "/api/settings", get_settings) {
        return jr(Status::CODE_200, settings_payload());
    }

    ENDPOINT("POST", "/api/settings", post_settings,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        json body = http_util::parse_body(request);
        std::string lang = body.value("default_language", "");
        app_.store.set_setting("default_language", lang);
        return jr(Status::CODE_200,
                  {{"ok", true}, {"default_language", lang}});
    }

    // Set the HF token: verify live, then store in the OS keyring. Shapes per
    // api-reference.md:392 — 200 ok, 400 failed verify, 500 keyring unavailable.
    ENDPOINT("POST", "/api/settings/hf-token", post_hf_token,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        json body = http_util::parse_body(request);
        std::string token = body.value("hf_token", "");
        auto [ok, detail] = secrets::verify_token(token);
        if (!ok)
            return jr(Status::CODE_400,
                      {{"token_set", secrets::resolve_hf_token().has_value()},
                       {"notice", detail},
                       {"notice_ok", false}});
        try {
            secrets::set(secrets::keys::HF_TOKEN, token);
        } catch (const secrets::SecretStoreUnavailable& e) {
            return jr(Status::CODE_500, {{"token_set", false},
                                         {"notice", std::string(e.what())},
                                         {"notice_ok", false}});
        }
        return jr(Status::CODE_200,
                  {{"token_set", true}, {"notice", detail}, {"notice_ok", true}});
    }

    ENDPOINT("POST", "/api/settings/hf-token/clear", clear_hf_token) {
        secrets::erase(secrets::keys::HF_TOKEN);
        return jr(Status::CODE_200,
                  {{"token_set", secrets::resolve_hf_token().has_value()},
                   {"notice", "Token cleared."},
                   {"notice_ok", true}});
    }

    ENDPOINT("GET", "/api/onboarding", get_onboarding) {
        json status = app_.manager.status();
        return jr(Status::CODE_200,
                  {{"token", ""},
                   {"sizes", onboarding_sizes()},
                   {"selected_size", status.value("active", "small")},
                   {"models", status},
                   {"diarize_model",
                    "pyannote/speaker-diarization-community-1"},
                   {"backup", app_.backup.status_json()}});
    }

    ENDPOINT("POST", "/api/onboarding/verify", onboarding_verify,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        json body = http_util::parse_body(request);
        auto [ok, detail] = secrets::verify_token(body.value("token", ""));
        return jr(Status::CODE_200, {{"ok", ok}, {"detail", detail}});
    }

    ENDPOINT("POST", "/api/onboarding", onboarding_finish,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        json body = http_util::parse_body(request);
        std::string model = body.value("model", "");
        std::string device = body.value("device", "");
        std::string token = body.value("token", "");
        if (!models::is_known_model(model))
            return jr(Status::CODE_400, {{"error", "Unknown model: " + model}});
        auto dev = parse_device(device);
        if (!dev)
            return jr(Status::CODE_400,
                      {{"error", "Unknown device: " + device}});
        if (*dev == Device::Cuda &&
            !app_.manager.status().value("cuda_available", false))
            return jr(Status::CODE_400,
                      {{"error", "CUDA device not available in this build."}});
        // A supplied token is verified then stored in the OS keyring. A keyring
        // failure is non-fatal — diarization works token-free — so we surface a
        // store_error but still finish onboarding (matches server.py's contract).
        if (!token.empty()) {
            auto [ok, detail] = secrets::verify_token(token);
            if (!ok) return jr(Status::CODE_400, {{"error", detail}});
            try {
                secrets::set(secrets::keys::HF_TOKEN, token);
            } catch (const secrets::SecretStoreUnavailable& e) {
                json out = finish_onboarding(model, *dev);
                out["store_error"] = e.what();
                return jr(Status::CODE_200, out);
            }
        }
        return jr(Status::CODE_200, finish_onboarding(model, *dev));
    }

    // Persist the chosen model/device + mark onboarded (shared by the keyring-ok
    // and keyring-unavailable paths). Returns the {ok:true} payload.
    json finish_onboarding(const std::string& model, Device device) {
        app_.store.set_setting("active_model", model);
        app_.store.set_setting("device", to_string(device));
        app_.manager.set_active(model);
        json out{{"ok", true}};
        try {
            app_.manager.set_device(device);  // no-op if already on it
        } catch (const std::exception& e) {
            // A GPU init failure during onboarding is non-fatal (mirrors the keyring
            // path): finish on the previous device and surface the error.
            out["device_error"] = e.what();
        }
        app_.store.set_setting("onboarded", "1");
        return out;
    }

    // --- SSE -------------------------------------------------------------
    ENDPOINT("GET", "/sessions/{id}/events", session_events, PATH(String, id)) {
        std::string sid = id;
        if (app_.store.get(sid).is_null())
            return jr(Status::CODE_404, json::object());
        AppState* app = &app_;
        auto initial = [app, sid]() -> std::optional<json> {
            json row = app->store.get(sid);
            if (row.is_null()) return json{{"status", "queued"}};
            std::string status = row.value("status", "");
            if (status == "done" || status == "error")
                return json{{"status", status}};
            if (row.contains("stage") && row["stage"].is_string()) {
                json ev = {{"stage", row["stage"]}};
                return ev;
            }
            return json{{"status", status.empty() ? "queued" : status}};
        };
        auto terminal = [](const json& e) {
            return e.contains("status") &&
                   (e["status"] == "done" || e["status"] == "error");
        };
        return sse::sse_response(app_.broker, sid, initial, terminal);
    }

    ENDPOINT("GET", "/models/events", models_events) {
        AppState* app = &app_;
        auto initial = [app]() -> std::optional<json> {
            return views::models_event(app->manager.status());
        };
        return sse::sse_response(app_.broker, kModelsChannel, initial);
    }

    // --- Binary downloads ------------------------------------------------
    ENDPOINT("GET", "/sessions/{id}/audio", session_audio, PATH(String, id)) {
        std::string sid = id;
        json row = app_.store.get(sid);
        if (row.is_null()) return jr(Status::CODE_404, json::object());
        std::string fn = row.value("audio_filename", "");
        if (fn.empty()) return jr(Status::CODE_404, json::object());
        std::string path =
            (fs::path(app_.cfg.data_dir) / "sessions" / sid / fn).string();
        return http_util::file_response(path, "application/octet-stream", false);
    }

    ENDPOINT("GET", "/sessions/{id}/download/{fmt}", download, PATH(String, id),
             PATH(String, fmt)) {
        std::string f = fmt;
        if (f != "srt" && f != "vtt" && f != "txt" && f != "json")
            return jr(Status::CODE_404, json::object());
        std::string path = (fs::path(app_.cfg.data_dir) / "sessions" /
                            std::string(id) / ("transcript." + f))
                               .string();
        return http_util::file_response(path, "application/octet-stream", true);
    }

    ENDPOINT("GET", "/sessions/{id}/export.md", export_md, PATH(String, id)) {
        std::string sid = id;
        json row = app_.store.get(sid);
        if (row.is_null()) return jr(Status::CODE_404, json::object());
        json result = app_.store.load_result(sid);
        if (!result.is_object()) result = json::object();
        json orig = result.contains("segments") ? result["segments"]
                                                 : json::array();
        json segs = app_.store.current_segments(sid, orig);
        result["segments"] = segs;
        std::string title = row.value("filename", "Transcript");
        std::string md = views::render_markdown(
            result, app_.store.get_speaker_names(sid), title);
        std::string fname = http_util::secure_filename(title);
        if (fname.empty()) fname = "transcript";
        auto resp = createResponse(Status::CODE_200, oatpp::String(md));
        resp->putHeader("Content-Type", "text/markdown");
        resp->putHeader("Content-Disposition",
                        "attachment; filename=\"" + fname + ".md\"");
        return resp;
    }

    // --- Cloud backup: Google Drive OAuth link (token lifecycle only) -----
    ENDPOINT("GET", "/api/backup/status", backup_status) {
        return jr(Status::CODE_200, app_.backup.status_json());
    }

    ENDPOINT("POST", "/api/backup/connect", backup_connect,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        if (app_.backup.is_linked())
            return jr(Status::CODE_200, app_.backup.status_json());
        if (!app_.backup.configured())
            return jr(Status::CODE_400,
                      {{"error",
                        "Cloud backup isn't configured on this server "
                        "(GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET unset)."}});
        json body = http_util::parse_body(request);
        std::string folder = body.value("backup_folder", "");
        if (!folder.empty()) app_.backup.set_folder(folder);
        if (!app_.backup.connect())
            return jr(Status::CODE_409,
                      {{"error", "A link is already in progress."}});
        return jr(Status::CODE_200, {{"connecting", true}});
    }

    // Trigger a backup now (async); the persistent status stream carries progress.
    ENDPOINT("POST", "/api/backup/now", backup_now) {
        if (!app_.backup.start_backup_now())
            return jr(Status::CODE_409,
                      {{"error", "Backup backend is not linked."}});
        return jr(Status::CODE_202, app_.backup.status_json());
    }

    // Pull the remote mirror down to local (maintenance op).
    ENDPOINT("POST", "/api/backup/restore", backup_restore) {
        try {
            int n = app_.backup.restore();
            return jr(Status::CODE_200,
                      {{"restored", n}, {"backup", app_.backup.status_json()}});
        } catch (const std::exception& e) {
            return jr(Status::CODE_409, {{"error", e.what()}});
        }
    }

    // Bootstrap conflict resolution: adopt the existing remote (restore down).
    ENDPOINT("POST", "/api/backup/bootstrap/adopt", backup_adopt) {
        try {
            int n = app_.backup.adopt();
            return jr(Status::CODE_200,
                      {{"restored", n}, {"backup", app_.backup.status_json()}});
        } catch (const std::exception& e) {
            return jr(Status::CODE_409, {{"error", e.what()}});
        }
    }

    // Bootstrap conflict resolution: overwrite the remote with local (GC old).
    ENDPOINT("POST", "/api/backup/bootstrap/overwrite", backup_overwrite) {
        try {
            json r = app_.backup.overwrite();
            r["backup"] = app_.backup.status_json();
            return jr(Status::CODE_200, r);
        } catch (const std::exception& e) {
            return jr(Status::CODE_409, {{"error", e.what()}});
        }
    }

    // Inspect the remote (size/generation) for the restore/conflict prompts.
    ENDPOINT("GET", "/api/backup/remote-info", backup_remote_info) {
        try {
            return jr(Status::CODE_200, app_.backup.remote_info());
        } catch (const std::exception& e) {
            return jr(Status::CODE_409, {{"error", e.what()}});
        }
    }

    ENDPOINT("POST", "/api/backup/disconnect", backup_disconnect) {
        app_.backup.disconnect();
        return jr(Status::CODE_200, app_.backup.status_json());
    }

    // Loopback OAuth redirect target — fulfils the pending consent flow and
    // serves a human-facing success/error page. Reserved root "oauth" keeps the
    // SPA catch-all from shadowing this.
    ENDPOINT("GET", "/oauth/callback", oauth_callback,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        auto qp = [&](const char* k) -> std::string {
            auto v = request->getQueryParameter(k);
            return v ? std::string(v) : std::string();
        };
        std::string error = qp("error");
        app_.backup.handle_callback(qp("code"), qp("state"), error);
        const char* html = error.empty() ? oauth::LinkFlow::success_html()
                                          : oauth::LinkFlow::error_html();
        auto resp = createResponse(Status::CODE_200, oatpp::String(html));
        resp->putHeader("Content-Type", "text/html; charset=utf-8");
        return resp;
    }

    // One-shot OAuth consent result stream. Payload envelope is
    // {status: connecting|linked|error, backup: <card>, message?}; terminal once
    // status leaves "connecting" (mirrors app/server.py /backup/events).
    ENDPOINT("GET", "/backup/events", backup_events) {
        AppState* app = &app_;
        auto initial = [app]() -> std::optional<json> {
            json card = app->backup.status_json();
            std::string st =
                card.value("linked", false)
                    ? "linked"
                    : (card.value("state", "") == "connecting" ? "connecting"
                                                               : "idle");
            return json{{"status", st}, {"backup", card}};
        };
        auto terminal = [](const json& e) {
            return e.contains("status") &&
                   (e["status"] == "linked" || e["status"] == "error");
        };
        return sse::sse_response(app_.broker, kBackupChannel, initial, terminal);
    }

    // Persistent backup sync-status stream. Payload is {status: <card>} on every
    // engine state transition (mirrors /backup/status/events).
    ENDPOINT("GET", "/backup/status/events", backup_status_events) {
        AppState* app = &app_;
        auto initial = [app]() -> std::optional<json> {
            return json{{"status", app->backup.status_json()}};
        };
        return sse::sse_response(app_.broker, kBackupStatusChannel, initial);
    }

    // --- Translation -----------------------------------------------------
    ENDPOINT("POST", "/api/sessions/{id}/translate", translate_session,
             PATH(String, id),
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        std::string sid = id;
        json row = app_.store.get(sid);
        if (row.is_null()) return jr(Status::CODE_404, json::object());
        if (row.value("status", "") != "done")
            return jr(Status::CODE_409,
                      {{"error", "Transcript is not ready to translate."}});
        json body = http_util::parse_body(request);
        std::string target = trim(body.value("target_language", ""));
        if (!valid_lang(target))
            return jr(Status::CODE_400, {{"error", "Invalid target language."}});
        if (!secrets::resolve_google_api_key().has_value())
            return jr(Status::CODE_400,
                      {{"error",
                        "Add a Google Translation API key in Settings first."}});
        std::string service =
            app_.store.get_setting("translation_service", "google")
                .value_or("google");
        if (service != "google") service = "google";
        app_.translate_queue.submit(sid, target, service);
        return jr(Status::CODE_200,
                  {{"lang", target}, {"status", "running"}, {"service", service}});
    }

    ENDPOINT("GET", "/api/sessions/{id}/translation/{lang}", view_translation,
             PATH(String, id), PATH(String, lang)) {
        std::string sid = id, l = lang;
        if (app_.store.get(sid).is_null() || !valid_lang(l))
            return jr(Status::CODE_404, json::object());
        json overlay = app_.store.load_translation(sid, l);
        if (overlay.is_null()) return jr(Status::CODE_404, json::object());
        json segs = translate::apply_overlay(current_segments(sid), overlay);
        json names = app_.store.get_speaker_names(sid);
        return jr(Status::CODE_200,
                  {{"target_language", l},
                   {"turns", views::build_turns(segs, names)},
                   {"segments", segs}});
    }

    // Live per-language translation progress. Emits the durable status map on
    // connect, then deltas; closes on a terminal done/error (translate_job.py).
    ENDPOINT("GET", "/sessions/{id}/translate/events", translate_events,
             PATH(String, id)) {
        std::string sid = id;
        if (app_.store.get(sid).is_null())
            return jr(Status::CODE_404, json::object());
        AppState* app = &app_;
        auto initial = [app, sid]() -> std::optional<json> {
            return json{{"translations", app->store.get_translations(sid)}};
        };
        auto terminal = [](const json& e) {
            return e.contains("status") &&
                   (e["status"] == "done" || e["status"] == "error");
        };
        return sse::sse_response(app_.broker, translate::channel(sid), initial,
                                 terminal);
    }

    // Generate the translation export on demand from the joined segments, so it
    // reflects current speakers + the original-text fallback for edited segments.
    ENDPOINT("GET", "/sessions/{id}/translation/{lang}/download/{fmt}",
             translation_download, PATH(String, id), PATH(String, lang),
             PATH(String, fmt)) {
        std::string sid = id, l = lang, f = fmt;
        if ((f != "srt" && f != "vtt" && f != "txt" && f != "json") ||
            !valid_lang(l))
            return jr(Status::CODE_404, json::object());
        json overlay = app_.store.load_translation(sid, l);
        if (overlay.is_null()) return jr(Status::CODE_404, json::object());
        json segs = translate::apply_overlay(current_segments(sid), overlay);

        std::string body;
        if (f == "json") {
            body = json{{"target_language", l}, {"segments", segs}}.dump();
        } else {
            const json wopts = {{"max_line_width", nullptr},
                                {"max_line_count", nullptr},
                                {"highlight_words", false}};
            json result = {{"segments", segs}, {"language", l}};
            if (f == "srt")
                body = whisperx::writers::write_srt(result, wopts);
            else if (f == "vtt")
                body = whisperx::writers::write_vtt(result, wopts);
            else
                body = whisperx::writers::write_txt(result, wopts);
        }
        std::string name = "transcript.translation." + l + "." + f;
        auto resp = createResponse(Status::CODE_200, oatpp::String(body));
        resp->putHeader("Content-Type", "application/octet-stream");
        resp->putHeader("Content-Disposition",
                        "attachment; filename=\"" + name + "\"");
        return resp;
    }

    // --- Translation settings (key storage + service preference) ---------
    ENDPOINT("POST", "/api/settings/google-key", post_google_key,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        json body = http_util::parse_body(request);
        std::string key = trim(body.value("google_key", ""));
        auto [ok, detail] = translate::verify_api_key(key);
        if (!ok) return jr(Status::CODE_400, google_key_payload(detail, false));
        try {
            secrets::set(secrets::keys::GOOGLE_TRANSLATE, key);
        } catch (const secrets::SecretStoreUnavailable& e) {
            return jr(Status::CODE_500, google_key_payload(e.what(), false));
        }
        return jr(Status::CODE_200,
                  google_key_payload("Key saved and verified.", true));
    }

    ENDPOINT("POST", "/api/settings/google-key/clear", clear_google_key) {
        secrets::erase(secrets::keys::GOOGLE_TRANSLATE);
        bool set = secrets::resolve_google_api_key().has_value();
        std::string notice =
            set ? "Cleared the stored key, but GOOGLE_TRANSLATE_API_KEY is still "
                  "set in the environment."
                : "Key cleared. Translation is now disabled.";
        return jr(Status::CODE_200, google_key_payload(notice, true));
    }

    ENDPOINT("POST", "/api/settings/translation-service", set_translation_service,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        json body = http_util::parse_body(request);
        std::string service = trim(body.value("translation_service", ""));
        if (service != "google")  // only "google" is registered
            return jr(Status::CODE_400,
                      {{"error", "Unknown translation service."}});
        app_.store.set_setting("translation_service", service);
        return jr(Status::CODE_200, {{"ok", true}});
    }

private:
    // server.py::_google_key_payload — current key_set + a user-facing notice.
    json google_key_payload(const std::string& notice, bool notice_ok) {
        return {{"key_set", secrets::resolve_google_api_key().has_value()},
                {"notice", notice},
                {"notice_ok", notice_ok}};
    }

    // server.py SERVICES registry — only Google is wired (the v2 REST backend).
    static const json& translation_services() {
        static const json v =
            json::array({{{"id", "google"}, {"label", "Google Translate"}}});
        return v;
    }

    // server.py TRANSLATION_LANGUAGES — the curated target-language picker.
    static const json& translation_languages() {
        static const json v = json::array(
            {{{"code", "en"}, {"name", "English"}, {"native", "English"}},
             {{"code", "es"}, {"name", "Spanish"}, {"native", "Español"}},
             {{"code", "fr"}, {"name", "French"}, {"native", "Français"}},
             {{"code", "de"}, {"name", "German"}, {"native", "Deutsch"}},
             {{"code", "it"}, {"name", "Italian"}, {"native", "Italiano"}},
             {{"code", "pt"}, {"name", "Portuguese"}, {"native", "Português"}},
             {{"code", "pt-BR"}, {"name", "Portuguese (Brazil)"},
              {"native", "Português (BR)"}},
             {{"code", "nl"}, {"name", "Dutch"}, {"native", "Nederlands"}},
             {{"code", "ru"}, {"name", "Russian"}, {"native", "Русский"}},
             {{"code", "ja"}, {"name", "Japanese"}, {"native", "日本語"}},
             {{"code", "ko"}, {"name", "Korean"}, {"native", "한국어"}},
             {{"code", "zh"}, {"name", "Chinese"}, {"native", "中文"}},
             {{"code", "ar"}, {"name", "Arabic"}, {"native", "العربية"}},
             {{"code", "hi"}, {"name", "Hindi"}, {"native", "हिन्दी"}}});
        return v;
    }

    // Constant tables (server.py) as lazy statics.
    static const json& transcribe_languages() {
        static const json v = json::array(
            {{{"code", ""}, {"label", "Auto-detect"}},
             {{"code", "en"}, {"label", "English"}},
             {{"code", "es"}, {"label", "Spanish"}},
             {{"code", "fr"}, {"label", "French"}},
             {{"code", "de"}, {"label", "German"}},
             {{"code", "it"}, {"label", "Italian"}},
             {{"code", "pt"}, {"label", "Portuguese"}},
             {{"code", "nl"}, {"label", "Dutch"}},
             {{"code", "ja"}, {"label", "Japanese"}},
             {{"code", "zh"}, {"label", "Chinese"}},
             {{"code", "ru"}, {"label", "Russian"}}});
        return v;
    }
    static const json& onboarding_sizes() {
        static const json v = json::array(
            {{{"id", "tiny"}, {"name", "Tiny"}, {"meta", "39M · 1GB"},
              {"note", "<b>Fastest, lowest accuracy.</b>"}},
             {{"id", "base"}, {"name", "Base"}, {"meta", "74M · 1GB"},
              {"note", "<b>Fast with decent accuracy.</b>"}},
             {{"id", "small"}, {"name", "Small"}, {"meta", "244M · 2GB"},
              {"note", "<b>Balanced speed and accuracy.</b>"}},
             {{"id", "medium"}, {"name", "Medium"}, {"meta", "769M · 5GB"},
              {"note", "<b>Strong accuracy, slower.</b>"}},
             {{"id", "large-v3"}, {"name", "Large-v3"}, {"meta", "1.5B · 10GB"},
              {"note", "<b>Best accuracy, multilingual.</b>"}},
             {{"id", "large-v3-turbo"}, {"name", "Large Turbo"},
              {"meta", "809M · 6GB"},
              {"note", "<b>Near-large accuracy, much faster.</b>"}}});
        return v;
    }
};

#include OATPP_CODEGEN_END(ApiController)

}  // namespace whisperx::server
