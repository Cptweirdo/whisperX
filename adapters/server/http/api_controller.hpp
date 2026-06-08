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
#include "secrets/hf_verify.hpp"
#include "secrets/keyring.hpp"
#include "sse/broker.hpp"
#include "sse/sse_response.hpp"

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
        std::string device = body.value("device", "");
        // CPU-only runtime: cpu is accepted (no-op); anything else is rejected.
        if (device != "cpu") {
            if (device.empty())
                return jr(Status::CODE_400, {{"error", "Unknown device: "}});
            json st = app_.manager.status();
            st["error"] =
                "Only the CPU device is available in this build (" + device +
                " requires a Python/GPU runtime).";
            return jr(Status::CODE_400, st);
        }
        if (app_.store.has_active_jobs()) {
            json st = app_.manager.status();
            st["error"] = "busy";
            return jr(Status::CODE_409, st);
        }
        app_.store.set_setting("device", "cpu");
        return jr(Status::CODE_200, app_.manager.status());
    }

    // --- Settings / onboarding (token storage + translation deferred) ----
    json settings_payload() {
        return {
            {"default_language", app_.store.get_setting("default_language", "")
                                     .value_or("")},
            {"languages", transcribe_languages()},
            {"models", app_.manager.status()},
            {"translation_service", "google"},
            {"translation_services", json::array()},  // deferred
            {"translation_languages", json::array()},
            {"google_key",
             {{"key_set",
               secrets::resolve_google_api_key().has_value()}}},
            {"diarize",
             {{"version", nullptr},
              {"model_name", "pyannote/speaker-diarization-community-1"},
              {"token_set", secrets::resolve_hf_token().has_value()}}},
            {"backup", backup_stub()},
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
                   {"backup", backup_stub()}});
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
        if (device != "cpu")
            return jr(Status::CODE_400,
                      {{"error", "Unknown device: " + device}});
        // A supplied token is verified then stored in the OS keyring. A keyring
        // failure is non-fatal — diarization works token-free — so we surface a
        // store_error but still finish onboarding (matches server.py's contract).
        if (!token.empty()) {
            auto [ok, detail] = secrets::verify_token(token);
            if (!ok) return jr(Status::CODE_400, {{"error", detail}});
            try {
                secrets::set(secrets::keys::HF_TOKEN, token);
            } catch (const secrets::SecretStoreUnavailable& e) {
                json out = finish_onboarding(model);
                out["store_error"] = e.what();
                return jr(Status::CODE_200, out);
            }
        }
        return jr(Status::CODE_200, finish_onboarding(model));
    }

    // Persist the chosen model/device + mark onboarded (shared by the keyring-ok
    // and keyring-unavailable paths). Returns the {ok:true} payload.
    json finish_onboarding(const std::string& model) {
        app_.store.set_setting("active_model", model);
        app_.store.set_setting("device", "cpu");
        app_.manager.set_active(model);
        app_.store.set_setting("onboarded", "1");
        return json{{"ok", true}};
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

    // --- Deferred features (translation + backup): SPA-friendly shapes ----
    json backup_stub() {
        return {{"state", "idle"},      {"linked", false},
                {"backend", nullptr},   {"dirty", nullptr},
                {"last_root", nullptr}, {"last_backup_at", nullptr},
                {"last_error", nullptr},{"interval", nullptr},
                {"provider_label", "Cloud backup"},
                {"last_human", nullptr},{"folder", nullptr},
                {"remote", nullptr}};
    }

    ENDPOINT("GET", "/api/backup/status", backup_status) {
        return jr(Status::CODE_200, backup_stub());
    }

    ENDPOINT("POST", "/api/sessions/{id}/translate", translate_session,
             PATH(String, id)) {
        return jr(Status::CODE_400,
                  {{"error", "Translation is not available in this build."}});
    }

private:
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
