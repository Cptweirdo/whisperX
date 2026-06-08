// SPA serving — the C++ port of server.py::spa() + Flask's /static handler.
// Serves the built Svelte SPA (app/static/spa) for any non-reserved path so
// client-side routing + deep links work, and serves the bundled assets under
// /static/*. Registered LAST so every concrete /api, /sessions/<id>/<sub>,
// /healthz, /models route wins (the oat++ router matches in insertion order).
#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "oatpp/core/macro/codegen.hpp"
#include "oatpp/web/server/api/ApiController.hpp"

#include "http/app_state.hpp"
#include "http/http_util.hpp"

namespace whisperx::server {

namespace fs = std::filesystem;

#include OATPP_CODEGEN_BEGIN(ApiController)

class SpaController : public oatpp::web::server::api::ApiController {
public:
    SpaController(AppState& app, const std::shared_ptr<ObjectMapper>& om)
        : oatpp::web::server::api::ApiController(om), app_(app) {}

private:
    AppState& app_;
    using Resp = std::shared_ptr<OutgoingResponse>;

    static std::string content_type(const std::string& path) {
        auto ext = fs::path(path).extension().string();
        if (ext == ".html") return "text/html; charset=utf-8";
        if (ext == ".js" || ext == ".mjs") return "text/javascript";
        if (ext == ".css") return "text/css";
        if (ext == ".json") return "application/json";
        if (ext == ".svg") return "image/svg+xml";
        if (ext == ".png") return "image/png";
        if (ext == ".woff2") return "font/woff2";
        if (ext == ".woff") return "font/woff";
        if (ext == ".ico") return "image/x-icon";
        return "application/octet-stream";
    }

    std::string request_path(const std::shared_ptr<IncomingRequest>& req) {
        auto p = req->getStartingLine().path;
        return p.getData() ? p.std_str() : "/";
    }

public:
    // Bundled SPA assets (prod base is /static/spa/). The static_dir is the
    // parent of the built SPA (app/static), so /static/spa/... resolves under it.
    ENDPOINT("GET", "/static/*", static_asset,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        std::string path = request_path(request);
        // strip leading "/static/"
        std::string rel = path.substr(std::string("/static/").size());
        if (rel.find("..") != std::string::npos)
            return createResponse(Status::CODE_404, "");
        std::string file = (fs::path(app_.cfg.static_dir) / rel).string();
        return http_util::file_response(file, content_type(file), false);
    }

    ENDPOINT("GET", "*", spa, REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        std::string path = request_path(request);
        std::string spa_path = path.empty() || path[0] != '/' ? path
                                                              : path.substr(1);
        std::string head = spa_path.substr(0, spa_path.find('/'));
        static const std::set<std::string> reserved = {
            "api", "healthz", "backup", "models", "static", "oauth"};
        if (reserved.count(head)) return createResponse(Status::CODE_404, "");

        std::string index =
            (fs::path(app_.cfg.spa_dir) / "index.html").string();
        std::error_code ec;
        if (!fs::exists(index, ec))
            return createResponse(
                Status::CODE_500,
                "SPA not built — run: cd app/web && bun run build");
        // A path with a file extension that no asset route served is a real miss.
        if (head.find('.') != std::string::npos &&
            !fs::exists(fs::path(app_.cfg.spa_dir) / spa_path, ec))
            return createResponse(Status::CODE_404, "");
        return http_util::file_response(index, "text/html; charset=utf-8",
                                        false);
    }
};

#include OATPP_CODEGEN_END(ApiController)

}  // namespace whisperx::server
