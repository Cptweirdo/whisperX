#include "http/http_util.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;
namespace http = oatpp::web::protocol::http;

namespace whisperx::server::http_util {

namespace {

std::string url_decode(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '+') {
            out += ' ';
        } else if (s[i] == '%' && i + 2 < s.size()) {
            auto hex = s.substr(i + 1, 2);
            out += static_cast<char>(std::strtol(hex.c_str(), nullptr, 16));
            i += 2;
        } else {
            out += s[i];
        }
    }
    return out;
}

// Streams a file as a chunked download body via a blocking ReadCallback.
class FileReadCallback : public oatpp::data::stream::ReadCallback {
public:
    explicit FileReadCallback(const std::string& path)
        : in_(path, std::ios::binary) {}
    oatpp::v_io_size read(void* buffer, v_buff_size count,
                          oatpp::async::Action&) override {
        if (!in_.good()) return 0;
        in_.read(static_cast<char*>(buffer), count);
        auto n = in_.gcount();
        return n > 0 ? static_cast<oatpp::v_io_size>(n) : 0;
    }

private:
    std::ifstream in_;
};

}  // namespace

json parse_body(const std::shared_ptr<http::incoming::Request>& request) {
    oatpp::String raw = request->readBodyToString();
    if (!raw) return json::object();
    std::string body = *raw;
    auto b = body.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return json::object();

    if (body[b] == '{' || body[b] == '[') {
        try {
            json j = json::parse(body);
            if (j.is_object()) return j;
        } catch (...) {
            // fall through to form parsing
        }
    }

    // application/x-www-form-urlencoded: k=v&k=v
    json obj = json::object();
    std::size_t pos = 0;
    while (pos < body.size()) {
        auto amp = body.find('&', pos);
        std::string pair = body.substr(pos, amp - pos);
        auto eq = pair.find('=');
        if (eq != std::string::npos) {
            std::string k = url_decode(pair.substr(0, eq));
            std::string v = url_decode(pair.substr(eq + 1));
            if (!k.empty()) obj[k] = v;
        }
        if (amp == std::string::npos) break;
        pos = amp + 1;
    }
    return obj;
}

std::shared_ptr<http::outgoing::Response> file_response(
    const std::string& path, const std::string& content_type, bool attachment,
    const std::string& download_name) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec))
        return http::outgoing::ResponseFactory::createResponse(
            http::Status::CODE_404, "");
    auto cb = std::make_shared<FileReadCallback>(path);
    auto body = std::make_shared<http::outgoing::StreamingBody>(cb);
    auto resp = http::outgoing::Response::createShared(http::Status::CODE_200,
                                                       body);
    resp->putHeader("Content-Type", content_type);
    if (attachment) {
        std::string name =
            download_name.empty() ? fs::path(path).filename().string()
                                  : download_name;
        resp->putHeader("Content-Disposition",
                        "attachment; filename=\"" + name + "\"");
    }
    return resp;
}

std::string secure_filename(const std::string& name) {
    // keep only the basename (strip any path components)
    std::string base = fs::path(name).filename().string();
    std::string out;
    out.reserve(base.size());
    for (char c : base) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '.' ||
            c == '_' || c == '-')
            out += c;
        else
            out += '_';
    }
    // strip leading dots/dashes/underscores
    auto start = out.find_first_not_of("._-");
    return start == std::string::npos ? "" : out.substr(start);
}

}  // namespace whisperx::server::http_util
