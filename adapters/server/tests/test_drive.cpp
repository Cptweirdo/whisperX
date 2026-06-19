// Catch2 tests for the Drive v3 files.* client — no network. A fake transport
// records the (method, url, headers, body) it was handed and returns canned
// responses, so we can pin URL/query shaping, multipart bodies, pagination,
// the `q` escaping, and error mapping.
#include <catch2/catch_test_macros.hpp>

#include <optional>
#include <string>
#include <vector>

#include "drive/drive_client.hpp"
#include "http/curl_client.hpp"

using namespace whisperx::server::drive;
using whisperx::server::net::Response;

namespace {

struct Call {
    std::string method, url, body;
    std::vector<std::string> headers;
};

// A scripted transport: each request pops the next canned Response and logs the
// call. Auth always returns a token.
struct Fake {
    std::vector<Call> calls;
    std::vector<Response> script;
    size_t i = 0;

    DriveClient::Transport transport() {
        return [this](const std::string& m, const std::string& u,
                      const std::vector<std::string>& h, const std::string& b) {
            calls.push_back({m, u, b, h});
            if (i < script.size()) return script[i++];
            Response r;
            r.status = 500;
            return r;
        };
    }
};

DriveClient make(Fake& fake) {
    DriveClient c([] { return std::optional<std::string>("Bearer TOK"); },
                  fake.transport());
    c.set_endpoints("https://api/drive/v3", "https://api/upload/drive/v3");
    return c;
}

Response ok(const std::string& body) {
    Response r;
    r.status = 200;
    r.body = body;
    return r;
}

bool has_header(const Call& c, const std::string& h) {
    for (const auto& x : c.headers)
        if (x == h) return true;
    return false;
}

}  // namespace

TEST_CASE("list builds a q/fields query and parses files + token",
          "[drive]") {
    Fake fake;
    fake.script.push_back(ok(
        R"({"files":[{"id":"f1","name":"a"},{"id":"f2","name":"b"}],)"
        R"("nextPageToken":"NT"})"));
    DriveClient c = make(fake);

    auto page = c.list("'root' in parents and trashed = false",
                       "nextPageToken,files(id,name)", 1000);
    REQUIRE(page.files.size() == 2);
    REQUIRE(page.files[0].id == "f1");
    REQUIRE(page.next_page_token == "NT");

    const Call& call = fake.calls.at(0);
    REQUIRE(call.method == "GET");
    REQUIRE(call.url.find("https://api/drive/v3/files?") == 0);
    REQUIRE(call.url.find("spaces=drive") != std::string::npos);
    REQUIRE(call.url.find("pageSize=1000") != std::string::npos);
    // q is percent-encoded (spaces -> %20, quotes -> %27)
    REQUIRE(call.url.find("q=%27root%27%20in%20parents") != std::string::npos);
    REQUIRE(has_header(call, "Authorization: Bearer TOK"));
}

TEST_CASE("get metadata hits /files/{id} with fields", "[drive]") {
    Fake fake;
    fake.script.push_back(ok(R"({"id":"x","name":"n","mimeType":"text/plain"})"));
    DriveClient c = make(fake);

    File f = c.get("x", "id,name,mimeType");
    REQUIRE(f.id == "x");
    REQUIRE(f.mime_type == "text/plain");
    REQUIRE(fake.calls.at(0).url ==
            "https://api/drive/v3/files/x?fields=id%2Cname%2CmimeType");
}

TEST_CASE("download uses alt=media and returns raw bytes", "[drive]") {
    Fake fake;
    Response r;
    r.status = 200;
    r.body = "RAWBYTES";
    fake.script.push_back(r);
    DriveClient c = make(fake);

    REQUIRE(c.download("fid") == "RAWBYTES");
    REQUIRE(fake.calls.at(0).url ==
            "https://api/drive/v3/files/fid?alt=media");
}

TEST_CASE("create_folder posts folder metadata to /files", "[drive]") {
    Fake fake;
    fake.script.push_back(ok(R"({"id":"folder1"})"));
    DriveClient c = make(fake);

    REQUIRE(c.create_folder("Backup", "parentX") == "folder1");
    const Call& call = fake.calls.at(0);
    REQUIRE(call.method == "POST");
    REQUIRE(call.url == "https://api/drive/v3/files?fields=id");
    REQUIRE(has_header(call, "Content-Type: application/json"));
    json sent = json::parse(call.body);
    REQUIRE(sent["name"] == "Backup");
    REQUIRE(sent["mimeType"] == "application/vnd.google-apps.folder");
    REQUIRE(sent["parents"] == json::array({"parentX"}));
}

TEST_CASE("create_file builds a multipart/related upload", "[drive]") {
    Fake fake;
    fake.script.push_back(ok(R"({"id":"obj1"})"));
    DriveClient c = make(fake);

    json meta = {{"name", "manifest.json"}, {"parents", {"root"}}};
    File f = c.create_file(meta, "application/json", "{\"v\":1}");
    REQUIRE(f.id == "obj1");

    const Call& call = fake.calls.at(0);
    REQUIRE(call.method == "POST");
    REQUIRE(call.url ==
            "https://api/upload/drive/v3/files?uploadType=multipart&fields=id");
    // boundary header matches the body delimiter
    std::string ct;
    for (const auto& h : call.headers)
        if (h.rfind("Content-Type: multipart/related; boundary=", 0) == 0)
            ct = h;
    REQUIRE(!ct.empty());
    std::string boundary = ct.substr(ct.find("boundary=") + 9);
    REQUIRE(call.body.find("--" + boundary) != std::string::npos);
    REQUIRE(call.body.find("Content-Type: application/json; charset=UTF-8") !=
            std::string::npos);
    REQUIRE(call.body.find("{\"v\":1}") != std::string::npos);
    REQUIRE(call.body.find("--" + boundary + "--") != std::string::npos);
}

TEST_CASE("update_content PATCHes uploadType=media with the raw body",
          "[drive]") {
    Fake fake;
    fake.script.push_back(ok(R"({"id":"m1"})"));
    DriveClient c = make(fake);

    c.update_content("m1", "application/json", "NEWCONTENT");
    const Call& call = fake.calls.at(0);
    REQUIRE(call.method == "PATCH");
    REQUIRE(call.url ==
            "https://api/upload/drive/v3/files/m1?uploadType=media&fields=id");
    REQUIRE(has_header(call, "Content-Type: application/json"));
    REQUIRE(call.body == "NEWCONTENT");
}

TEST_CASE("remove issues a DELETE and tolerates an empty 204", "[drive]") {
    Fake fake;
    Response r;
    r.status = 204;  // no body
    fake.script.push_back(r);
    DriveClient c = make(fake);

    c.remove("gone");
    REQUIRE(fake.calls.at(0).method == "DELETE");
    REQUIRE(fake.calls.at(0).url == "https://api/drive/v3/files/gone");
}

TEST_CASE("find_child escapes the name inside the q literal", "[drive]") {
    Fake fake;
    fake.script.push_back(ok(R"({"files":[{"id":"hit"}]})"));
    DriveClient c = make(fake);

    auto id = c.find_child("o'brien\\x", "par", false);
    REQUIRE(id.has_value());
    REQUIRE(*id == "hit");
    // backslash and quote are backslash-escaped in the q literal, then percent
    // encoded for the URL: \' -> %5C%27 , \\ -> %5C%5C
    REQUIRE(fake.calls.at(0).url.find("o%27brien") == std::string::npos);
    REQUIRE(fake.calls.at(0).url.find("o%5C%27brien%5C%5Cx") !=
            std::string::npos);
}

TEST_CASE("ensure_folder reuses an existing folder before creating", "[drive]") {
    Fake fake;
    fake.script.push_back(ok(R"({"files":[{"id":"existing"}]})"));
    DriveClient c = make(fake);

    REQUIRE(c.ensure_folder("objects", "root") == "existing");
    REQUIRE(fake.calls.size() == 1);  // only the list, no create
}

TEST_CASE("ensure_folder creates when none exists", "[drive]") {
    Fake fake;
    fake.script.push_back(ok(R"({"files":[]})"));   // find_child -> miss
    fake.script.push_back(ok(R"({"id":"new"})"));   // create_folder
    DriveClient c = make(fake);

    REQUIRE(c.ensure_folder("objects", "root") == "new");
    REQUIRE(fake.calls.size() == 2);
    REQUIRE(fake.calls.at(1).method == "POST");
}

TEST_CASE("non-2xx maps to DriveError with the status", "[drive]") {
    Fake fake;
    Response r;
    r.status = 403;
    r.body = R"({"error":{"message":"insufficient"}})";
    fake.script.push_back(r);
    DriveClient c = make(fake);

    try {
        c.get("x");
        FAIL("expected DriveError");
    } catch (const DriveError& e) {
        REQUIRE(e.status == 403);
    }
}

TEST_CASE("missing token throws before any request", "[drive]") {
    Fake fake;
    DriveClient c([] { return std::optional<std::string>(); }, fake.transport());
    try {
        c.get("x");
        FAIL("expected DriveError");
    } catch (const DriveError& e) {
        REQUIRE(e.status == 0);
    }
    REQUIRE(fake.calls.empty());
}
