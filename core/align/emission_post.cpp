#include "align/emission_post.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "text/utf8.hpp"

namespace whisperx::align {

PostResult emission_post(const float* logits, std::size_t T, std::size_t V,
                         int blank_id, const std::string& text_clean,
                         const std::map<std::string, int>& dictionary) {
    // 1. log_softmax over the label axis (stable max-subtract), fp32 — matches
    //    torch.log_softmax element-for-element within the emission_atol budget.
    std::vector<float> ls(T * V);
    for (std::size_t t = 0; t < T; ++t) {
        const float* row = logits + t * V;
        float m = row[0];
        for (std::size_t v = 1; v < V; ++v) m = std::max(m, row[v]);
        float sum = 0.0f;
        for (std::size_t v = 0; v < V; ++v) sum += std::exp(row[v] - m);
        const float logsum = std::log(sum);
        float* out = ls.data() + t * V;
        for (std::size_t v = 0; v < V; ++v) out[v] = (row[v] - m) - logsum;
    }

    // 2. detect OOV codepoints (the Python `has_wildcard` test).
    const auto chars = text::utf8_chars(text_clean);
    auto lookup = [&](const text::Utf8Char& c) {
        return dictionary.find(text_clean.substr(c.offset, c.length));
    };
    bool has_wildcard = false;
    for (const auto& c : chars)
        if (lookup(c) == dictionary.end()) {
            has_wildcard = true;
            break;
        }

    PostResult r;
    r.T = T;
    if (!has_wildcard) {
        r.V = V;
        r.emission = std::move(ls);
        r.tokens.reserve(chars.size());
        for (const auto& c : chars) r.tokens.push_back(lookup(c)->second);
        return r;
    }

    // 3. wildcard: append the max-non-blank column (over log-softmax values) and map
    //    OOV codepoints to that new last column (id == V). alignment.py:296-303.
    const std::size_t Vp = V + 1;
    const int wildcard_id = static_cast<int>(V);
    r.V = Vp;
    r.emission.resize(T * Vp);
    for (std::size_t t = 0; t < T; ++t) {
        const float* src = ls.data() + t * V;
        float* dst = r.emission.data() + t * Vp;
        float wmax = -std::numeric_limits<float>::infinity();
        for (std::size_t v = 0; v < V; ++v) {
            dst[v] = src[v];
            if (static_cast<int>(v) != blank_id) wmax = std::max(wmax, src[v]);
        }
        dst[V] = wmax;
    }
    r.tokens.reserve(chars.size());
    for (const auto& c : chars) {
        const auto it = lookup(c);
        r.tokens.push_back(it == dictionary.end() ? wildcard_id : it->second);
    }
    return r;
}

}  // namespace whisperx::align
