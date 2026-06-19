#include "align/trellis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "text/utf8.hpp"

namespace whisperx::align {

namespace {
constexpr float kInf = std::numeric_limits<float>::infinity();
}  // namespace

std::vector<float> get_trellis(const Emission& emission,
                               const std::vector<int>& tokens, int blank_id) {
    const std::size_t T = emission.T;
    const std::size_t J = tokens.size();
    const std::size_t cols = J + 1;
    std::vector<float> trellis((T + 1) * cols);
    auto tr = [&](std::size_t t, std::size_t j) -> float& {
        return trellis[t * cols + j];
    };

    // trellis[0,0]=0; trellis[1:,0]=cumsum(emission[:,blank]); the ±inf
    // boundaries (alignment.py:433-436). Order matches Python: the cumsum fills
    // column 0, then the bottom J rows of column 0 are overwritten with +inf.
    tr(0, 0) = 0.0f;
    float acc = 0.0f;
    for (std::size_t t = 0; t < T; ++t) {
        acc += emission.at(t, static_cast<std::size_t>(blank_id));
        tr(t + 1, 0) = acc;
    }
    // trellis[0, -J:] = -inf  (row 0, columns 1..J)
    for (std::size_t j = 1; j <= J; ++j) tr(0, j) = -kInf;
    // trellis[-J:, 0] = +inf  (column 0, last J rows)
    for (std::size_t t = T + 1 - J; t <= T; ++t) tr(t, 0) = kInf;

    // The maximum(stay, change) recurrence (alignment.py:438-444).
    for (std::size_t t = 0; t < T; ++t) {
        const float blank = emission.at(t, static_cast<std::size_t>(blank_id));
        for (std::size_t j = 0; j < J; ++j) {
            const float stay = tr(t, j + 1) + blank;
            const float change =
                tr(t, j) + emission.at(t, static_cast<std::size_t>(tokens[j]));
            tr(t + 1, j + 1) = stay > change ? stay : change;
        }
    }
    return trellis;
}

std::optional<std::vector<Point>> backtrack(const std::vector<float>& trellis,
                                            const Emission& emission,
                                            const std::vector<int>& tokens,
                                            int blank_id) {
    const std::size_t T = emission.T;
    const std::size_t J = tokens.size();
    const std::size_t cols = J + 1;
    auto tr = [&](std::size_t t, std::size_t j) -> float {
        return trellis[t * cols + j];
    };

    std::size_t j = J;  // last column
    // t_start = argmax over rows of column j (torch.argmax → first max).
    std::size_t t_start = 0;
    float best = tr(0, j);
    for (std::size_t t = 1; t <= T; ++t) {
        if (tr(t, j) > best) {
            best = tr(t, j);
            t_start = t;
        }
    }

    std::vector<Point> path;
    bool broke = false;
    for (std::size_t t = t_start; t >= 1; --t) {
        const float stayed =
            tr(t - 1, j) + emission.at(t - 1, static_cast<std::size_t>(blank_id));
        const float changed =
            tr(t - 1, j - 1) +
            emission.at(t - 1, static_cast<std::size_t>(tokens[j - 1]));
        const bool change = changed > stayed;
        const int label_idx = change ? tokens[j - 1] : blank_id;
        const float prob =
            std::exp(emission.at(t - 1, static_cast<std::size_t>(label_idx)));
        path.push_back({static_cast<int>(j) - 1, static_cast<int>(t) - 1, prob});
        if (change) {
            --j;
            if (j == 0) {
                broke = true;
                break;
            }
        }
    }
    if (!broke) return std::nullopt;  // alignment.py: for/else → return None

    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<Segment> merge_repeats(const std::vector<Point>& path,
                                   const std::string& transcript) {
    // transcript is indexed by *codepoint* (Python str semantics) — tokens map to
    // codepoints of the cleaned text, which may be multibyte (Cyrillic clips).
    const std::vector<whisperx::text::Utf8Char> chars =
        whisperx::text::utf8_chars(transcript);

    std::vector<Segment> segments;
    std::size_t i1 = 0;
    while (i1 < path.size()) {
        std::size_t i2 = i1;
        while (i2 < path.size() &&
               path[i1].token_index == path[i2].token_index) {
            ++i2;
        }
        float score = 0.0f;
        for (std::size_t k = i1; k < i2; ++k) score += path[k].score;
        score /= static_cast<float>(i2 - i1);

        const auto& ch = chars[static_cast<std::size_t>(path[i1].token_index)];
        segments.push_back({transcript.substr(ch.offset, ch.length),
                            path[i1].time_index, path[i2 - 1].time_index + 1,
                            score});
        i1 = i2;
    }
    return segments;
}

std::vector<Segment> merge_words(const std::vector<Segment>& segments,
                                 const std::string& separator) {
    std::vector<Segment> words;
    std::size_t i1 = 0, i2 = 0;
    while (i1 < segments.size()) {
        if (i2 >= segments.size() || segments[i2].label == separator) {
            if (i1 != i2) {
                std::string word;
                float num = 0.0f;
                int den = 0;
                for (std::size_t k = i1; k < i2; ++k) {
                    word += segments[k].label;
                    num += segments[k].score * segments[k].length();
                    den += segments[k].length();
                }
                words.push_back({word, segments[i1].start, segments[i2 - 1].end,
                                 num / static_cast<float>(den)});
            }
            i1 = i2 + 1;
            i2 = i1;
        } else {
            ++i2;
        }
    }
    return words;
}

}  // namespace whisperx::align
