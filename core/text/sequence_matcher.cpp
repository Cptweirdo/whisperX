#include "text/sequence_matcher.hpp"

#include <algorithm>
#include <array>
#include <utility>

// TODO: consider replacing this hand-rolled difflib port with rapidfuzz
// (rapidfuzz-cpp) once it's pulled in as a dep — it ships a fast, well-tested
// SequenceMatcher equivalent. Would need to confirm bit-for-bit parity with
// CPython's autojunk=False matching-block output (the tie-break + sentinel) that
// realign_words depends on before swapping.
namespace whisperx::text {

SequenceMatcher::SequenceMatcher(std::vector<std::string> a,
                                 std::vector<std::string> b)
    : a_(std::move(a)), b_(std::move(b)) {
    // __chain_b (autojunk=False, isjunk=None): just b2j, no junk/popular purge.
    for (int i = 0; i < static_cast<int>(b_.size()); ++i) {
        b2j_[b_[i]].push_back(i);
    }
}

Match SequenceMatcher::find_longest_match(int alo, int ahi, int blo,
                                          int bhi) const {
    int besti = alo, bestj = blo, bestsize = 0;
    std::unordered_map<int, int> j2len;
    static const std::vector<int> nothing;
    for (int i = alo; i < ahi; ++i) {
        std::unordered_map<int, int> newj2len;
        auto it = b2j_.find(a_[i]);
        const std::vector<int>& js = (it != b2j_.end()) ? it->second : nothing;
        for (int j : js) {
            if (j < blo) continue;
            if (j >= bhi) break;
            int prev = 0;
            auto pit = j2len.find(j - 1);
            if (pit != j2len.end()) prev = pit->second;
            const int k = prev + 1;
            newj2len[j] = k;
            // Strict `>`: earliest i wins, then earliest j (b2j is ascending).
            if (k > bestsize) {
                besti = i - k + 1;
                bestj = j - k + 1;
                bestsize = k;
            }
        }
        j2len = std::move(newj2len);
    }
    // No junk: the two "extend over equal, non-junk neighbours" loops collapse to
    // plain equality extension; the third (junk-only extension) never runs.
    while (besti > alo && bestj > blo && a_[besti - 1] == b_[bestj - 1]) {
        --besti;
        --bestj;
        ++bestsize;
    }
    while (besti + bestsize < ahi && bestj + bestsize < bhi &&
           a_[besti + bestsize] == b_[bestj + bestsize]) {
        ++bestsize;
    }
    return Match{besti, bestj, bestsize};
}

const std::vector<Match>& SequenceMatcher::get_matching_blocks() {
    if (computed_) return matching_blocks_;
    const int la = static_cast<int>(a_.size());
    const int lb = static_cast<int>(b_.size());

    // queue of (alo, ahi, blo, bhi); pop the last (matches Python list.pop()).
    std::vector<std::array<int, 4>> queue = {{0, la, 0, lb}};
    std::vector<Match> blocks;
    while (!queue.empty()) {
        const auto box = queue.back();
        queue.pop_back();
        const int alo = box[0], ahi = box[1], blo = box[2], bhi = box[3];
        const Match m = find_longest_match(alo, ahi, blo, bhi);
        if (m.size) {
            blocks.push_back(m);
            if (alo < m.a && blo < m.b) {
                queue.push_back({alo, m.a, blo, m.b});
            }
            if (m.a + m.size < ahi && m.b + m.size < bhi) {
                queue.push_back({m.a + m.size, ahi, m.b + m.size, bhi});
            }
        }
    }

    // matching_blocks.sort() — lexicographic by (a, b, size).
    std::sort(blocks.begin(), blocks.end(), [](const Match& x, const Match& y) {
        if (x.a != y.a) return x.a < y.a;
        if (x.b != y.b) return x.b < y.b;
        return x.size < y.size;
    });

    // Coalesce adjacent blocks, then append the (la, lb, 0) sentinel.
    int i1 = 0, j1 = 0, k1 = 0;
    std::vector<Match> non_adjacent;
    for (const Match& blk : blocks) {
        if (i1 + k1 == blk.a && j1 + k1 == blk.b) {
            k1 += blk.size;
        } else {
            if (k1) non_adjacent.push_back(Match{i1, j1, k1});
            i1 = blk.a;
            j1 = blk.b;
            k1 = blk.size;
        }
    }
    if (k1) non_adjacent.push_back(Match{i1, j1, k1});
    non_adjacent.push_back(Match{la, lb, 0});

    matching_blocks_ = std::move(non_adjacent);
    computed_ = true;
    return matching_blocks_;
}

}  // namespace whisperx::text
