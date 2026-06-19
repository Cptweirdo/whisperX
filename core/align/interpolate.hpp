// Port of whisperx/utils.py::interpolate_nans — fills missing word/segment
// timestamps. Header-only (tiny). Operates on a sequence of optional doubles
// (nullopt == NaN). Mirrors pandas' two-branch logic:
//
//   if method == "ignore": return x                       # NaNs preserved
//   if x.notnull().sum() > 1:
//       return x.interpolate(method).ffill().bfill()
//   else:
//       return x.ffill().bfill()                           # 0/1 valid → broadcast
//
// For method="nearest", every NaN is filled with the value at the nearest valid
// integer index (tie → lower index — note: scipy's interp1d tie-break is not
// formally specified, a small residual fidelity risk recorded in the brief). With
// ≤1 valid point this degenerates to ffill/bfill (broadcast the lone value).
#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace whisperx::align {

inline std::vector<std::optional<double>> interpolate_nans(
    std::vector<std::optional<double>> x, const std::string& method) {
    if (method == "ignore") return x;

    // collect valid indices
    std::vector<std::size_t> valid;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (x[i].has_value()) valid.push_back(i);
    if (valid.empty()) return x;  // all NaN — nothing to fill

    std::vector<std::optional<double>> out = x;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i].has_value()) continue;
        // nearest valid index; tie → lower (valid is ascending, first wins).
        std::size_t best = valid[0];
        std::size_t best_d = best > i ? best - i : i - best;
        for (std::size_t j : valid) {
            std::size_t d = j > i ? j - i : i - j;
            if (d < best_d) {
                best_d = d;
                best = j;
            }
        }
        out[i] = x[best];
    }
    return out;
}

}  // namespace whisperx::align
