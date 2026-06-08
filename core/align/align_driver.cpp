#include "align/align_driver.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "align/align.hpp"
#include "align/char_clean.hpp"
#include "align/emission_post.hpp"
#include "audio/audio_constants.hpp"

namespace whisperx::align {

namespace {

// blank/pad id — model-constant (alignment.py:236-240): scan the dictionary for a
// "[pad]"/"<pad>" key, default 0.
int detect_blank_id(const std::map<std::string, int>& dictionary) {
    int blank_id = 0;
    for (const auto& [ch, code] : dictionary)
        if (ch == "[pad]" || ch == "<pad>") blank_id = code;
    return blank_id;
}

}  // namespace

json align_run(const json& transcript, Wav2Vec2Onnx& model,
               const std::map<std::string, int>& dictionary,
               std::span<const float> audio, const std::string& language,
               bool batchable, const std::string& interpolate_method,
               bool return_char_alignments,
               const std::function<void(double)>& progress) {
    constexpr double kSR = static_cast<double>(audio::kSampleRate);
    const double max_duration = static_cast<double>(audio.size()) / kSR;
    const int blank_id = detect_blank_id(dictionary);

    // Accept either {"segments": [...]} or a bare segment array.
    const json& segments =
        transcript.is_object() ? transcript.at("segments") : transcript;
    const std::size_t total = segments.size();

    // 1. Per-segment char-cleaning (alignment.py:245-298, minus the dead punkt span
    //    computation — align_assemble recomputes sentence spans natively).
    std::vector<CleanResult> seg_data(total);
    for (std::size_t i = 0; i < total; ++i)
        seg_data[i] = clean_segment(segments[i].at("text").get<std::string>(),
                                    language, dictionary);

    // 2. Gather alignable segments and run one batched ORT forward (alignment.py:
    //    308-322). A segment is alignable iff it has clean chars and starts before
    //    the audio ends.
    std::vector<std::size_t> gather_idx;
    std::vector<std::span<const float>> gather_wav;
    for (std::size_t i = 0; i < total; ++i) {
        if (seg_data[i].clean_char.empty()) continue;
        const double start = segments[i].at("start").get<double>();
        if (start >= max_duration) continue;
        std::size_t f1 = static_cast<std::size_t>(start * kSR);  // int() truncates
        std::size_t f2 =
            static_cast<std::size_t>(segments[i].at("end").get<double>() * kSR);
        if (f1 > audio.size()) f1 = audio.size();
        if (f2 > audio.size()) f2 = audio.size();
        if (f2 < f1) f2 = f1;
        gather_idx.push_back(i);
        gather_wav.push_back(audio.subspan(f1, f2 - f1));
    }
    // sdx -> (flat logits, {T, V})
    std::map<std::size_t, std::pair<std::vector<float>,
                                    std::pair<std::size_t, std::size_t>>>
        logits;
    if (!gather_wav.empty()) {
        std::vector<std::pair<std::size_t, std::size_t>> shapes;
        std::vector<std::vector<float>> raw =
            model.forward(gather_wav, shapes, batchable);
        for (std::size_t k = 0; k < gather_idx.size(); ++k)
            logits[gather_idx[k]] = {std::move(raw[k]), shapes[k]};
    }

    // 3. Per segment: emission_post -> align_assemble (or a stub on the failure
    //    paths), exactly mirroring the Python loop's control flow.
    json aligned_segments = json::array();
    for (std::size_t i = 0; i < total; ++i) {
        const json& seg = segments[i];
        const double t1 = seg.at("start").get<double>();
        const double t2 = seg.at("end").get<double>();
        const std::string text = seg.at("text").get<std::string>();
        std::optional<double> avg_logprob;
        if (seg.contains("avg_logprob") && !seg.at("avg_logprob").is_null())
            avg_logprob = seg.at("avg_logprob").get<double>();

        auto stub = [&]() {
            json s = {{"start", t1},
                      {"end", t2},
                      {"text", text},
                      {"words", json::array()},
                      {"chars", nullptr}};
            if (avg_logprob) s["avg_logprob"] = *avg_logprob;
            if (return_char_alignments) s["chars"] = json::array();
            return s;
        };

        // No alignable chars / start past audio end -> stub, no progress (matches
        // the Python `continue` before the callback).
        if (seg_data[i].clean_char.empty() || t1 >= max_duration) {
            aligned_segments.push_back(stub());
            continue;
        }

        const auto& [flat, tv] = logits.at(i);
        PostResult post = emission_post(flat.data(), tv.first, tv.second, blank_id,
                                        seg_data[i].clean_char, dictionary);
        Emission emission{post.emission.data(), post.T, post.V};
        AssembleResult res = align_assemble(
            emission, post.tokens, blank_id, seg_data[i].clean_char, text,
            seg_data[i].clean_cdx, t1, t2, language, interpolate_method,
            return_char_alignments, avg_logprob);

        if (!res.ok) {
            aligned_segments.push_back(stub());
            continue;  // backtrack failed -> stub, no progress (Python parity)
        }
        for (auto& sub : res.subsegments)
            aligned_segments.push_back(std::move(sub));
        if (progress)
            progress((static_cast<double>(i + 1) / total) * 100.0);
    }

    // 4. Flatten the word segments (alignment.py:556-559).
    json word_segments = json::array();
    for (const auto& seg : aligned_segments)
        for (const auto& w : seg.at("words")) word_segments.push_back(w);

    return json{{"segments", std::move(aligned_segments)},
                {"word_segments", std::move(word_segments)}};
}

}  // namespace whisperx::align
