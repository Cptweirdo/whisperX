// Phase-2B bench: decode + silero-VAD real-time-factor (RTF) on one clip.
// Seeds the Phase-6 timing gates. Intentionally dependency-light (std::chrono,
// no Google Benchmark) so it rides along with the WHISPERX_CORE_AUDIO build.
//
//   bench_audio <audio_file> [silero_vad.onnx]
//
// RTF = wall_time / audio_duration; < 1.0 is faster than real time.
#include <chrono>
#include <cstdio>
#include <string>

#include "audio/decode.hpp"
#include "audio/vad_silero.hpp"

using clk = std::chrono::steady_clock;

static double secs_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <audio_file> [silero_vad.onnx]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const std::string model = argc > 2 ? argv[2] : "models/silero_vad.onnx";

    namespace audio = whisperx::audio;

    // --- decode (averaged over a few runs) ---
    constexpr int kRuns = 5;
    audio::AudioBuffer buf;
    double decode_total = 0.0;
    for (int i = 0; i < kRuns; ++i) {
        auto t0 = clk::now();
        buf = audio::load_audio(path);
        decode_total += secs_since(t0);
    }
    const double decode_avg = decode_total / kRuns;
    const double dur = buf.duration_s();

    std::printf("clip            : %s\n", path.c_str());
    std::printf("duration        : %.3f s (%zu samples @ %d Hz)\n", dur,
                buf.size(), buf.sample_rate);
    std::printf("decode          : %.4f s  RTF %.4f  (avg of %d)\n", decode_avg,
                dur > 0 ? decode_avg / dur : 0.0, kRuns);

    // --- silero VAD (single run; model load dominates first call) ---
    auto t0 = clk::now();
    auto segs = audio::silero_segments(buf, model, 0.5, 30.0);
    const double vad_t = secs_since(t0);
    std::printf("vad (silero/ORT): %.4f s  RTF %.4f  (%zu segments)\n", vad_t,
                dur > 0 ? vad_t / dur : 0.0, segs.size());
    return 0;
}
