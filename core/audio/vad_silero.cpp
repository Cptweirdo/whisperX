#include "audio/vad_silero.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "sherpa-onnx/c-api/c-api.h"

namespace whisperx::audio {

namespace {
// RAII for the sherpa VAD handle (exception-safe drain loop below).
struct VadHandle {
    const SherpaOnnxVoiceActivityDetector* p = nullptr;
    ~VadHandle() {
        if (p) SherpaOnnxDestroyVoiceActivityDetector(p);
    }
};
}  // namespace

std::vector<whisperx::vad::VadSegment> silero_segments(
    const AudioBuffer& audio, const std::string& model_path, double onset,
    double chunk_size) {
    SherpaOnnxVadModelConfig config;
    std::memset(&config, 0, sizeof(config));
    config.silero_vad.model = model_path.c_str();
    config.silero_vad.threshold = static_cast<float>(onset);
    // sherpa-onnx silero defaults (kept off the whisperx knobs, which only set
    // threshold + max_speech_duration — see whisperx/vads/silero.py:42-51).
    config.silero_vad.min_silence_duration = 0.1f;
    config.silero_vad.min_speech_duration = 0.25f;
    config.silero_vad.max_speech_duration = static_cast<float>(chunk_size);
    config.silero_vad.window_size = 512;
    config.sample_rate = audio.sample_rate;
    config.num_threads = 1;
    config.provider = "cpu";  // non-null: sherpa wraps this in std::string
    config.debug = 0;

    VadHandle vad;
    vad.p = SherpaOnnxCreateVoiceActivityDetector(
        &config, static_cast<float>(chunk_size));
    if (!vad.p)
        throw std::runtime_error(
            "failed to create silero VAD (model: " + model_path + ")");

    std::vector<whisperx::vad::VadSegment> out;
    const double sr = audio.sample_rate;
    auto drain = [&]() {
        while (!SherpaOnnxVoiceActivityDetectorEmpty(vad.p)) {
            const SherpaOnnxSpeechSegment* seg =
                SherpaOnnxVoiceActivityDetectorFront(vad.p);
            out.push_back({seg->start / sr, (seg->start + seg->n) / sr,
                           std::string("UNKNOWN")});
            SherpaOnnxDestroySpeechSegment(seg);
            SherpaOnnxVoiceActivityDetectorPop(vad.p);
        }
    };

    // Offline: stream window_size samples at a time, draining completed
    // segments as we go (feeding the whole clip at once overflows sherpa's
    // internal circular buffer and mis-segments). Flush the tail at the end.
    const int32_t win = config.silero_vad.window_size > 0
                            ? config.silero_vad.window_size
                            : 512;
    const int32_t n = static_cast<int32_t>(audio.samples.size());
    for (int32_t i = 0; i < n; i += win) {
        const int32_t len = std::min(win, n - i);
        SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad.p,
                                                      audio.samples.data() + i,
                                                      len);
        drain();
    }
    SherpaOnnxVoiceActivityDetectorFlush(vad.p);
    drain();
    return out;
}

}  // namespace whisperx::audio
