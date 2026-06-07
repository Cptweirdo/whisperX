// Core audio hyperparameters — the C++ mirror of whisperx/audio.py:13-22.
// These bind the decode/VAD/mel/align stages to a single 16 kHz mono contract;
// keep them in lockstep with the Python constants.
#pragma once

namespace whisperx::audio {

constexpr int kSampleRate = 16000;        // SAMPLE_RATE
constexpr int kNFft = 400;                // N_FFT
constexpr int kHopLength = 160;           // HOP_LENGTH
constexpr int kChunkLength = 30;          // CHUNK_LENGTH (seconds)
constexpr int kNSamples = kChunkLength * kSampleRate;  // N_SAMPLES = 480000
constexpr int kNFrames = kNSamples / kHopLength;       // N_FRAMES = 3000

constexpr int kNSamplesPerToken = kHopLength * 2;          // 320 (stride-2 conv)
constexpr int kFramesPerSecond = kSampleRate / kHopLength;  // 100 (10 ms/frame)
constexpr int kTokensPerSecond = kSampleRate / kNSamplesPerToken;  // 50

}  // namespace whisperx::audio
