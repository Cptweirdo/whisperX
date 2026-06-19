// sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-whisper-decoder.h"
#include "sherpa-onnx/csrc/offline-whisper-model.h"

namespace sherpa_onnx {

class OfflineWhisperGreedySearchDecoder : public OfflineWhisperDecoder {
 public:
  OfflineWhisperGreedySearchDecoder(const OfflineWhisperModelConfig &config,
                                    OfflineWhisperModel *model)
      : config_(config), model_(model) {}

  std::vector<OfflineWhisperDecoderResult> Decode(
      Ort::Value cross_k, Ort::Value cross_v,
      int32_t num_feature_frames) override;

  /** whisperX batched-decode patch: lockstep greedy decode over a batch of N
   * encoder outputs (cross_k/cross_v have batch dim N at axis 1).
   *
   * Preconditions (the recognizer guards them): config_.language is non-empty
   * (no per-row language detection), config_.task is set, and neither
   * enable_token_timestamps nor enable_segment_timestamps is on — the batch
   * path decodes plain text only.
   *
   * @param num_feature_frames One entry per row; bounds each row's token count.
   * @return One result per row, in row order.
   */
  std::vector<OfflineWhisperDecoderResult> DecodeBatch(
      Ort::Value cross_k, Ort::Value cross_v,
      const std::vector<int32_t> &num_feature_frames);

  void SetConfig(const OfflineWhisperModelConfig &config) override;

 private:
  OfflineWhisperModelConfig config_;
  OfflineWhisperModel *model_;  // not owned
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_GREEDY_SEARCH_DECODER_H_
