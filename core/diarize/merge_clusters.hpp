// Centroid-level cluster merge — the post-pass that fixes sherpa FastClustering's
// over-segmentation. FastClustering greedily clusters noisy ~2s chunk embeddings,
// so one real speaker often lands in several clusters (12+ "speakers" on a real
// 2-speaker call). Pooled per-cluster embeddings average that noise out: centroids
// of fragments of the same speaker sit much closer than centroids of distinct
// speakers, so a second agglomeration over the centroids cleanly re-joins the
// fragments. NB: pooled-embedding cosine distances live on a smaller scale than
// chunk distances — merge thresholds here are ~0.3, not the ~0.8 chunk scale.
#pragma once

#include <map>
#include <vector>

namespace whisperx::diarize {

// Agglomerate cluster centroids with average linkage: repeatedly merge the
// closest pair (cosine distance < threshold), the merged centroid being the
// talk-time-weighted mean. Survivor keeps the smaller speaker id. Zero-norm
// centroids (failed embedding extraction) never merge. Returns the
// old-speaker -> surviving-speaker mapping (identity when nothing merges).
std::map<int, int> merge_speaker_clusters(
    const std::map<int, std::vector<float>>& centroids,
    const std::map<int, double>& talk_time, float threshold);

}  // namespace whisperx::diarize
