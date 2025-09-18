#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace packboost {

struct HistogramBuffers {
    std::vector<float> grad;
    std::vector<float> hess;
    std::vector<int32_t> count;
};

HistogramBuffers build_histograms_cpu(
    const uint8_t* bins,
    const float* gradients,
    const float* hessians,
    const int16_t* era_inverse,
    std::size_t n_rows,
    std::size_t n_features,
    int max_bins,
    int n_eras);

HistogramBuffers build_frontier_histograms_cpu(
    const uint8_t* bins,
    const int32_t* node_indices,
    const int32_t* node_offsets,
    const int32_t* feature_subset,
    const float* gradients,
    const float* hessians,
    const int16_t* era_inverse,
    std::size_t n_rows,
    std::size_t n_features_total,
    std::size_t n_nodes,
    std::size_t n_features_subset,
    int max_bins,
    int n_eras);

#ifdef PACKBOOST_ENABLE_CUDA
HistogramBuffers build_histograms_cuda(
    const uint8_t* bins,
    const float* gradients,
    const float* hessians,
    const int16_t* era_inverse,
    std::size_t n_rows,
    std::size_t n_features,
    int max_bins,
    int n_eras);
#endif

}  // namespace packboost
