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
