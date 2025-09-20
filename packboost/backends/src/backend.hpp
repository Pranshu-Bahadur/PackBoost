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

struct FrontierEvalResult {
    std::vector<int32_t> best_feature;
    std::vector<int32_t> best_threshold;
    std::vector<float> score;
    std::vector<float> agreement;
    std::vector<float> left_value;
    std::vector<float> right_value;
    std::vector<float> base_value;
    std::vector<int32_t> left_offsets;
    std::vector<int32_t> right_offsets;
    std::vector<int32_t> left_indices;
    std::vector<int32_t> right_indices;
};

struct FastpathResult {
    std::vector<int32_t> best_feature;
    std::vector<int32_t> best_threshold;
    std::vector<float> score;
    std::vector<float> agreement;
    std::vector<float> left_grad;
    std::vector<float> left_hess;
    std::vector<int32_t> left_count;
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

FrontierEvalResult evaluate_frontier_cpu(
    const uint8_t* bins,
    const int32_t* node_indices,
    const int32_t* node_offsets,
    const int32_t* node_era_offsets,
    const int32_t* era_group_eras,
    const int32_t* era_group_offsets,
    const int32_t* feature_subset,
    const float* gradients,
    const float* hessians,
    std::size_t n_rows,
    std::size_t n_features_total,
    std::size_t n_nodes,
    std::size_t n_features_subset,
    int max_bins,
    int n_eras_total,
    double lambda_l2,
    double lambda_dro,
    int min_samples_leaf,
    double direction_weight,
    int era_tile_size);

FastpathResult fastpath_frontier_cpu(
    const uint8_t* bins,
    const float* gradients,
    const float* hessians,
    const int32_t* node_indices,
    const int32_t* node_base_offsets,
    const int32_t* node_era_offsets,
    std::size_t n_rows,
    std::size_t n_features_total,
    std::size_t n_nodes,
    int n_eras,
    const int32_t* feature_subset,
    std::size_t n_features_subset,
    int max_bins,
    const float* parent_grad,
    const float* parent_hess,
    const int32_t* parent_count,
    float lambda_l2,
    float lambda_dro,
    int min_samples_leaf,
    float direction_weight);

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

FrontierEvalResult evaluate_frontier_cuda(
    const uint8_t* bins,
    const int32_t* node_indices,
    const int32_t* node_offsets,
    const int32_t* node_era_offsets,
    const int32_t* era_group_eras,
    const int32_t* era_group_offsets,
    const int32_t* feature_subset,
    const float* gradients,
    const float* hessians,
    std::size_t n_rows,
    std::size_t n_features_total,
    std::size_t n_nodes,
    std::size_t n_features_subset,
    int max_bins,
    int n_eras_total,
    double lambda_l2,
    double lambda_dro,
    int min_samples_leaf,
    double direction_weight,
    int era_tile_size,
    int threads_per_block,
    int rows_per_thread);

FastpathResult fastpath_frontier_cuda(
    const uint8_t* bins,
    const float* gradients,
    const float* hessians,
    const int32_t* node_indices,
    const int32_t* node_base_offsets,
    const int32_t* node_era_offsets,
    std::size_t n_rows,
    std::size_t n_features_total,
    std::size_t n_nodes,
    int n_eras,
    const int32_t* feature_subset,
    std::size_t n_features_subset,
    int max_bins,
    const float* parent_grad,
    const float* parent_hess,
    const int32_t* parent_count,
    float lambda_l2,
    float lambda_dro,
    int min_samples_leaf,
    float direction_weight);
#endif

}  // namespace packboost
