#include "backend.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace packboost {

namespace {

constexpr int WARP_SIZE = 32;
constexpr float NEG_INF_F = -CUDART_INF_F;

inline void check(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

__inline__ __device__ int lane_id() {
    int id;
    asm volatile("mov.s32 %0, %laneid;" : "=r"(id));
    return id;
}

__global__ void histogram_kernel(
    const uint8_t* __restrict__ bins,
    const float* __restrict__ gradients,
    const float* __restrict__ hessians,
    const int16_t* __restrict__ era_inverse,
    int n_rows,
    int n_features,
    int max_bins,
    int n_eras,
    float* __restrict__ out_grad,
    float* __restrict__ out_hess,
    int32_t* __restrict__ out_count) {

    extern __shared__ unsigned char shared_raw[];
    float* shared_grad = reinterpret_cast<float*>(shared_raw);
    float* shared_hess = shared_grad + max_bins * n_eras;
    int32_t* shared_count = reinterpret_cast<int32_t*>(shared_hess + max_bins * n_eras);

    const int feature = blockIdx.x;
    // zero shared memory
    for (int idx = threadIdx.x; idx < max_bins * n_eras; idx += blockDim.x) {
        shared_grad[idx] = 0.0f;
        shared_hess[idx] = 0.0f;
        shared_count[idx] = 0;
    }
    __syncthreads();

    for (int row = threadIdx.x; row < n_rows; row += blockDim.x) {
        const int era = static_cast<int>(era_inverse[row]);
        if (era < 0 || era >= n_eras) continue;
        const uint8_t bin = bins[row * n_features + feature];
        if (bin >= max_bins) continue;
        const int idx = bin * n_eras + era;
        atomicAdd(shared_grad + idx, gradients[row]);
        atomicAdd(shared_hess + idx, hessians[row]);
        atomicAdd(shared_count + idx, 1);
    }
    __syncthreads();

    float* grad_out = out_grad + feature * max_bins * n_eras;
    float* hess_out = out_hess + feature * max_bins * n_eras;
    int32_t* count_out = out_count + feature * max_bins * n_eras;

    for (int idx = threadIdx.x; idx < max_bins * n_eras; idx += blockDim.x) {
        atomicAdd(grad_out + idx, shared_grad[idx]);
        atomicAdd(hess_out + idx, shared_hess[idx]);
        atomicAdd(count_out + idx, shared_count[idx]);
    }
}

void launch_histogram_kernel(
    const uint8_t* bins,
    const float* gradients,
    const float* hessians,
    const int16_t* era_inverse,
    int n_rows,
    int n_features,
    int max_bins,
    int n_eras,
    float* out_grad,
    float* out_hess,
    int32_t* out_count) {

    const dim3 grid(n_features);
    const int threads = 256;
    const std::size_t shared = static_cast<std::size_t>(max_bins) * n_eras * (2 * sizeof(float) + sizeof(int32_t));
    histogram_kernel<<<grid, threads, shared>>>(
        bins, gradients, hessians, era_inverse,
        n_rows, n_features, max_bins, n_eras,
        out_grad, out_hess, out_count);
    check(cudaDeviceSynchronize(), "histogram_kernel launch failed");
}

__global__ void frontier_feature_kernel(
    const uint8_t* __restrict__ bins,
    const int32_t* __restrict__ node_indices,
    const int32_t* __restrict__ node_offsets,
    const int32_t* __restrict__ node_era_offsets,
    const int32_t* __restrict__ era_group_offsets,
    const int32_t* __restrict__ feature_subset,
    const float* __restrict__ gradients,
    const float* __restrict__ hessians,
    int n_rows,
    int n_features_total,
    int max_bins,
    int thresholds,
    int n_features_subset,
    int n_eras_total,
    int rows_per_thread,
    int min_samples_leaf,
    float lambda,
    float lambda_dro,
    float direction_weight,
    float eps,
    int32_t* best_threshold_out,
    float* score_out,
    float* agreement_out,
    float* left_value_out,
    float* right_value_out,
    int32_t* left_count_out,
    int32_t* right_count_out)
{
    if (rows_per_thread < 1) {
        rows_per_thread = 1;
    }

    const int node_idx = blockIdx.x;
    const int feature_idx = blockIdx.y;
    if (node_idx >= gridDim.x || feature_idx >= n_features_subset) {
        return;
    }

    const int32_t node_begin = node_offsets[node_idx];
    const int32_t node_end = node_offsets[node_idx + 1];
    const int32_t node_rows = node_end - node_begin;

    const int feature_slot = node_idx * n_features_subset + feature_idx;

    if (node_rows == 0 || thresholds <= 0) {
        if (threadIdx.x == 0) {
            best_threshold_out[feature_slot] = -1;
            score_out[feature_slot] = NEG_INF_F;
            agreement_out[feature_slot] = 0.0f;
            left_value_out[feature_slot] = 0.0f;
            right_value_out[feature_slot] = 0.0f;
            left_count_out[feature_slot] = node_rows;
            right_count_out[feature_slot] = 0;
        }
        return;
    }

    const int32_t era_begin = node_era_offsets[node_idx];
    const int32_t era_end = node_era_offsets[node_idx + 1];
    if (era_begin == era_end) {
        if (threadIdx.x == 0) {
            best_threshold_out[feature_slot] = -1;
            score_out[feature_slot] = NEG_INF_F;
            agreement_out[feature_slot] = 0.0f;
            left_value_out[feature_slot] = 0.0f;
            right_value_out[feature_slot] = 0.0f;
            left_count_out[feature_slot] = node_rows;
            right_count_out[feature_slot] = 0;
        }
        return;
    }

    const int32_t feature = feature_subset[feature_idx];
    const float era_normaliser = fmaxf(1.0f, static_cast<float>(n_eras_total));

    extern __shared__ unsigned char shared_raw[];
    float* hist_grad = reinterpret_cast<float*>(shared_raw);
    float* hist_hess = hist_grad + max_bins;
    int32_t* hist_count = reinterpret_cast<int32_t*>(hist_hess + max_bins);
    float* gain_sum = reinterpret_cast<float*>(hist_count + max_bins);
    float* gain_sq_sum = gain_sum + thresholds;
    float* left_grad_acc = gain_sq_sum + thresholds;
    float* left_hess_acc = left_grad_acc + thresholds;
    float* right_grad_acc = left_hess_acc + thresholds;
    float* right_hess_acc = right_grad_acc + thresholds;
    int32_t* left_count_acc = reinterpret_cast<int32_t*>(right_hess_acc + thresholds);
    int32_t* right_count_acc = left_count_acc + thresholds;
    int32_t* contribution_acc = right_count_acc + thresholds;
    int32_t* direction_acc = contribution_acc + thresholds;

    for (int idx = threadIdx.x; idx < thresholds; idx += blockDim.x) {
        gain_sum[idx] = 0.0f;
        gain_sq_sum[idx] = 0.0f;
        left_grad_acc[idx] = 0.0f;
        left_hess_acc[idx] = 0.0f;
        right_grad_acc[idx] = 0.0f;
        right_hess_acc[idx] = 0.0f;
        left_count_acc[idx] = 0;
        right_count_acc[idx] = 0;
        contribution_acc[idx] = 0;
        direction_acc[idx] = 0;
    }
    __syncthreads();

    for (int32_t era_idx = era_begin; era_idx < era_end; ++era_idx) {
        const int32_t range_begin = era_group_offsets[era_idx];
        const int32_t range_end = era_group_offsets[era_idx + 1];
        if (range_begin >= range_end) {
            continue;
        }

        for (int bin = threadIdx.x; bin < max_bins; bin += blockDim.x) {
            hist_grad[bin] = 0.0f;
            hist_hess[bin] = 0.0f;
            hist_count[bin] = 0;
        }
        __syncthreads();

        for (int32_t pos_block = range_begin + threadIdx.x * rows_per_thread;
             pos_block < range_end;
             pos_block += blockDim.x * rows_per_thread) {
            for (int r = 0; r < rows_per_thread; ++r) {
                const int32_t pos = pos_block + r;
                if (pos >= range_end) {
                    break;
                }
                const int32_t row = node_indices[pos];
                if (row < 0 || row >= n_rows) {
                    continue;
                }
                const std::size_t feature_offset = static_cast<std::size_t>(row) * n_features_total + static_cast<std::size_t>(feature);
                const uint8_t bin = bins[feature_offset];
                if (bin >= max_bins) {
                    continue;
                }
                const float grad = gradients[row];
                const float hess = hessians[row];
                atomicAdd(&hist_grad[bin], grad);
                atomicAdd(&hist_hess[bin], hess);
                atomicAdd(&hist_count[bin], 1);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float total_grad = 0.0f;
            float total_hess = 0.0f;
            int total_count = 0;
            for (int bin = 0; bin < max_bins; ++bin) {
                total_grad += hist_grad[bin];
                total_hess += hist_hess[bin];
                total_count += hist_count[bin];
            }
            if (total_count == 0) {
                continue;
            }

            const float parent_gain = 0.5f * (total_grad * total_grad) / (total_hess + lambda + eps);
            float left_grad_running = 0.0f;
            float left_hess_running = 0.0f;
            int left_count_running = 0;

            for (int threshold = 0; threshold < thresholds; ++threshold) {
                left_grad_running += hist_grad[threshold];
                left_hess_running += hist_hess[threshold];
                left_count_running += hist_count[threshold];

                const int right_count = total_count - left_count_running;
                const float right_grad = total_grad - left_grad_running;
                const float right_hess = total_hess - left_hess_running;

                const float left_gain = (left_count_running > 0)
                    ? 0.5f * (left_grad_running * left_grad_running) / (left_hess_running + lambda + eps)
                    : 0.0f;
                const float right_gain = (right_count > 0)
                    ? 0.5f * (right_grad * right_grad) / (right_hess + lambda + eps)
                    : 0.0f;
                const float gain = left_gain + right_gain - parent_gain;

                const int idx = threshold;
                gain_sum[idx] += gain;
                gain_sq_sum[idx] += gain * gain;
                contribution_acc[idx] += 1;
                left_grad_acc[idx] += left_grad_running;
                left_hess_acc[idx] += left_hess_running;
                right_grad_acc[idx] += right_grad;
                right_hess_acc[idx] += right_hess;
                left_count_acc[idx] += left_count_running;
                right_count_acc[idx] += right_count;

                const float left_value = (left_count_running > 0)
                    ? -left_grad_running / (left_hess_running + lambda + eps)
                    : 0.0f;
                const float right_value = (right_count > 0)
                    ? -right_grad / (right_hess + lambda + eps)
                    : 0.0f;
                direction_acc[idx] += (left_value >= right_value) ? 1 : -1;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float best_score = NEG_INF_F;
        int best_threshold = -1;
        float best_agreement = 0.0f;
        float best_left_value = 0.0f;
        float best_right_value = 0.0f;
        int best_left_count = node_rows;
        int best_right_count = 0;

        for (int threshold = 0; threshold < thresholds; ++threshold) {
            const int contributions = contribution_acc[threshold];
            const int left_total = left_count_acc[threshold];
            const int right_total = right_count_acc[threshold];
            if (left_total < min_samples_leaf || right_total < min_samples_leaf) {
                continue;
            }

            float mean_gain = gain_sum[threshold] / era_normaliser;
            float variance = gain_sq_sum[threshold] / era_normaliser - mean_gain * mean_gain;
            if (variance < 0.0f) {
                variance = 0.0f;
            }
            const float std_gain = std::sqrt(variance);
            const float dro_score = mean_gain - lambda_dro * std_gain;
            int missing = n_eras_total - contributions;
            if (missing < 0) {
                missing = 0;
            }
            const float agreement = fabsf(static_cast<float>(direction_acc[threshold] + missing)) / era_normaliser;
            const float final_score = dro_score + direction_weight * agreement;

            if (final_score > best_score) {
                best_score = final_score;
                best_threshold = threshold;
                best_agreement = agreement;
                best_left_value = -left_grad_acc[threshold] / (left_hess_acc[threshold] + lambda + eps);
                best_right_value = -right_grad_acc[threshold] / (right_hess_acc[threshold] + lambda + eps);
                best_left_count = left_total;
                best_right_count = right_total;
            }
        }

        if (best_threshold < 0) {
            best_threshold_out[feature_slot] = -1;
            score_out[feature_slot] = NEG_INF_F;
            agreement_out[feature_slot] = 0.0f;
            left_value_out[feature_slot] = 0.0f;
            right_value_out[feature_slot] = 0.0f;
            left_count_out[feature_slot] = node_rows;
            right_count_out[feature_slot] = 0;
        } else {
            best_threshold_out[feature_slot] = best_threshold;
            score_out[feature_slot] = best_score;
            agreement_out[feature_slot] = best_agreement;
            left_value_out[feature_slot] = best_left_value;
            right_value_out[feature_slot] = best_right_value;
            left_count_out[feature_slot] = best_left_count;
            right_count_out[feature_slot] = best_right_count;
        }
    }
}

__global__ void frontier_select_kernel(
    const int32_t* __restrict__ feature_subset,
    int n_features_subset,
    const int32_t* __restrict__ node_offsets,
    int n_nodes,
    const int32_t* __restrict__ best_threshold_per_feature,
    const float* __restrict__ score_per_feature,
    const float* __restrict__ agreement_per_feature,
    const float* __restrict__ left_value_per_feature,
    const float* __restrict__ right_value_per_feature,
    const int32_t* __restrict__ left_count_per_feature,
    const int32_t* __restrict__ right_count_per_feature,
    int32_t* best_feature_out,
    int32_t* best_threshold_out,
    float* score_out,
    float* agreement_out,
    float* left_value_out,
    float* right_value_out,
    int32_t* left_count_out,
    int32_t* right_count_out)
{
    const int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= n_nodes) {
        return;
    }

    const int32_t node_rows = node_offsets[node_idx + 1] - node_offsets[node_idx];
    int best_feature_slot = -1;
    int32_t best_threshold = 0;
    float best_score = NEG_INF_F;
    float best_agreement = 0.0f;
    float best_left_value = 0.0f;
    float best_right_value = 0.0f;
    int32_t best_left_count = node_rows;
    int32_t best_right_count = 0;

    for (int feat_idx = 0; feat_idx < n_features_subset; ++feat_idx) {
        const int slot = node_idx * n_features_subset + feat_idx;
        const int32_t threshold = best_threshold_per_feature[slot];
        const float score = score_per_feature[slot];
        if (threshold < 0 || !isfinite(score)) {
            continue;
        }
        if (score > best_score) {
            best_score = score;
            best_feature_slot = feat_idx;
            best_threshold = threshold;
            best_agreement = agreement_per_feature[slot];
            best_left_value = left_value_per_feature[slot];
            best_right_value = right_value_per_feature[slot];
            best_left_count = left_count_per_feature[slot];
            best_right_count = right_count_per_feature[slot];
        }
    }

    if (best_feature_slot < 0) {
        best_feature_out[node_idx] = -1;
        best_threshold_out[node_idx] = 0;
        score_out[node_idx] = NEG_INF_F;
        agreement_out[node_idx] = 0.0f;
        left_value_out[node_idx] = 0.0f;
        right_value_out[node_idx] = 0.0f;
        left_count_out[node_idx] = node_rows;
        right_count_out[node_idx] = 0;
    } else {
        best_feature_out[node_idx] = feature_subset[best_feature_slot];
        best_threshold_out[node_idx] = best_threshold;
        score_out[node_idx] = best_score;
        agreement_out[node_idx] = best_agreement;
        left_value_out[node_idx] = best_left_value;
        right_value_out[node_idx] = best_right_value;
        left_count_out[node_idx] = best_left_count;
        right_count_out[node_idx] = best_right_count;
    }
}

}  // namespace

HistogramBuffers build_histograms_cuda(
    const uint8_t* bins,
    const float* gradients,
    const float* hessians,
    const int16_t* era_inverse,
    std::size_t n_rows,
    std::size_t n_features,
    int max_bins,
    int n_eras) {

    const std::size_t row_bytes = n_rows * n_features * sizeof(uint8_t);
    const std::size_t vec_bytes = n_rows * sizeof(float);
    const std::size_t era_bytes = n_rows * sizeof(int16_t);
    const std::size_t hist_size = n_features * max_bins * n_eras;
    const std::size_t hist_bytes_f = hist_size * sizeof(float);
    const std::size_t hist_bytes_i = hist_size * sizeof(int32_t);

    uint8_t* d_bins = nullptr;
    float* d_grad = nullptr;
    float* d_hess = nullptr;
    int16_t* d_era = nullptr;
    float* d_out_grad = nullptr;
    float* d_out_hess = nullptr;
    int32_t* d_out_count = nullptr;

    check(cudaMalloc(&d_bins, row_bytes), "cudaMalloc bins");
    check(cudaMalloc(&d_grad, vec_bytes), "cudaMalloc grad");
    check(cudaMalloc(&d_hess, vec_bytes), "cudaMalloc hess");
    check(cudaMalloc(&d_era, era_bytes), "cudaMalloc era");
    check(cudaMalloc(&d_out_grad, hist_bytes_f), "cudaMalloc out_grad");
    check(cudaMalloc(&d_out_hess, hist_bytes_f), "cudaMalloc out_hess");
    check(cudaMalloc(&d_out_count, hist_bytes_i), "cudaMalloc out_count");
    check(cudaMemset(d_out_grad, 0, hist_bytes_f), "cudaMemset out_grad");
    check(cudaMemset(d_out_hess, 0, hist_bytes_f), "cudaMemset out_hess");
    check(cudaMemset(d_out_count, 0, hist_bytes_i), "cudaMemset out_count");

    check(cudaMemcpy(d_bins, bins, row_bytes, cudaMemcpyHostToDevice), "cudaMemcpy bins");
    check(cudaMemcpy(d_grad, gradients, vec_bytes, cudaMemcpyHostToDevice), "cudaMemcpy grad");
    check(cudaMemcpy(d_hess, hessians, vec_bytes, cudaMemcpyHostToDevice), "cudaMemcpy hess");
    check(cudaMemcpy(d_era, era_inverse, era_bytes, cudaMemcpyHostToDevice), "cudaMemcpy era");

    launch_histogram_kernel(
        d_bins,
        d_grad,
        d_hess,
        d_era,
        static_cast<int>(n_rows),
        static_cast<int>(n_features),
        max_bins,
        n_eras,
        d_out_grad,
        d_out_hess,
        d_out_count);

    HistogramBuffers buffers;
    buffers.grad.resize(hist_size);
    buffers.hess.resize(hist_size);
    buffers.count.resize(hist_size);

    check(cudaMemcpy(buffers.grad.data(), d_out_grad, hist_bytes_f, cudaMemcpyDeviceToHost), "cudaMemcpy back grad");
    check(cudaMemcpy(buffers.hess.data(), d_out_hess, hist_bytes_f, cudaMemcpyDeviceToHost), "cudaMemcpy back hess");
    check(cudaMemcpy(buffers.count.data(), d_out_count, hist_bytes_i, cudaMemcpyDeviceToHost), "cudaMemcpy back count");

    cudaFree(d_bins);
    cudaFree(d_grad);
    cudaFree(d_hess);
    cudaFree(d_era);
    cudaFree(d_out_grad);
    cudaFree(d_out_hess);
    cudaFree(d_out_count);

    return buffers;
}

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
    int rows_per_thread) {

    (void)era_group_eras;
    (void)era_tile_size;
    if (threads_per_block <= 0 || threads_per_block > 1024) {
        threads_per_block = 128;
    }
    if (rows_per_thread <= 0) {
        rows_per_thread = 1;
    }

    FrontierEvalResult result;

    result.best_feature.assign(n_nodes, -1);
    result.best_threshold.assign(n_nodes, 0);
    result.score.assign(n_nodes, -std::numeric_limits<float>::infinity());
    result.agreement.assign(n_nodes, 0.0f);
    result.left_value.assign(n_nodes, 0.0f);
    result.right_value.assign(n_nodes, 0.0f);
    result.base_value.assign(n_nodes, 0.0f);
    result.left_offsets.assign(n_nodes + 1, 0);
    result.right_offsets.assign(n_nodes + 1, 0);

    if (n_nodes == 0 || n_features_subset == 0 || max_bins <= 1) {
        return result;
    }

    const int thresholds = std::max(0, max_bins - 1);
    if (thresholds <= 0) {
        return result;
    }

    const int32_t total_indices = node_offsets[n_nodes];
    const int32_t total_era_groups = node_era_offsets[n_nodes];

    const float lambda = static_cast<float>(lambda_l2);
    const float lambda_dro_f = static_cast<float>(lambda_dro);
    const float direction_weight_f = static_cast<float>(direction_weight);
    const float eps = 1e-12f;

    std::vector<void*> allocations;
    auto cleanup = [&]() {
        for (void*& ptr : allocations) {
            if (ptr != nullptr) {
                cudaFree(ptr);
                ptr = nullptr;
            }
        }
    };

    uint8_t* d_bins = nullptr;
    int32_t* d_node_indices = nullptr;
    int32_t* d_node_offsets = nullptr;
    int32_t* d_node_era_offsets = nullptr;
    int32_t* d_era_group_offsets = nullptr;
    int32_t* d_feature_subset = nullptr;
    float* d_gradients = nullptr;
    float* d_hessians = nullptr;

    int32_t* d_best_threshold_per_feature = nullptr;
    float* d_score_per_feature = nullptr;
    float* d_agreement_per_feature = nullptr;
    float* d_left_value_per_feature = nullptr;
    float* d_right_value_per_feature = nullptr;
    int32_t* d_left_count_per_feature = nullptr;
    int32_t* d_right_count_per_feature = nullptr;

    int32_t* d_best_feature = nullptr;
    int32_t* d_best_threshold = nullptr;
    float* d_score = nullptr;
    float* d_agreement = nullptr;
    float* d_left_value = nullptr;
    float* d_right_value = nullptr;
    int32_t* d_left_count = nullptr;
    int32_t* d_right_count = nullptr;

    try {
        auto malloc_and_track = [&](void** ptr, std::size_t bytes, const char* msg) {
            if (bytes == 0) {
                *ptr = nullptr;
                return;
            }
            check(cudaMalloc(ptr, bytes), msg);
            allocations.push_back(*ptr);
        };

        auto memcpy_to_device = [&](void* dst, const void* src, std::size_t bytes, const char* msg) {
            if (bytes == 0) {
                return;
            }
            check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), msg);
        };

        const std::size_t bins_bytes = n_rows * n_features_total * sizeof(uint8_t);
        const std::size_t indices_bytes = static_cast<std::size_t>(total_indices) * sizeof(int32_t);
        const std::size_t node_offsets_bytes = (n_nodes + 1) * sizeof(int32_t);
        const std::size_t node_era_offsets_bytes = (n_nodes + 1) * sizeof(int32_t);
        const std::size_t era_group_offsets_bytes = (static_cast<std::size_t>(total_era_groups) + 1) * sizeof(int32_t);
        const std::size_t feature_subset_bytes = n_features_subset * sizeof(int32_t);
        const std::size_t grad_bytes = n_rows * sizeof(float);
        const std::size_t hess_bytes = n_rows * sizeof(float);

        malloc_and_track(reinterpret_cast<void**>(&d_bins), bins_bytes, "cudaMalloc bins");
        malloc_and_track(reinterpret_cast<void**>(&d_node_indices), indices_bytes, "cudaMalloc node_indices");
        malloc_and_track(reinterpret_cast<void**>(&d_node_offsets), node_offsets_bytes, "cudaMalloc node_offsets");
        malloc_and_track(reinterpret_cast<void**>(&d_node_era_offsets), node_era_offsets_bytes, "cudaMalloc node_era_offsets");
        malloc_and_track(reinterpret_cast<void**>(&d_era_group_offsets), era_group_offsets_bytes, "cudaMalloc era_group_offsets");
        malloc_and_track(reinterpret_cast<void**>(&d_feature_subset), feature_subset_bytes, "cudaMalloc feature_subset");
        malloc_and_track(reinterpret_cast<void**>(&d_gradients), grad_bytes, "cudaMalloc gradients");
        malloc_and_track(reinterpret_cast<void**>(&d_hessians), hess_bytes, "cudaMalloc hessians");

        memcpy_to_device(d_bins, bins, bins_bytes, "cudaMemcpy bins");
        memcpy_to_device(d_node_indices, node_indices, indices_bytes, "cudaMemcpy node_indices");
        memcpy_to_device(d_node_offsets, node_offsets, node_offsets_bytes, "cudaMemcpy node_offsets");
        memcpy_to_device(d_node_era_offsets, node_era_offsets, node_era_offsets_bytes, "cudaMemcpy node_era_offsets");
        memcpy_to_device(d_era_group_offsets, era_group_offsets, era_group_offsets_bytes, "cudaMemcpy era_group_offsets");
        memcpy_to_device(d_feature_subset, feature_subset, feature_subset_bytes, "cudaMemcpy feature_subset");
        memcpy_to_device(d_gradients, gradients, grad_bytes, "cudaMemcpy gradients");
        memcpy_to_device(d_hessians, hessians, hess_bytes, "cudaMemcpy hessians");

        const std::size_t feature_slots = n_nodes * n_features_subset;
        const std::size_t feature_slot_bytes_i = feature_slots * sizeof(int32_t);
        const std::size_t feature_slot_bytes_f = feature_slots * sizeof(float);

        malloc_and_track(reinterpret_cast<void**>(&d_best_threshold_per_feature), feature_slot_bytes_i, "cudaMalloc best_threshold_per_feature");
        malloc_and_track(reinterpret_cast<void**>(&d_score_per_feature), feature_slot_bytes_f, "cudaMalloc score_per_feature");
        malloc_and_track(reinterpret_cast<void**>(&d_agreement_per_feature), feature_slot_bytes_f, "cudaMalloc agreement_per_feature");
        malloc_and_track(reinterpret_cast<void**>(&d_left_value_per_feature), feature_slot_bytes_f, "cudaMalloc left_value_per_feature");
        malloc_and_track(reinterpret_cast<void**>(&d_right_value_per_feature), feature_slot_bytes_f, "cudaMalloc right_value_per_feature");
        malloc_and_track(reinterpret_cast<void**>(&d_left_count_per_feature), feature_slot_bytes_i, "cudaMalloc left_count_per_feature");
        malloc_and_track(reinterpret_cast<void**>(&d_right_count_per_feature), feature_slot_bytes_i, "cudaMalloc right_count_per_feature");

        const std::size_t node_bytes_i = n_nodes * sizeof(int32_t);
        const std::size_t node_bytes_f = n_nodes * sizeof(float);

        malloc_and_track(reinterpret_cast<void**>(&d_best_feature), node_bytes_i, "cudaMalloc best_feature");
        malloc_and_track(reinterpret_cast<void**>(&d_best_threshold), node_bytes_i, "cudaMalloc best_threshold");
        malloc_and_track(reinterpret_cast<void**>(&d_score), node_bytes_f, "cudaMalloc score");
        malloc_and_track(reinterpret_cast<void**>(&d_agreement), node_bytes_f, "cudaMalloc agreement");
        malloc_and_track(reinterpret_cast<void**>(&d_left_value), node_bytes_f, "cudaMalloc left_value");
        malloc_and_track(reinterpret_cast<void**>(&d_right_value), node_bytes_f, "cudaMalloc right_value");
        malloc_and_track(reinterpret_cast<void**>(&d_left_count), node_bytes_i, "cudaMalloc left_count");
        malloc_and_track(reinterpret_cast<void**>(&d_right_count), node_bytes_i, "cudaMalloc right_count");

        const dim3 grid(static_cast<unsigned int>(n_nodes), static_cast<unsigned int>(n_features_subset), 1);
        const int threads = threads_per_block;
        const std::size_t shared_bytes =
            static_cast<std::size_t>(max_bins) * (2 * sizeof(float) + sizeof(int32_t)) +
            static_cast<std::size_t>(thresholds) * (6 * sizeof(float) + 4 * sizeof(int32_t));

        frontier_feature_kernel<<<grid, threads, shared_bytes>>>(
            d_bins,
            d_node_indices,
            d_node_offsets,
            d_node_era_offsets,
            d_era_group_offsets,
            d_feature_subset,
            d_gradients,
            d_hessians,
            static_cast<int>(n_rows),
            static_cast<int>(n_features_total),
            max_bins,
            thresholds,
            static_cast<int>(n_features_subset),
            n_eras_total,
            rows_per_thread,
            min_samples_leaf,
            lambda,
            lambda_dro_f,
            direction_weight_f,
            eps,
            d_best_threshold_per_feature,
            d_score_per_feature,
            d_agreement_per_feature,
            d_left_value_per_feature,
            d_right_value_per_feature,
            d_left_count_per_feature,
            d_right_count_per_feature);
        check(cudaDeviceSynchronize(), "frontier_feature_kernel launch failed");

        const int threads_select = threads_per_block;
        const int blocks_select = static_cast<int>((n_nodes + threads_select - 1) / threads_select);

        frontier_select_kernel<<<blocks_select, threads_select>>>(
            d_feature_subset,
            static_cast<int>(n_features_subset),
            d_node_offsets,
            static_cast<int>(n_nodes),
            d_best_threshold_per_feature,
            d_score_per_feature,
            d_agreement_per_feature,
            d_left_value_per_feature,
            d_right_value_per_feature,
            d_left_count_per_feature,
            d_right_count_per_feature,
            d_best_feature,
            d_best_threshold,
            d_score,
            d_agreement,
            d_left_value,
            d_right_value,
            d_left_count,
            d_right_count);
        check(cudaDeviceSynchronize(), "frontier_select_kernel launch failed");

        std::vector<int32_t> best_feature_host(n_nodes);
        std::vector<int32_t> best_threshold_host(n_nodes);
        std::vector<float> score_host(n_nodes);
        std::vector<float> agreement_host(n_nodes);
        std::vector<float> left_value_host(n_nodes);
        std::vector<float> right_value_host(n_nodes);
        std::vector<int32_t> left_count_host(n_nodes);
        std::vector<int32_t> right_count_host(n_nodes);

        check(cudaMemcpy(best_feature_host.data(), d_best_feature, node_bytes_i, cudaMemcpyDeviceToHost), "cudaMemcpy best_feature");
        check(cudaMemcpy(best_threshold_host.data(), d_best_threshold, node_bytes_i, cudaMemcpyDeviceToHost), "cudaMemcpy best_threshold");
        check(cudaMemcpy(score_host.data(), d_score, node_bytes_f, cudaMemcpyDeviceToHost), "cudaMemcpy score");
        check(cudaMemcpy(agreement_host.data(), d_agreement, node_bytes_f, cudaMemcpyDeviceToHost), "cudaMemcpy agreement");
        check(cudaMemcpy(left_value_host.data(), d_left_value, node_bytes_f, cudaMemcpyDeviceToHost), "cudaMemcpy left_value");
        check(cudaMemcpy(right_value_host.data(), d_right_value, node_bytes_f, cudaMemcpyDeviceToHost), "cudaMemcpy right_value");
        check(cudaMemcpy(left_count_host.data(), d_left_count, node_bytes_i, cudaMemcpyDeviceToHost), "cudaMemcpy left_count");
        check(cudaMemcpy(right_count_host.data(), d_right_count, node_bytes_i, cudaMemcpyDeviceToHost), "cudaMemcpy right_count");

        cleanup();

        // Compute base values on host
        for (std::size_t node = 0; node < n_nodes; ++node) {
            const int32_t begin = node_offsets[node];
            const int32_t end = node_offsets[node + 1];
            float grad_sum = 0.0f;
            float hess_sum = 0.0f;
            for (int32_t idx = begin; idx < end; ++idx) {
                const int32_t row = node_indices[idx];
                if (row < 0 || static_cast<std::size_t>(row) >= n_rows) {
                    continue;
                }
                grad_sum += gradients[row];
                hess_sum += hessians[row];
            }
            if (end > begin) {
                result.base_value[node] = static_cast<float>(-grad_sum / (hess_sum + lambda + eps));
            } else {
                result.base_value[node] = 0.0f;
            }
        }

        result.best_feature = std::move(best_feature_host);
        result.best_threshold = std::move(best_threshold_host);
        result.score = std::move(score_host);
        result.agreement = std::move(agreement_host);
        result.left_value = std::move(left_value_host);
        result.right_value = std::move(right_value_host);

        std::vector<int32_t> left_counts = std::move(left_count_host);
        std::vector<int32_t> right_counts = std::move(right_count_host);

        result.left_offsets[0] = 0;
        result.right_offsets[0] = 0;
        for (std::size_t node = 0; node < n_nodes; ++node) {
            const int32_t node_rows = node_offsets[node + 1] - node_offsets[node];
            if (result.best_feature[node] < 0) {
                result.score[node] = -std::numeric_limits<float>::infinity();
                result.agreement[node] = 0.0f;
                result.left_value[node] = result.base_value[node];
                result.right_value[node] = result.base_value[node];
                left_counts[node] = node_rows;
                right_counts[node] = 0;
            }
            result.left_offsets[node + 1] = result.left_offsets[node] + std::max<int32_t>(0, left_counts[node]);
            result.right_offsets[node + 1] = result.right_offsets[node] + std::max<int32_t>(0, right_counts[node]);
        }

        result.left_indices.resize(static_cast<std::size_t>(result.left_offsets.back()));
        result.right_indices.resize(static_cast<std::size_t>(result.right_offsets.back()));

        for (std::size_t node = 0; node < n_nodes; ++node) {
            const int32_t begin = node_offsets[node];
            const int32_t end = node_offsets[node + 1];
            int32_t left_pos = result.left_offsets[node];
            int32_t right_pos = result.right_offsets[node];
            const int32_t feature = result.best_feature[node];
            const int32_t threshold = result.best_threshold[node];

            if (feature < 0 || threshold < 0) {
                for (int32_t idx = begin; idx < end; ++idx) {
                    result.left_indices[static_cast<std::size_t>(left_pos++)] = node_indices[idx];
                }
                continue;
            }

            for (int32_t idx = begin; idx < end; ++idx) {
                const int32_t row = node_indices[idx];
                if (row < 0 || static_cast<std::size_t>(row) >= n_rows) {
                    continue;
                }
                const std::size_t offset = static_cast<std::size_t>(row) * n_features_total + static_cast<std::size_t>(feature);
                const uint8_t bin = bins[offset];
                if (bin <= threshold) {
                    result.left_indices[static_cast<std::size_t>(left_pos++)] = row;
                } else {
                    result.right_indices[static_cast<std::size_t>(right_pos++)] = row;
                }
            }
        }

    } catch (...) {
        cleanup();
        throw;
    }

    return result;
}

}  // namespace packboost

py::tuple cuda_histogram_binding(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bins,
    py::array_t<float, py::array::c_style | py::array::forcecast> gradients,
    py::array_t<float, py::array::c_style | py::array::forcecast> hessians,
    py::array_t<int16_t, py::array::c_style | py::array::forcecast> era_inverse,
    int max_bins,
    int n_eras) {

    py::buffer_info bins_info = bins.request();
    py::buffer_info grad_info = gradients.request();
    py::buffer_info hess_info = hessians.request();
    py::buffer_info era_info = era_inverse.request();

    if (bins_info.ndim != 2) {
        throw std::invalid_argument("bins must be 2D");
    }

    const std::size_t n_rows = static_cast<std::size_t>(bins_info.shape[0]);
    const std::size_t n_features = static_cast<std::size_t>(bins_info.shape[1]);

    auto buffers = packboost::build_histograms_cuda(
        static_cast<uint8_t*>(bins_info.ptr),
        static_cast<float*>(grad_info.ptr),
        static_cast<float*>(hess_info.ptr),
        static_cast<int16_t*>(era_info.ptr),
        n_rows,
        n_features,
        max_bins,
        n_eras);

    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(n_features),
        static_cast<py::ssize_t>(max_bins),
        static_cast<py::ssize_t>(n_eras),
    };
    py::array_t<float> grad_arr(shape);
    py::array_t<float> hess_arr(shape);
    py::array_t<int32_t> count_arr(shape);
    std::memcpy(grad_arr.mutable_data(), buffers.grad.data(), buffers.grad.size() * sizeof(float));
    std::memcpy(hess_arr.mutable_data(), buffers.hess.data(), buffers.hess.size() * sizeof(float));
    std::memcpy(count_arr.mutable_data(), buffers.count.data(), buffers.count.size() * sizeof(int32_t));

    return py::make_tuple(std::move(grad_arr), std::move(hess_arr), std::move(count_arr));
}

py::tuple cuda_frontier_evaluate_binding(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bins,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> node_indices,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> node_offsets,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> node_era_offsets,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> era_group_eras,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> era_group_offsets,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> feature_subset,
    py::array_t<float, py::array::c_style | py::array::forcecast> gradients,
    py::array_t<float, py::array::c_style | py::array::forcecast> hessians,
    int max_bins,
    int n_eras_total,
    double lambda_l2,
    double lambda_dro,
    int min_samples_leaf,
    double direction_weight,
    int era_tile_size,
    int threads_per_block,
    int rows_per_thread) {

    py::buffer_info bins_info = bins.request();
    py::buffer_info idx_info = node_indices.request();
    py::buffer_info offsets_info = node_offsets.request();
    py::buffer_info node_era_info = node_era_offsets.request();
    py::buffer_info era_group_info = era_group_eras.request();
    py::buffer_info era_group_offsets_info = era_group_offsets.request();
    py::buffer_info feat_info = feature_subset.request();
    py::buffer_info grad_info = gradients.request();
    py::buffer_info hess_info = hessians.request();

    if (bins_info.ndim != 2) {
        throw std::invalid_argument("bins must be 2D");
    }

    const std::size_t n_rows = static_cast<std::size_t>(bins_info.shape[0]);
    const std::size_t n_features_total = static_cast<std::size_t>(bins_info.shape[1]);
    const std::size_t n_nodes = static_cast<std::size_t>(offsets_info.shape[0] - 1);
    const std::size_t n_features_subset = static_cast<std::size_t>(feat_info.shape[0]);

    auto result = packboost::evaluate_frontier_cuda(
        static_cast<uint8_t*>(bins_info.ptr),
        static_cast<int32_t*>(idx_info.ptr),
        static_cast<int32_t*>(offsets_info.ptr),
        static_cast<int32_t*>(node_era_info.ptr),
        static_cast<int32_t*>(era_group_info.ptr),
        static_cast<int32_t*>(era_group_offsets_info.ptr),
        static_cast<int32_t*>(feat_info.ptr),
        static_cast<float*>(grad_info.ptr),
        static_cast<float*>(hess_info.ptr),
        n_rows,
        n_features_total,
        n_nodes,
        n_features_subset,
        max_bins,
        n_eras_total,
        lambda_l2,
        lambda_dro,
        min_samples_leaf,
        direction_weight,
        era_tile_size,
        threads_per_block,
        rows_per_thread);

    std::vector<py::ssize_t> vec_shape = {static_cast<py::ssize_t>(n_nodes)};
    py::array_t<int32_t> feature_arr(vec_shape);
    py::array_t<int32_t> threshold_arr(vec_shape);
    py::array_t<float> score_arr(vec_shape);
    py::array_t<float> agreement_arr(vec_shape);
    py::array_t<float> left_value_arr(vec_shape);
    py::array_t<float> right_value_arr(vec_shape);
    py::array_t<float> base_value_arr(vec_shape);

    std::vector<py::ssize_t> offset_shape = {static_cast<py::ssize_t>(n_nodes + 1)};
    py::array_t<int32_t> left_offsets_arr(offset_shape);
    py::array_t<int32_t> right_offsets_arr(offset_shape);

    std::vector<py::ssize_t> left_idx_shape = {static_cast<py::ssize_t>(result.left_indices.size())};
    std::vector<py::ssize_t> right_idx_shape = {static_cast<py::ssize_t>(result.right_indices.size())};
    py::array_t<int32_t> left_indices_arr(left_idx_shape);
    py::array_t<int32_t> right_indices_arr(right_idx_shape);

    std::memcpy(feature_arr.mutable_data(), result.best_feature.data(), n_nodes * sizeof(int32_t));
    std::memcpy(threshold_arr.mutable_data(), result.best_threshold.data(), n_nodes * sizeof(int32_t));
    std::memcpy(score_arr.mutable_data(), result.score.data(), n_nodes * sizeof(float));
    std::memcpy(agreement_arr.mutable_data(), result.agreement.data(), n_nodes * sizeof(float));
    std::memcpy(left_value_arr.mutable_data(), result.left_value.data(), n_nodes * sizeof(float));
    std::memcpy(right_value_arr.mutable_data(), result.right_value.data(), n_nodes * sizeof(float));
    std::memcpy(base_value_arr.mutable_data(), result.base_value.data(), n_nodes * sizeof(float));
    std::memcpy(left_offsets_arr.mutable_data(), result.left_offsets.data(), (n_nodes + 1) * sizeof(int32_t));
    std::memcpy(right_offsets_arr.mutable_data(), result.right_offsets.data(), (n_nodes + 1) * sizeof(int32_t));
    if (!result.left_indices.empty()) {
        std::memcpy(left_indices_arr.mutable_data(), result.left_indices.data(), result.left_indices.size() * sizeof(int32_t));
    }
    if (!result.right_indices.empty()) {
        std::memcpy(right_indices_arr.mutable_data(), result.right_indices.data(), result.right_indices.size() * sizeof(int32_t));
    }

    return py::make_tuple(
        feature_arr,
        threshold_arr,
        score_arr,
        agreement_arr,
        left_value_arr,
        right_value_arr,
        base_value_arr,
        left_offsets_arr,
        right_offsets_arr,
        left_indices_arr,
        right_indices_arr);
}
