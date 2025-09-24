#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace py::literals;

namespace {

#define CUDA_CHECK(expr)                                                                              \
    do {                                                                                              \
        cudaError_t err__ = (expr);                                                                   \
        if (err__ != cudaSuccess) {                                                                   \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__));       \
        }                                                                                             \
    } while (0)

constexpr int WARP_SIZE = 32;
constexpr int MAX_BINS = 128;
__device__ constexpr float NEG_INF = -1.0e30f;

__host__ __device__ inline int max_int(int a, int b) { return a > b ? a : b; }
__host__ __device__ inline int min_int(int a, int b) { return a < b ? a : b; }
__host__ __device__ inline int clamp_int(int v, int low, int high) {
    return v < low ? low : (v > high ? high : v);
}

__device__ inline float warp_reduce_sum(float value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ inline int warp_reduce_sum_int(int value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void cuda_find_best_splits_kernel(
    const int8_t* __restrict__ bins,           // [rows_total, bins_stride]
    int bins_stride,
    const float* __restrict__ grad,            // [rows_total]
    const float* __restrict__ hess,            // [rows_total]
    const int32_t* __restrict__ node_row_splits,   // [num_nodes + 1]
    const int32_t* __restrict__ node_era_splits,   // [num_nodes * (num_eras + 1)]
    const float* __restrict__ era_weights,         // [num_nodes * num_eras]
    const float* __restrict__ total_grad_nodes,    // [num_nodes]
    const float* __restrict__ total_hess_nodes,    // [num_nodes]
    const int64_t* __restrict__ total_count_nodes, // [num_nodes]
    const int32_t* __restrict__ feature_ids,       // [num_features]
    int num_nodes,
    int num_features,
    int num_bins,
    int num_eras,
    int k_cuts,
    int cut_mode,  // 0 = even, 1 = mass
    int min_samples_leaf,
    float lambda_l2,
    float lambda_dro,
    float direction_weight,
    float* __restrict__ out_scores,            // [num_nodes * num_features]
    int32_t* __restrict__ out_thresholds,      // [num_nodes * num_features]
    float* __restrict__ out_left_grad,         // [num_nodes * num_features]
    float* __restrict__ out_left_hess,         // [num_nodes * num_features]
    int64_t* __restrict__ out_left_count       // [num_nodes * num_features]
) {
    int node_id = blockIdx.y;
    int feature_offset = blockIdx.x;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    if (node_id >= num_nodes || feature_offset >= num_features) {
        return;
    }

    int feature_id = feature_ids[feature_offset];
    int out_index = node_id * num_features + feature_offset;

    if (num_bins <= 1) {
        if (lane == 0) {
            out_scores[out_index] = NEG_INF;
            out_thresholds[out_index] = -1;
            out_left_grad[out_index] = 0.0f;
            out_left_hess[out_index] = 0.0f;
            out_left_count[out_index] = 0;
        }
        return;
    }

    const int32_t* era_offsets = node_era_splits + node_id * (num_eras + 1);
    int row_start = node_row_splits[node_id];
    int row_end = node_row_splits[node_id + 1];
    if (row_start < 0) {
        row_start = 0;
    }
    if (row_end > rows_total) {
        row_end = rows_total;
    }
    if (row_start > row_end) {
        row_start = row_end;
    }

    if (feature_id < 0 || feature_id >= bins_stride) {
        if (lane == 0) {
            out_scores[out_index] = NEG_INF;
            out_thresholds[out_index] = -1;
            out_left_grad[out_index] = 0.0f;
            out_left_hess[out_index] = 0.0f;
            out_left_count[out_index] = 0;
        }
        return;
    }

    int node_total_rows = row_end - row_start;

    if (node_total_rows < 2 * min_samples_leaf) {
        if (lane == 0) {
            out_scores[out_index] = NEG_INF;
            out_thresholds[out_index] = -1;
            out_left_grad[out_index] = 0.0f;
            out_left_hess[out_index] = 0.0f;
            out_left_count[out_index] = 0;
        }
        return;
    }

    extern __shared__ unsigned char shared_raw[];
    unsigned char* cursor = shared_raw;
    int32_t* count_bins = reinterpret_cast<int32_t*>(cursor);
    cursor += sizeof(int32_t) * num_bins;
    float* grad_bins = reinterpret_cast<float*>(cursor);
    cursor += sizeof(float) * num_bins;
    float* hess_bins = reinterpret_cast<float*>(cursor);
    cursor += sizeof(float) * num_bins;
    float* mean_arr = reinterpret_cast<float*>(cursor);
    cursor += sizeof(float) * num_bins;
    float* M2_arr = reinterpret_cast<float*>(cursor);
    cursor += sizeof(float) * num_bins;
    float* weight_arr = reinterpret_cast<float*>(cursor);
    cursor += sizeof(float) * num_bins;
    float* dir_arr = reinterpret_cast<float*>(cursor);
    cursor += sizeof(float) * num_bins;
    float* left_grad_arr = reinterpret_cast<float*>(cursor);
    cursor += sizeof(float) * num_bins;
    float* left_hess_arr = reinterpret_cast<float*>(cursor);
    cursor += sizeof(float) * num_bins;
    uintptr_t aligned_ptr = (reinterpret_cast<uintptr_t>(cursor) + alignof(int64_t) - 1) & ~(alignof(int64_t) - 1);
    int64_t* left_count_arr = reinterpret_cast<int64_t*>(aligned_ptr);
    cursor = reinterpret_cast<unsigned char*>(left_count_arr + num_bins);

    int num_thresholds_full = max_int(num_bins - 1, 1);

    int num_eval = 0;
    int thresholds_local[MAX_BINS];

    if (lane == 0) {
        if (k_cuts <= 0 || k_cuts >= num_thresholds_full) {
            num_eval = num_thresholds_full;
            for (int t = 0; t < num_eval; ++t) {
                thresholds_local[t] = t;
            }
        } else if (cut_mode == 0) {  // even
            num_eval = k_cuts;
            if (k_cuts == 1) {
                thresholds_local[0] = 0;
            } else {
                double step = static_cast<double>(num_thresholds_full - 1) / static_cast<double>(k_cuts - 1);
                for (int t = 0; t < k_cuts; ++t) {
                    double raw = step * static_cast<double>(t);
                    int thr = static_cast<int>(raw + 0.5);
                    thr = clamp_int(thr, 0, num_thresholds_full - 1);
                    thresholds_local[t] = thr;
                }
            }
        }
    }
    __syncwarp();
    num_eval = __shfl_sync(0xffffffff, num_eval, 0);

    if (k_cuts > 0 && k_cuts < num_thresholds_full && cut_mode == 1) {
        for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
            count_bins[idx] = 0;
        }
        __syncwarp();

        for (int row = row_start + threadIdx.x; row < row_end; row += blockDim.x) {
            int bin = static_cast<int>(static_cast<unsigned char>(bins[row * bins_stride + feature_id]));
            if (bin >= 0 && bin < num_bins) {
                atomicAdd(&count_bins[bin], 1);
            }
        }
        __syncwarp();

        if (lane == 0) {
            num_eval = 0;
            int total = 0;
            for (int b = 0; b < num_bins; ++b) {
                total += count_bins[b];
            }
            if (total <= 0) {
                if (k_cuts == 1) {
                    thresholds_local[0] = 0;
                    num_eval = 1;
                } else {
                    num_eval = k_cuts;
                    double step = static_cast<double>(num_thresholds_full - 1) / static_cast<double>(k_cuts - 1);
                    for (int t = 0; t < k_cuts; ++t) {
                        double raw = step * static_cast<double>(t);
                        int thr = static_cast<int>(raw + 0.5);
                        thr = clamp_int(thr, 0, num_thresholds_full - 1);
                        thresholds_local[t] = thr;
                    }
                }
            } else {
                double upper = static_cast<double>(total) * (1.0 - 1e-12);
                int cand_count = 0;
                for (int i = 0; i < k_cuts; ++i) {
                    double alpha = (k_cuts == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(k_cuts - 1);
                    double target = upper * alpha;
                    int64_t running = 0;
                    int selected = 0;
                    for (int b = 0; b < num_bins; ++b) {
                        running += count_bins[b];
                        if (running >= target) {
                            selected = b;
                            break;
                        }
                    }
                    int thr = clamp_int(selected - 1, 0, num_thresholds_full - 1);
                    thresholds_local[cand_count++] = thr;
                }
                // sort and deduplicate
                for (int i = 1; i < cand_count; ++i) {
                    int key = thresholds_local[i];
                    int j = i - 1;
                    while (j >= 0 && thresholds_local[j] > key) {
                        thresholds_local[j + 1] = thresholds_local[j];
                        --j;
                    }
                    thresholds_local[j + 1] = key;
                }
                int unique_count = 0;
                for (int i = 0; i < cand_count; ++i) {
                    if (i == 0 || thresholds_local[i] != thresholds_local[i - 1]) {
                        thresholds_local[unique_count++] = thresholds_local[i];
                    }
                }
                while (unique_count < k_cuts) {
                    thresholds_local[unique_count] = thresholds_local[unique_count - 1];
                    ++unique_count;
                }
                num_eval = min_int(unique_count, k_cuts);
            }
        }
        __syncwarp();
        num_eval = __shfl_sync(0xffffffff, num_eval, 0);
    }

    if (lane == 0 && num_eval == 0) {
        num_eval = min_int(num_thresholds_full, MAX_BINS);
        for (int t = 0; t < num_eval; ++t) {
            thresholds_local[t] = t;
        }
    }
    __syncwarp();
    num_eval = __shfl_sync(0xffffffff, num_eval, 0);
    num_eval = min_int(num_eval, MAX_BINS);

    for (int idx = threadIdx.x; idx < num_eval; idx += blockDim.x) {
        mean_arr[idx] = 0.0f;
        M2_arr[idx] = 0.0f;
        weight_arr[idx] = 0.0f;
        dir_arr[idx] = 0.0f;
        left_grad_arr[idx] = 0.0f;
        left_hess_arr[idx] = 0.0f;
        left_count_arr[idx] = 0;
    }
    __syncwarp();

    for (int era = 0; era < num_eras; ++era) {
        int start = era_offsets[era];
        int end = era_offsets[era + 1];
        start = max_int(start, row_start);
        end = min_int(end, row_end);
        if (start >= end) {
            continue;
        }

        for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
            count_bins[b] = 0;
            grad_bins[b] = 0.0f;
            hess_bins[b] = 0.0f;
        }
        __syncwarp();

        for (int row = start + threadIdx.x; row < end; row += blockDim.x) {
            if (row < row_start || row >= row_end) {
                continue;
            }
            int bin = static_cast<int>(static_cast<unsigned char>(bins[row * bins_stride + feature_id]));
            float g = grad[row];
            float h = hess[row];
            if (bin >= 0 && bin < num_bins) {
                atomicAdd(&count_bins[bin], 1);
                atomicAdd(&grad_bins[bin], g);
                atomicAdd(&hess_bins[bin], h);
            }
        }
        __syncwarp();

        float total_grad_e = 0.0f;
        float total_hess_e = 0.0f;
        int total_count_e = 0;
        for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
            total_grad_e += grad_bins[b];
            total_hess_e += hess_bins[b];
            total_count_e += count_bins[b];
        }
        total_grad_e = warp_reduce_sum(total_grad_e);
        total_hess_e = warp_reduce_sum(total_hess_e);
        total_count_e = warp_reduce_sum_int(total_count_e);

        float era_weight = 0.0f;
        if (lane == 0) {
            era_weight = era_weights[node_id * num_eras + era];
        }
        era_weight = __shfl_sync(0xffffffff, era_weight, 0);

        float parent_gain = 0.0f;
        if (lane == 0) {
            parent_gain = 0.5f * (total_grad_e * total_grad_e) / (total_hess_e + lambda_l2);
        }
        parent_gain = __shfl_sync(0xffffffff, parent_gain, 0);

        if (lane == 0) {
            int idx_thr = 0;
            int left_count_running = 0;
            float left_grad_running = 0.0f;
            float left_hess_running = 0.0f;

            for (int b = 0; b < num_bins - 1 && idx_thr < num_eval; ++b) {
                left_count_running += count_bins[b];
                left_grad_running += grad_bins[b];
                left_hess_running += hess_bins[b];

                while (idx_thr < num_eval && thresholds_local[idx_thr] == b) {
                    int left_count = left_count_running;
                    int right_count = total_count_e - left_count;
                    if (left_count > 0 && right_count > 0 && era_weight > 0.0f) {
                        float left_grad = left_grad_running;
                        float right_grad = total_grad_e - left_grad;
                        float left_hess = left_hess_running;
                        float right_hess = total_hess_e - left_hess;
                        float denom_L = left_hess + lambda_l2;
                        float denom_R = right_hess + lambda_l2;
                        float gain = 0.5f * ((left_grad * left_grad) / denom_L + (right_grad * right_grad) / denom_R) - parent_gain;

                        float delta = gain - mean_arr[idx_thr];
                        float new_weight = weight_arr[idx_thr] + era_weight;
                        float mean_new = mean_arr[idx_thr] + (era_weight / new_weight) * delta;
                        float delta2 = gain - mean_new;
                        M2_arr[idx_thr] += era_weight * delta * delta2;
                        mean_arr[idx_thr] = mean_new;
                        weight_arr[idx_thr] = new_weight;
                        left_grad_arr[idx_thr] += left_grad;
                        left_hess_arr[idx_thr] += left_hess;
                        left_count_arr[idx_thr] += static_cast<int64_t>(left_count);
                        if (direction_weight != 0.0f) {
                            float left_val = -left_grad / denom_L;
                            float right_val = -right_grad / denom_R;
                            dir_arr[idx_thr] += era_weight * ((left_val > right_val) ? 1.0f : -1.0f);
                        }
                    }
                    ++idx_thr;
                }
            }
        }
        __syncwarp();
    }

    if (lane == 0) {
        float best_score = NEG_INF;
            int best_threshold = -1;
            float best_left_grad = 0.0f;
            float best_left_hess = 0.0f;
            int64_t best_left_count = 0;

            float total_grad_node = total_grad_nodes[node_id];
            float total_hess_node = total_hess_nodes[node_id];
            int64_t total_count_node = total_count_nodes[node_id];

            for (int idx = 0; idx < num_eval; ++idx) {
                int thr = thresholds_local[idx];
                if (thr < 0 || thr >= num_bins - 1) {
                    continue;
                }
                int64_t left_count = left_count_arr[idx];
                int64_t right_count = total_count_node - left_count;
            if (left_count < min_samples_leaf || right_count < min_samples_leaf) {
                continue;
            }
            float weight_sum = weight_arr[idx];
            float std = (weight_sum > 0.0f) ? sqrtf(fmaxf(0.0f, M2_arr[idx] / weight_sum)) : 0.0f;
            float score = mean_arr[idx] - lambda_dro * std;
            if (direction_weight != 0.0f && weight_sum > 0.0f) {
                score += direction_weight * (dir_arr[idx] / weight_sum);
            }

            bool better = score > best_score;
            if (!better && fabsf(score - best_score) <= 1e-12f) {
                if (left_count > best_left_count) {
                    better = true;
                } else if (left_count == best_left_count) {
                    if (thr > best_threshold) {
                        better = true;
                    }
                }
            }

            if (better) {
                best_score = score;
                best_threshold = thr;
                best_left_grad = left_grad_arr[idx];
                best_left_hess = left_hess_arr[idx];
                best_left_count = left_count;
            }
        }

        if (!std::isfinite(best_score)) {
            best_threshold = -1;
            best_left_grad = 0.0f;
            best_left_hess = 0.0f;
            best_left_count = 0;
        }

        out_scores[out_index] = best_score;
        out_thresholds[out_index] = best_threshold;
        out_left_grad[out_index] = best_left_grad;
        out_left_hess[out_index] = best_left_hess;
        out_left_count[out_index] = best_left_count;
    }
}

py::dict find_best_splits_batched_cuda(
    py::object bins,
    py::object grad,
    py::object hess,
    py::object node_row_splits,
    py::object node_era_splits,
    py::object era_weights,
    py::object total_grad,
    py::object total_hess,
    py::object total_count,
    py::object feature_ids,
    int max_bins,
    int k_cuts,
    const std::string& cut_selection,
    float lambda_l2,
    float lambda_dro,
    float direction_weight,
    int min_samples_leaf
) {
    if (max_bins > MAX_BINS) {
        throw std::invalid_argument("CUDA backend supports up to 128 bins per feature.");
    }

    py::tuple bins_shape = py::tuple(bins.attr("shape"));
    int64_t rows_total = py::int_(bins_shape[0]);
    int64_t bins_stride = py::int_(bins_shape[1]);

    py::tuple node_row_shape = py::tuple(node_row_splits.attr("shape"));
    int64_t num_nodes = py::int_(node_row_shape[0]) - 1;
    if (num_nodes <= 0) {
        return py::dict(
            "scores"_a=py::none(),
            "thresholds"_a=py::none(),
            "left_grad"_a=py::none(),
            "left_hess"_a=py::none(),
            "left_count"_a=py::none(),
            "kernel_ms"_a=0.0
        );
    }

    py::tuple feature_shape = py::tuple(feature_ids.attr("shape"));
    int64_t num_features = py::int_(feature_shape[0]);

    int cut_mode = (cut_selection == "mass") ? 1 : 0;

    uintptr_t bins_ptr_val = py::int_(bins.attr("data_ptr")());
    uintptr_t grad_ptr_val = py::int_(grad.attr("data_ptr")());
    uintptr_t hess_ptr_val = py::int_(hess.attr("data_ptr")());
    uintptr_t node_row_ptr_val = py::int_(node_row_splits.attr("data_ptr")());
    uintptr_t node_era_ptr_val = py::int_(node_era_splits.attr("data_ptr")());
    uintptr_t era_weights_ptr_val = py::int_(era_weights.attr("data_ptr")());
    uintptr_t total_grad_ptr_val = py::int_(total_grad.attr("data_ptr")());
    uintptr_t total_hess_ptr_val = py::int_(total_hess.attr("data_ptr")());
    uintptr_t total_count_ptr_val = py::int_(total_count.attr("data_ptr")());
    uintptr_t feature_ids_ptr_val = py::int_(feature_ids.attr("data_ptr")());

    py::tuple era_shape = py::tuple(era_weights.attr("shape"));
    int64_t num_eras = py::int_(era_shape[1]);

    py::module torch = py::module::import("torch");
    py::object device = bins.attr("device");

    auto scores_tensor = torch.attr("empty")(
        py::make_tuple(num_nodes, num_features),
        "device"_a=device,
        "dtype"_a=torch.attr("float32")
    );
    auto thresholds_tensor = torch.attr("empty")(
        py::make_tuple(num_nodes, num_features),
        "device"_a=device,
        "dtype"_a=torch.attr("int32")
    );
    auto left_grad_tensor = torch.attr("zeros")(
        py::make_tuple(num_nodes, num_features),
        "device"_a=device,
        "dtype"_a=torch.attr("float32")
    );
    auto left_hess_tensor = torch.attr("zeros")(
        py::make_tuple(num_nodes, num_features),
        "device"_a=device,
        "dtype"_a=torch.attr("float32")
    );
    auto left_count_tensor = torch.attr("zeros")(
        py::make_tuple(num_nodes, num_features),
        "device"_a=device,
        "dtype"_a=torch.attr("int64")
    );

    scores_tensor.attr("fill_")(-std::numeric_limits<float>::infinity());
    thresholds_tensor.attr("fill_")(-1);

    uintptr_t scores_ptr_val = py::int_(scores_tensor.attr("data_ptr")());
    uintptr_t thresholds_ptr_val = py::int_(thresholds_tensor.attr("data_ptr")());
    uintptr_t left_grad_ptr_val = py::int_(left_grad_tensor.attr("data_ptr")());
    uintptr_t left_hess_ptr_val = py::int_(left_hess_tensor.attr("data_ptr")());
    uintptr_t left_count_ptr_val = py::int_(left_count_tensor.attr("data_ptr")());

    dim3 block_dim(WARP_SIZE, 1, 1);
    dim3 grid_dim(static_cast<unsigned int>(num_features), static_cast<unsigned int>(num_nodes), 1);
    size_t per_bin_bytes = sizeof(int32_t) + 8 * sizeof(float) + sizeof(int64_t);
    size_t shared_mem = static_cast<size_t>(max_bins) * per_bin_bytes + (alignof(int64_t) - 1);

    cudaEvent_t start_evt;
    cudaEvent_t stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventRecord(start_evt));

    cuda_find_best_splits_kernel<<<grid_dim, block_dim, shared_mem>>>(
        reinterpret_cast<const int8_t*>(bins_ptr_val),
        static_cast<int>(bins_stride),
        reinterpret_cast<const float*>(grad_ptr_val),
        reinterpret_cast<const float*>(hess_ptr_val),
        reinterpret_cast<const int32_t*>(node_row_ptr_val),
        reinterpret_cast<const int32_t*>(node_era_ptr_val),
        reinterpret_cast<const float*>(era_weights_ptr_val),
        reinterpret_cast<const float*>(total_grad_ptr_val),
        reinterpret_cast<const float*>(total_hess_ptr_val),
        reinterpret_cast<const int64_t*>(total_count_ptr_val),
        reinterpret_cast<const int32_t*>(feature_ids_ptr_val),
        static_cast<int>(num_nodes),
        static_cast<int>(num_features),
        max_bins,
        static_cast<int>(num_eras),
        k_cuts,
        cut_mode,
        min_samples_leaf,
        lambda_l2,
        lambda_dro,
        direction_weight,
        reinterpret_cast<float*>(scores_ptr_val),
        reinterpret_cast<int32_t*>(thresholds_ptr_val),
        reinterpret_cast<float*>(left_grad_ptr_val),
        reinterpret_cast<float*>(left_hess_ptr_val),
        reinterpret_cast<int64_t*>(left_count_ptr_val)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_evt, stop_evt));
    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));

    py::dict result;
    result["scores"] = scores_tensor;
    result["thresholds"] = thresholds_tensor;
    result["left_grad"] = left_grad_tensor;
    result["left_hess"] = left_hess_tensor;
    result["left_count"] = left_count_tensor;
    result["kernel_ms"] = kernel_ms;
    return result;
}

}  // namespace

void register_cuda_frontier(py::module_& m) {
    m.def(
        "_cuda_available",
        []() { return true; },
        "Native CUDA backend is available."
    );
    m.def(
        "find_best_splits_batched_cuda",
        &find_best_splits_batched_cuda,
        py::arg("bins"),
        py::arg("grad"),
        py::arg("hess"),
        py::arg("node_row_splits"),
        py::arg("node_era_splits"),
        py::arg("era_weights"),
        py::arg("total_grad"),
        py::arg("total_hess"),
        py::arg("total_count"),
        py::arg("feature_ids"),
        py::arg("max_bins"),
        py::arg("k_cuts"),
        py::arg("cut_selection"),
        py::arg("lambda_l2"),
        py::arg("lambda_dro"),
        py::arg("direction_weight"),
        py::arg("min_samples_leaf"),
        "Find best splits for a batch of nodes using the CUDA backend."
    );
}
