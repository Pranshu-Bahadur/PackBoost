// frontier_cuda.cu
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
constexpr int MAX_BINS  = 128;
constexpr int THREADS   = 128;                  // 4 warps per block
constexpr int WARPS     = THREADS / WARP_SIZE;

__device__ constexpr float NEG_INF = -1.0e30f;

__host__ __device__ inline int max_int(int a, int b) { return a > b ? a : b; }
__host__ __device__ inline int min_int(int a, int b) { return a < b ? a : b; }
__host__ __device__ inline int clamp_int(int v, int low, int high) {
    return v < low ? low : (v > high ? high : v);
}

__device__ inline float warp_reduce_sum(float v) {
    for (int offs = WARP_SIZE / 2; offs > 0; offs >>= 1) v += __shfl_down_sync(0xffffffff, v, offs);
    return v;
}
__device__ inline int warp_reduce_sum_int(int v) {
    for (int offs = WARP_SIZE / 2; offs > 0; offs >>= 1) v += __shfl_down_sync(0xffffffff, v, offs);
    return v;
}

// Block-wide Kogge–Stone inclusive scan for <= MAX_BINS elements in shared memory
__device__ inline void block_scan_prefix_int(float* g, float* h, int* c, int n) {
    for (int offset = 1; offset < n; offset <<= 1) {
        __syncthreads();
        int b = threadIdx.x;
        if (b < n) {
            float g_prev = (b >= offset) ? g[b - offset] : 0.0f;
            float h_prev = (b >= offset) ? h[b - offset] : 0.0f;
            int   c_prev = (b >= offset) ? c[b - offset] : 0;
            __syncthreads();
            g[b] += g_prev;
            h[b] += h_prev;
            c[b] += c_prev;
        }
    }
    __syncthreads();
}

__global__ void cuda_find_best_splits_kernel(
    // bins is FEATURE-MAJOR: [num_features, rows_dataset]
    const int8_t* __restrict__ bins_fmajor,
    int rows_dataset,                               // equals bins.shape[0] before transpose
    const float* __restrict__ grad,                 // [rows_total_compact]
    const float* __restrict__ hess,                 // [rows_total_compact]
    const int32_t* __restrict__ rows_index,         // [rows_total_compact] -> original row ids
    const int32_t* __restrict__ node_row_splits,    // [num_nodes + 1] (offsets into rows_index/grad/hess)
    const int32_t* __restrict__ node_era_splits,    // [num_nodes * (num_eras + 1)] (same indexing space)
    const float*   __restrict__ era_weights,        // [num_nodes * num_eras]
    const float*   __restrict__ total_grad_nodes,   // [num_nodes]
    const float*   __restrict__ total_hess_nodes,   // [num_nodes]
    const int64_t* __restrict__ total_count_nodes,  // [num_nodes]
    const int32_t* __restrict__ feature_ids,        // [num_features]
    int num_nodes,
    int num_features,
    int num_bins,
    int num_eras,
    int k_cuts,
    int cut_mode,                                   // 0 even, 1 mass
    int min_samples_leaf,
    float lambda_l2,
    float lambda_dro,
    float direction_weight,
    int rows_total_compact,                         // len(rows_index), also len(grad/hess)
    // outputs
    float*   __restrict__ out_scores,               // [num_nodes * num_features]
    int32_t* __restrict__ out_thresholds,           // [num_nodes * num_features]
    float*   __restrict__ out_left_grad,            // [num_nodes * num_features]
    float*   __restrict__ out_left_hess,            // [num_nodes * num_features]
    int64_t* __restrict__ out_left_count            // [num_nodes * num_features]
) {
    const int node_id      = blockIdx.y;
    const int feature_off  = blockIdx.x;
    const int lane         = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id      = threadIdx.x / WARP_SIZE;

    if (node_id >= num_nodes || feature_off >= num_features) return;

    const int feature_id   = feature_ids[feature_off];
    const int out_index    = node_id * num_features + feature_off;

    if (feature_id < 0 || feature_id >= num_features || num_bins <= 1) {
        if (threadIdx.x == 0) {
            out_scores[out_index]     = NEG_INF;
            out_thresholds[out_index] = -1;
            out_left_grad[out_index]  = 0.f;
            out_left_hess[out_index]  = 0.f;
            out_left_count[out_index] = 0;
        }
        return;
    }

    const int32_t* era_offsets = node_era_splits + node_id * (num_eras + 1);

    int row_start = node_row_splits[node_id];
    int row_end   = node_row_splits[node_id + 1];
    row_start     = max_int(0, row_start);
    row_end       = min_int(rows_total_compact, row_end);
    if (row_start >= row_end) {
        if (threadIdx.x == 0) {
            out_scores[out_index]     = NEG_INF;
            out_thresholds[out_index] = -1;
            out_left_grad[out_index]  = 0.f;
            out_left_hess[out_index]  = 0.f;
            out_left_count[out_index] = 0;
        }
        return;
    }

    const int node_total_rows = row_end - row_start;
    if (node_total_rows < 2 * min_samples_leaf) {
        if (threadIdx.x == 0) {
            out_scores[out_index]     = NEG_INF;
            out_thresholds[out_index] = -1;
            out_left_grad[out_index]  = 0.f;
            out_left_hess[out_index]  = 0.f;
            out_left_count[out_index] = 0;
        }
        return;
    }

    // ---- Shared memory layout -------------------------------------------------
    extern __shared__ unsigned char shared_raw[];
    unsigned char* cursor = shared_raw;

    // Per-warp private histograms (reduce later): [WARPS][num_bins]
    int32_t* cnt_w = reinterpret_cast<int32_t*>(cursor);
    cursor += WARPS * num_bins * sizeof(int32_t);
    float* grd_w = reinterpret_cast<float*>(cursor);
    cursor += WARPS * num_bins * sizeof(float);
    float* hss_w = reinterpret_cast<float*>(cursor);
    cursor += WARPS * num_bins * sizeof(float);

    // Reduced per-era histogram for the block
    int32_t* count_bins = reinterpret_cast<int32_t*>(cursor);
    cursor += num_bins * sizeof(int32_t);
    float*   grad_bins = reinterpret_cast<float*>(cursor);
    cursor += num_bins * sizeof(float);
    float*   hess_bins = reinterpret_cast<float*>(cursor);
    cursor += num_bins * sizeof(float);

    // Total (across eras) histogram for mass-cut selection
    int32_t* count_total = reinterpret_cast<int32_t*>(cursor);
    cursor += num_bins * sizeof(int32_t);

    // Threshold list (<= MAX_BINS-1, but we size to num_bins for simplicity)
    int32_t* thresholds_sh = reinterpret_cast<int32_t*>(cursor);
    cursor += num_bins * sizeof(int32_t);

    // Per-threshold accumulators (DES/Welford)
    float* mean_arr       = reinterpret_cast<float*>(cursor); cursor += num_bins * sizeof(float);
    float* M2_arr         = reinterpret_cast<float*>(cursor); cursor += num_bins * sizeof(float);
    float* weight_arr     = reinterpret_cast<float*>(cursor); cursor += num_bins * sizeof(float);
    float* dir_arr        = reinterpret_cast<float*>(cursor); cursor += num_bins * sizeof(float);
    float* left_grad_arr  = reinterpret_cast<float*>(cursor); cursor += num_bins * sizeof(float);
    float* left_hess_arr  = reinterpret_cast<float*>(cursor); cursor += num_bins * sizeof(float);

    // align to 8 for int64
    uintptr_t aligned_ptr = (reinterpret_cast<uintptr_t>(cursor) + alignof(int64_t) - 1) & ~(alignof(int64_t) - 1);
    int64_t* left_count_arr = reinterpret_cast<int64_t*>(aligned_ptr);
    cursor = reinterpret_cast<unsigned char*>(left_count_arr + num_bins);
    // ---------------------------------------------------------------------------

    const int full_thresholds = max_int(num_bins - 1, 1);
    int num_eval = 0;

    // Zero per-threshold accumulators
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        mean_arr[i]      = 0.f;
        M2_arr[i]        = 0.f;
        weight_arr[i]    = 0.f;
        dir_arr[i]       = 0.f;
        left_grad_arr[i] = 0.f;
        left_hess_arr[i] = 0.f;
        left_count_arr[i]= 0;
    }
    // Zero the all-era total counts (for mass)
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) count_total[i] = 0;
    __syncthreads();

    // ------------------ Prepass (counts only) for mass cuts -------------------
    if (k_cuts > 0 && k_cuts < full_thresholds && cut_mode == 1) {
        for (int b = (threadIdx.x & (WARP_SIZE - 1)); b < num_bins; b += WARP_SIZE) {
            cnt_w[warp_id * num_bins + b] = 0;
        }
        __syncthreads();

        for (int r = row_start + threadIdx.x; r < row_end; r += blockDim.x) {
            const int32_t ridx = rows_index[r];
            const long long off = (long long)feature_id * (long long)rows_dataset + (long long)ridx;
            const int bin = int((unsigned char)bins_fmajor[off]);
            if (bin >= 0 && bin < num_bins) {
                atomicAdd(&cnt_w[warp_id * num_bins + bin], 1);
            }
        }
        __syncthreads();

        for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
            int sum = 0;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) sum += cnt_w[w * num_bins + b];
            count_total[b] = sum;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            int K = k_cuts;
            if (K <= 0 || K >= full_thresholds) {
                num_eval = full_thresholds;
                for (int t = 0; t < num_eval; ++t) thresholds_sh[t] = t;
            } else {
                long long total = 0;
                for (int b = 0; b < num_bins; ++b) total += (long long)count_total[b];

                if (total <= 0) {
                    if (K == 1) { num_eval = 1; thresholds_sh[0] = 0; }
                    else {
                        num_eval = K;
                        double step = double(full_thresholds - 1) / double(K - 1);
                        for (int t = 0; t < K; ++t) {
                            int thr = int(std::round(step * t));
                            thresholds_sh[t] = clamp_int(thr, 0, full_thresholds - 1);
                        }
                    }
                } else {
                    int cand = 0;
                    num_eval = K;
                    for (int i = 0; i < K; ++i) {
                        double alpha  = (K == 1) ? 0.0 : double(i) / double(K - 1);
                        double target = (double)total * (1.0 - 1e-12) * alpha;
                        long long run = 0;
                        int selected = 0;
                        for (int b = 0; b < num_bins; ++b) { run += (long long)count_total[b]; if (run >= target) { selected = b; break; } }
                        int thr = clamp_int(selected - 1, 0, full_thresholds - 1);
                        thresholds_sh[cand++] = thr;
                    }
                    // sort & unique
                    for (int i = 1; i < cand; ++i) {
                        int key = thresholds_sh[i]; int j = i - 1;
                        while (j >= 0 && thresholds_sh[j] > key) { thresholds_sh[j + 1] = thresholds_sh[j]; --j; }
                        thresholds_sh[j + 1] = key;
                    }
                    int uniq = 0;
                    for (int i = 0; i < cand; ++i) if (i == 0 || thresholds_sh[i] != thresholds_sh[i - 1]) thresholds_sh[uniq++] = thresholds_sh[i];
                    while (uniq < K) { thresholds_sh[uniq] = thresholds_sh[uniq - 1]; ++uniq; }
                    num_eval = min_int(uniq, K);
                }
            }
        }
        __syncthreads();
        num_eval = __shfl_sync(0xffffffff, num_eval, 0);
    } else {
        if (threadIdx.x == 0) {
            if (k_cuts <= 0 || k_cuts >= full_thresholds) {
                num_eval = full_thresholds;
                for (int t = 0; t < num_eval; ++t) thresholds_sh[t] = t;
            } else {
                num_eval = k_cuts;
                if (k_cuts == 1) {
                    thresholds_sh[0] = 0;
                } else {
                    double step = double(full_thresholds - 1) / double(k_cuts - 1);
                    for (int t = 0; t < k_cuts; ++t) {
                        int thr = int(std::round(step * t));
                        thresholds_sh[t] = clamp_int(thr, 0, full_thresholds - 1);
                    }
                }
            }
        }
        __syncthreads();
        num_eval = __shfl_sync(0xffffffff, num_eval, 0);
    }

    num_eval = min_int(num_eval, MAX_BINS - 1);  // conservative cap
    if (num_eval <= 0) {
        if (threadIdx.x == 0) {
            out_scores[out_index]     = NEG_INF;
            out_thresholds[out_index] = -1;
            out_left_grad[out_index]  = 0.f;
            out_left_hess[out_index]  = 0.f;
            out_left_count[out_index] = 0;
        }
        return;
    }

    // ----------------------- Main per-era loop (DES) ---------------------------
    for (int era = 0; era < num_eras; ++era) {
        int e_s = max_int(era_offsets[era], row_start);
        int e_e = min_int(era_offsets[era + 1], row_end);
        if (e_s >= e_e) continue;

        // zero per-warp private hists
        for (int b = (threadIdx.x & (WARP_SIZE - 1)); b < num_bins; b += WARP_SIZE) {
            cnt_w[warp_id * num_bins + b] = 0;
            grd_w[warp_id * num_bins + b] = 0.f;
            hss_w[warp_id * num_bins + b] = 0.f;
        }
        __syncthreads();

        // build per-warp histograms
        for (int r = e_s + threadIdx.x; r < e_e; r += blockDim.x) {
            const int32_t ridx = rows_index[r];
            const long long off = (long long)feature_id * (long long)rows_dataset + (long long)ridx; // 64-bit safe
            const int bin = int((unsigned char)bins_fmajor[off]);
            if (bin >= 0 && bin < num_bins) {
                const float g = grad[r];
                const float h = hess[r];
                atomicAdd(&cnt_w[warp_id * num_bins + bin], 1);
                atomicAdd(&grd_w[warp_id * num_bins + bin], g);
                atomicAdd(&hss_w[warp_id * num_bins + bin], h);
            }
        }
        __syncthreads();

        // reduce across warps into count_bins/grad_bins/hess_bins
        for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
            int   c = 0; float g = 0.f, hh = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                c  += cnt_w[w * num_bins + b];
                g  += grd_w[w * num_bins + b];
                hh += hss_w[w * num_bins + b];
            }
            count_bins[b] = c;
            grad_bins[b]  = g;
            hess_bins[b]  = hh;
        }
        __syncthreads();

        // totals for this era
        float total_grad_e = 0.f, total_hess_e = 0.f; int total_count_e = 0;
        for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
            total_grad_e  += grad_bins[b];
            total_hess_e  += hess_bins[b];
            total_count_e += count_bins[b];
        }
        total_grad_e  = warp_reduce_sum(total_grad_e);
        total_hess_e  = warp_reduce_sum(total_hess_e);
        total_count_e = warp_reduce_sum_int(total_count_e);

        float era_weight = 0.f;
        if (lane == 0) era_weight = era_weights[node_id * num_eras + era];
        era_weight = __shfl_sync(0xffffffff, era_weight, 0);

        float parent_gain = 0.f;
        if (lane == 0) parent_gain = 0.5f * (total_grad_e * total_grad_e) / (total_hess_e + lambda_l2);
        parent_gain = __shfl_sync(0xffffffff, parent_gain, 0);

        // prefix scan over bins => left stats at every bin edge b
        block_scan_prefix_int(grad_bins, hess_bins, count_bins, num_bins - 1);

        // one thread per threshold updates Welford accumulators
        if (threadIdx.x < num_eval) {
            const int thr = thresholds_sh[threadIdx.x];
            if (thr >= 0 && thr < (num_bins - 1) && era_weight > 0.f) {
                const int   left_count = count_bins[thr];
                const float left_grad  = grad_bins[thr];
                const float left_hess  = hess_bins[thr];

                const int   right_count = total_count_e - left_count;
                if (left_count > 0 && right_count > 0) {
                    const float right_grad = total_grad_e - left_grad;
                    const float right_hess = total_hess_e - left_hess;

                    const float denomL = left_hess  + lambda_l2;
                    const float denomR = right_hess + lambda_l2;
                    const float gain   = 0.5f * ((left_grad*left_grad)/denomL + (right_grad*right_grad)/denomR) - parent_gain;

                    // Welford
                    const float delta = gain - mean_arr[threadIdx.x];
                    const float new_w = weight_arr[threadIdx.x] + era_weight;
                    const float mean2 = mean_arr[threadIdx.x] + (era_weight / new_w) * delta;
                    const float delta2= gain - mean2;

                    M2_arr       [threadIdx.x] += era_weight * delta * delta2;
                    mean_arr     [threadIdx.x]  = mean2;
                    weight_arr   [threadIdx.x]  = new_w;

                    left_grad_arr[threadIdx.x] += left_grad;
                    left_hess_arr[threadIdx.x] += left_hess;
                    left_count_arr[threadIdx.x] += (int64_t)left_count;

                    if (direction_weight != 0.f) {
                        const float left_val  = -left_grad  / denomL;
                        const float right_val = -right_grad / denomR;
                        dir_arr[threadIdx.x] += era_weight * ((left_val > right_val) ? 1.f : -1.f);
                    }
                }
            }
        }
        __syncthreads();
    } // eras

    // -------------------------- Select best within feature ---------------------
    if (threadIdx.x == 0) {
        float best_score      = NEG_INF;
        int   best_threshold  = -1;
        float best_left_grad  = 0.f;
        float best_left_hess  = 0.f;
        int64_t best_left_cnt = 0;

        const float   total_grad_node  = total_grad_nodes[node_id];
        const float   total_hess_node  = total_hess_nodes[node_id];
        const int64_t total_count_node = total_count_nodes[node_id];

        for (int i = 0; i < num_eval; ++i) {
            const int thr = thresholds_sh[i];
            if (thr < 0 || thr >= num_bins - 1) continue;

            const int64_t lcnt = left_count_arr[i];
            const int64_t rcnt = total_count_node - lcnt;
            if (lcnt < (int64_t)min_samples_leaf || rcnt < (int64_t)min_samples_leaf) continue;

            const float wsum = weight_arr[i];
            const float std  = (wsum > 0.f) ? sqrtf(fmaxf(0.f, M2_arr[i] / wsum)) : 0.f;
            float score      = mean_arr[i] - lambda_dro * std;
            if (direction_weight != 0.f && wsum > 0.f) score += direction_weight * (dir_arr[i] / wsum);

            bool better = (score > best_score);
            if (!better && fabsf(score - best_score) <= 1e-12f) {
                if (lcnt > best_left_cnt) better = true;
                else if (lcnt == best_left_cnt && thr > best_threshold) better = true;
            }
            if (better) {
                best_score      = score;
                best_threshold  = thr;
                best_left_grad  = left_grad_arr[i];
                best_left_hess  = left_hess_arr[i];
                best_left_cnt   = lcnt;
            }
        }

        if (!isfinite(best_score)) {
            best_threshold = -1;
            best_left_grad = 0.f;
            best_left_hess = 0.f;
            best_left_cnt  = 0;
        }

        out_scores     [out_index] = best_score;
        out_thresholds [out_index] = best_threshold;
        out_left_grad  [out_index] = best_left_grad;
        out_left_hess  [out_index] = best_left_hess;
        out_left_count [out_index] = best_left_cnt;
    }
}

// ------------------------------- Host wrapper --------------------------------

py::dict find_best_splits_batched_cuda(
    py::object bins,               // torch.int8 [N, F] (row-major)
    py::object grad,               // torch.float32 [Rcat]
    py::object hess,               // torch.float32 [Rcat]
    py::object rows_index,         // torch.int32   [Rcat]  (concatenated row ids)
    py::object node_row_splits,    // torch.int32   [num_nodes+1]
    py::object node_era_splits,    // torch.int32   [num_nodes, num_eras+1]
    py::object era_weights,        // torch.float32 [num_nodes, num_eras]
    py::object total_grad,         // torch.float32 [num_nodes]
    py::object total_hess,         // torch.float32 [num_nodes]
    py::object total_count,        // torch.int64   [num_nodes]
    py::object feature_ids,        // torch.int32   [num_features]
    int max_bins,
    int k_cuts,
    const std::string& cut_selection,
    float lambda_l2,
    float lambda_dro,
    float direction_weight,
    int min_samples_leaf,
    int rows_total_compact        // == rows_index.shape[0]
) {
    if (max_bins > MAX_BINS) {
        throw std::invalid_argument("CUDA backend supports up to 128 bins per feature.");
    }

    py::module torch = py::module::import("torch");

    // shapes
    py::tuple bins_shape = py::tuple(bins.attr("shape"));   // [N_dataset, F]
    const long long rows_dataset = (long long)py::int_(bins_shape[0]);

    py::tuple node_row_shape = py::tuple(node_row_splits.attr("shape"));
    const int num_nodes = (int)py::int_(node_row_shape[0]) - 1;
    if (num_nodes <= 0) {
        return py::dict("scores"_a=py::none(), "thresholds"_a=py::none(),
                        "left_grad"_a=py::none(), "left_hess"_a=py::none(),
                        "left_count"_a=py::none(), "kernel_ms"_a=0.0);
    }

    py::tuple feature_shape = py::tuple(feature_ids.attr("shape"));
    const int num_features = (int)py::int_(feature_shape[0]);

    const int cut_mode = (cut_selection == "mass") ? 1 : 0;

    // Pointers
    const uintptr_t grad_ptr       = py::int_(grad.attr("data_ptr")());
    const uintptr_t hess_ptr       = py::int_(hess.attr("data_ptr")());
    const uintptr_t rows_idx_ptr   = py::int_(rows_index.attr("data_ptr")());
    const uintptr_t node_row_ptr   = py::int_(node_row_splits.attr("data_ptr")());
    const uintptr_t node_era_ptr   = py::int_(node_era_splits.attr("data_ptr")());
    const uintptr_t era_w_ptr      = py::int_(era_weights.attr("data_ptr")());
    const uintptr_t total_grad_ptr = py::int_(total_grad.attr("data_ptr")());
    const uintptr_t total_hess_ptr = py::int_(total_hess.attr("data_ptr")());
    const uintptr_t total_cnt_ptr  = py::int_(total_count.attr("data_ptr")());
    const uintptr_t feat_ids_ptr   = py::int_(feature_ids.attr("data_ptr")());

    // Build feature-major view once (GPU-side, keeps API unchanged)
    py::object bins_fmajor = bins.attr("t")().attr("contiguous")();
    const uintptr_t bins_fmajor_ptr = py::int_(bins_fmajor.attr("data_ptr")());

    // Output tensors
    py::object device = bins.attr("device");
    auto scores_tensor = torch.attr("empty")(py::make_tuple(num_nodes, num_features),
                                             "device"_a=device, "dtype"_a=torch.attr("float32"));
    auto thresholds_tensor = torch.attr("empty")(py::make_tuple(num_nodes, num_features),
                                                 "device"_a=device, "dtype"_a=torch.attr("int32"));
    auto left_grad_tensor = torch.attr("zeros")(py::make_tuple(num_nodes, num_features),
                                                "device"_a=device, "dtype"_a=torch.attr("float32"));
    auto left_hess_tensor = torch.attr("zeros")(py::make_tuple(num_nodes, num_features),
                                                "device"_a=device, "dtype"_a=torch.attr("float32"));
    auto left_count_tensor = torch.attr("zeros")(py::make_tuple(num_nodes, num_features),
                                                 "device"_a=device, "dtype"_a=torch.attr("int64"));

    scores_tensor.attr("fill_")(-std::numeric_limits<float>::infinity());
    thresholds_tensor.attr("fill_")(-1);

    const uintptr_t scores_ptr  = py::int_(scores_tensor.attr("data_ptr")());
    const uintptr_t thr_ptr     = py::int_(thresholds_tensor.attr("data_ptr")());
    const uintptr_t lgrad_ptr   = py::int_(left_grad_tensor.attr("data_ptr")());
    const uintptr_t lhess_ptr   = py::int_(left_hess_tensor.attr("data_ptr")());
    const uintptr_t lcnt_ptr    = py::int_(left_count_tensor.attr("data_ptr")());

    // Launch config
    dim3 block_dim(THREADS, 1, 1);
    dim3 grid_dim((unsigned)num_features, (unsigned)num_nodes, 1);

    // Shared memory sizing uses host-side max_bins
    const int NB = max_bins;
    size_t shmem =
        (size_t)WARPS * NB * (sizeof(int32_t) + 2*sizeof(float)) +   // per-warp hist
        (size_t)NB * (sizeof(int32_t) + 2*sizeof(float)) +           // reduced per-era (count+grad+hess)
        (size_t)NB * sizeof(int32_t) +                               // count_total
        (size_t)NB * sizeof(int32_t) +                               // thresholds
        (size_t)NB * (6*sizeof(float) + sizeof(int64_t)) +           // per-threshold accums
        (alignof(int64_t) - 1);

    // Time the kernel only
    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventRecord(start_evt));

    // num_eras from era_weights shape
    const int num_eras =
        (int)py::int_(py::tuple(era_weights.attr("shape"))[1]);

    cuda_find_best_splits_kernel<<<grid_dim, block_dim, shmem>>>(
        reinterpret_cast<const int8_t*>(bins_fmajor_ptr),
        (int)rows_dataset,                                       // rows per feature in feature-major
        reinterpret_cast<const float*>(grad_ptr),
        reinterpret_cast<const float*>(hess_ptr),
        reinterpret_cast<const int32_t*>(rows_idx_ptr),
        reinterpret_cast<const int32_t*>(node_row_ptr),
        reinterpret_cast<const int32_t*>(node_era_ptr),
        reinterpret_cast<const float*>(era_w_ptr),
        reinterpret_cast<const float*>(total_grad_ptr),
        reinterpret_cast<const float*>(total_hess_ptr),
        reinterpret_cast<const int64_t*>(total_cnt_ptr),
        reinterpret_cast<const int32_t*>(feat_ids_ptr),
        (int)num_nodes,
        (int)num_features,
        (int)max_bins,                                           // num_bins (<=128)
        (int)num_eras,
        (int)k_cuts,
        (int)cut_mode,
        (int)min_samples_leaf,
        (float)lambda_l2,
        (float)lambda_dro,
        (float)direction_weight,
        (int)rows_total_compact,
        reinterpret_cast<float*>(scores_ptr),
        reinterpret_cast<int32_t*>(thr_ptr),
        reinterpret_cast<float*>(lgrad_ptr),
        reinterpret_cast<float*>(lhess_ptr),
        reinterpret_cast<int64_t*>(lcnt_ptr)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_evt, stop_evt));
    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));

    py::dict result;
    result["scores"]     = scores_tensor;
    result["thresholds"] = thresholds_tensor;
    result["left_grad"]  = left_grad_tensor;
    result["left_hess"]  = left_hess_tensor;
    result["left_count"] = left_count_tensor;
    result["kernel_ms"]  = kernel_ms;
    return result;
}

} // namespace

void register_cuda_frontier(py::module_& m) {
    m.def("_cuda_available", []() { return true; }, "Native CUDA backend is available.");

    // API matches booster.py (rows_index + rows_total_compact present)
    m.def(
        "find_best_splits_batched_cuda",
        &find_best_splits_batched_cuda,
        py::arg("bins"),               // [N,F] int8, row-major
        py::arg("grad"),               // [Rcat]
        py::arg("hess"),               // [Rcat]
        py::arg("rows_index"),         // [Rcat] int32
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
        py::arg("rows_total_compact"),
        "Find best splits for a batch of nodes using the CUDA backend."
    );
}
