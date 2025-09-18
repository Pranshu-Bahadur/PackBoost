#include "backend.hpp"

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace packboost {

namespace {

constexpr int WARP_SIZE = 32;

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
        static_cast<int32_t*>(era_info.ptr),
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
