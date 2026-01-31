#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Original GPU kernel (unchanged)
__global__ void _encode_cuts(
    const int8_t* __restrict__ X,  // [F, N]
    uint32_t* __restrict__ XB,     // [4*F, M]
    int F, int N, int stride)
{
    const int f  = blockIdx.x;
    const int bi = blockIdx.y;
    const int wi = threadIdx.x;

    if (f >= F || wi >= 32 || blockDim.x != 32) return;

    __shared__ uint32_t sm[32][32];

    for (int i = 0; i < stride; ++i) {
        const int i_in  = 32*32*stride*bi + 32*32*i + wi;
        const int i_out =    32*stride*bi +    32*i + wi;

        uint32_t v0 = 0u, v1 = 0u, v2 = 0u, v3 = 0u;

        for (int k = 0; k < 32; ++k) {
            const int col = 32*k + i_in;
            uint32_t v = 0u;
            if (col < N) {
                const size_t idx = (size_t)f * (size_t)N + (size_t)col;
                v = (uint32_t)(uint8_t)X[idx];
            }
            sm[wi][(k + wi) & 31] = v;
        }
        __syncwarp();

        for (int k = 0; k < 32; ++k) {
            const uint32_t v = sm[k][(k + wi) & 31];
            v0 |= ((v > 0u) ? 1u : 0u) << k;
            v1 |= ((v > 1u) ? 1u : 0u) << k;
            v2 |= ((v > 2u) ? 1u : 0u) << k;
            v3 |= ((v > 3u) ? 1u : 0u) << k;
        }

        if (i_out < ((N + 31) >> 5)) {
            const size_t M = (size_t)((N + 31) >> 5);
            const size_t base = (size_t)4 * (size_t)f * M + (size_t)i_out;
            XB[base + 0*M] = v0;
            XB[base + 1*M] = v1;
            XB[base + 2*M] = v2;
            XB[base + 3*M] = v3;
        }
        __syncwarp();
    }
}

// CPU fallback implementation
torch::Tensor encode_cuts_cpu(const torch::Tensor& X) {
    const int64_t N = X.size(0);
    const int64_t F = X.size(1);
    const int64_t M = (N + 31) / 32;

    auto XB = torch::zeros({4 * F, M}, X.options().dtype(torch::kUInt32).device(torch::kCPU));
    
    auto X_acc = X.accessor<int8_t, 2>();
    auto XB_acc = XB.accessor<uint32_t, 2>();

    // Process each feature
    for (int64_t f = 0; f < F; ++f) {
        // Process each 32-sample word
        for (int64_t w = 0; w < M; ++w) {
            uint32_t bits[4] = {0, 0, 0, 0};
            
            // Pack 32 samples into 4 bit-planes
            for (int64_t k = 0; k < 32; ++k) {
                int64_t sample_idx = w * 32 + k;
                if (sample_idx < N) {
                    uint8_t val = (uint8_t)X_acc[sample_idx][f];
                    bits[0] |= ((val > 0) ? 1u : 0u) << k;
                    bits[1] |= ((val > 1) ? 1u : 0u) << k;
                    bits[2] |= ((val > 2) ? 1u : 0u) << k;
                    bits[3] |= ((val > 3) ? 1u : 0u) << k;
                }
            }
            
            // Write out the 4 bit-planes
            for (int t = 0; t < 4; ++t) {
                XB_acc[4*f + t][w] = bits[t];
            }
        }
    }
    
    return XB;
}

// Main host API with memory check
torch::Tensor encode_cuts(torch::Tensor X /* [N,F] int8 */) {
    // Check if input is on CUDA
    const bool input_on_cuda = X.is_cuda();
    
    if (!input_on_cuda) {
        // Input is on CPU, use CPU path
        return encode_cuts_cpu(X);
    }

    // Input is on CUDA - check available memory
    const int64_t N = X.size(0);
    const int64_t F = X.size(1);
    const int64_t M = (N + 31) >> 5;
    
    // Calculate required memory
    // - Transposed dX: F * N * sizeof(int8_t)
    // - Output dXB: 4 * F * M * sizeof(uint32_t)
    const size_t dX_bytes = (size_t)F * (size_t)N * sizeof(int8_t);
    const size_t dXB_bytes = (size_t)4 * (size_t)F * (size_t)M * sizeof(uint32_t);
    const size_t total_required = dX_bytes + dXB_bytes;
    
    // Add 20% safety margin for CUDA overhead
    const size_t required_with_margin = (size_t)(total_required * 1.2);
    
    // Query available GPU memory
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    
    if (err != cudaSuccess || free_bytes < required_with_margin) {
        // Not enough GPU memory - fall back to CPU
        TORCH_WARN("Insufficient GPU memory for encode_cuts (need ~",
                   required_with_margin / (1024*1024), " MB, have ",
                   free_bytes / (1024*1024), " MB). Falling back to CPU.");
        
        // Move to CPU, process, and return result on CPU
        // (caller can move back to GPU if needed)
        auto X_cpu = X.cpu();
        return encode_cuts_cpu(X_cpu);
    }
    
    // Sufficient GPU memory - proceed with CUDA path
    auto dX = X.transpose(0, 1).contiguous();
    const int F_i = (int)F;
    const int N_i = (int)N;
    const int M_i = (int)M;

    auto dXB = torch::empty({(int64_t)4 * F, M},
                             dX.options().dtype(torch::kUInt32));

    const int strides = 64;
    int stride = (N_i + (32*32*strides) - 1) / (32*32*strides);
    if (stride < 1) stride = 1;

    dim3 grid(F_i, strides, 1);
    dim3 block(32, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    _encode_cuts<<<grid, block, 0, stream.stream()>>>(
        dX.data_ptr<int8_t>(), dXB.data_ptr<uint32_t>(), F_i, N_i, stride);

    return dXB;
}
