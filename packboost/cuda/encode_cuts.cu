#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using at::Tensor;

__global__ void _encode_cuts(
    const int8_t* __restrict__ X,
    uint32_t* __restrict__ XB,
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

torch::Tensor encode_cuts_cpu(const torch::Tensor& X) {
    const int64_t N = X.size(0);
    const int64_t F = X.size(1);
    const int64_t M = (N + 31) / 32;

    auto XB = torch::zeros({4 * F, M}, 
                           torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
    
    auto X_acc = X.accessor<int8_t, 2>();
    auto XB_acc = XB.accessor<uint32_t, 2>();

    for (int64_t f = 0; f < F; ++f) {
        for (int64_t w = 0; w < M; ++w) {
            uint32_t bits[4] = {0, 0, 0, 0};
            
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
            
            for (int t = 0; t < 4; ++t) {
                XB_acc[4*f + t][w] = bits[t];
            }
        }
    }
    
    return XB;
}

torch::Tensor encode_cuts(torch::Tensor X /* [N,F] int8 */) {
    const bool input_on_cuda = X.is_cuda();
    
    if (!input_on_cuda) {
        return encode_cuts_cpu(X);
    }

    const int64_t N = X.size(0);
    const int64_t F = X.size(1);
    const int64_t M = (N + 31) >> 5;
    
    const size_t X_bytes = (size_t)N * (size_t)F * sizeof(int8_t);
    const size_t dX_bytes = (size_t)F * (size_t)N * sizeof(int8_t);
    const size_t dXB_bytes = (size_t)4 * (size_t)F * (size_t)M * sizeof(uint32_t);
    
    size_t total_required = dX_bytes + dXB_bytes;
    if (!X.is_contiguous()) {
        total_required += X_bytes;
    }
    
    const size_t required_with_margin = (size_t)(total_required * 1.2);
    
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    
    if (err != cudaSuccess || free_bytes < required_with_margin) {
        TORCH_WARN("Insufficient GPU memory for encode_cuts (need ~",
                   required_with_margin / (1024*1024), " MB, have ",
                   free_bytes / (1024*1024), " MB). Computing on CPU...");
        
        auto X_cpu = X.cpu();
        auto XB_cpu = encode_cuts_cpu(X_cpu);
        
        const size_t result_bytes = dXB_bytes;
        const size_t result_with_margin = (size_t)(result_bytes * 1.1);
        
        cudaMemGetInfo(&free_bytes, &total_bytes);
        
        if (free_bytes >= result_with_margin) {
            TORCH_WARN("Uploading CPU result to GPU (~", 
                       result_bytes / (1024*1024), " MB)");
            return XB_cpu.to(X.device());
        } else {
            TORCH_CHECK(false, 
                "Cannot fit encode_cuts result on GPU (need ", 
                result_with_margin / (1024*1024), " MB, have ",
                free_bytes / (1024*1024), " MB). ",
                "Solutions:\n",
                "  1. Reduce dataset size\n",
                "  2. Use GPU with more memory\n",
                "  3. Set device='cpu' (entire pipeline, very slow)"
            );
        }
    }
    
    auto X_contig = X.contiguous();
    auto dX = X_contig.transpose(0, 1).contiguous();

    auto dXB = torch::empty({(int64_t)4 * F, M},
                             dX.options().dtype(torch::kUInt32));

    const int strides = 64;
    int stride = ((int)N + (32*32*strides) - 1) / (32*32*strides);
    if (stride < 1) stride = 1;

    dim3 grid((int)F, strides, 1);
    dim3 block(32, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    _encode_cuts<<<grid, block, 0, stream.stream()>>>(
        dX.data_ptr<int8_t>(), 
        dXB.data_ptr<uint32_t>(), 
        (int)F, (int)N, stride);

    return dXB;
}
