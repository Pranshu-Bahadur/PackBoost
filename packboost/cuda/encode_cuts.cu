#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Murky-accurate encode_cuts with shared-memory rotate, strides fixed at 64.
//
// OPTIMIZED: Now accepts X in original [N, F] layout to avoid 
// the massive memory spike from X.transpose(0,1).contiguous().
//
// Shapes:
//   API X: [N, F] int8 (Row-major usually, but we use strides)
//   Output dXB: [4*F, M] uint32, where M = ceil_div(N, 32)
//
// Mapping:
//   For feature f and word w (covering samples i=32*w..32*w+31),
//   bit k in XB[4*f + t, w] = 1  iff  (uint8)X[f, 32*w + k] > t,  t∈{0,1,2,3}.

__global__ void _encode_cuts(
    const int8_t* __restrict__ X,  // Input pointer
    uint32_t* __restrict__ XB,     // [4*F, M]
    int F, int N, int stride,      // stride = ceil(N / (32*32*strides))
    int stride_n, int stride_f)    // Strides for input X (N-dim, F-dim)
{
    const int f  = blockIdx.x;       // feature index
    const int bi = blockIdx.y;       // tile block along N (words grouped by 32)
    const int wi = threadIdx.x;      // lane 0..31

    if (f >= F || wi >= 32 || blockDim.x != 32) return;

    // Shared memory tile and rotate, as in Murky
    __shared__ uint32_t sm[32][32];  // tile[row, col]

    // Each block.y iterates over 'stride' tiles
    for (int i = 0; i < stride; ++i) {
        // Input/sample side indexing
        // i_in is the base column within the 32x32 tile for this lane
        const int i_in  = 32*32*stride*bi + 32*32*i + wi;   // sample-column base for this lane
        const int i_out =    32*stride*bi +    32*i + wi;   // output word index for this lane

        // Accumulators for 4 bit-planes
        uint32_t v0 = 0u, v1 = 0u, v2 = 0u, v3 = 0u;

        // 1) Load a 32x32 tile into shared with rotated column index (k+wi)%32
        //    Source: X[sample, feature] -> X[col, f]
        for (int k = 0; k < 32; ++k) {
            const int col = 32*k + i_in; // This is the Sample index (row in X)
            uint32_t v = 0u;
            if (col < N) {
                // Direct strided access: X[col, f]
                // X is usually [N, F], so we use stride_n for 'col' and stride_f for 'f'
                const size_t idx = (size_t)col * (size_t)stride_n + (size_t)f * (size_t)stride_f;
                v = (uint32_t)(uint8_t)X[idx]; // unsigned semantics
            }
            sm[wi][(k + wi) & 31] = v;
        }
        __syncwarp();

        // 2) Read back rotated to realize the 32x32 transpose logic
        for (int k = 0; k < 32; ++k) {
            const uint32_t v = sm[k][(k + wi) & 31];
            v0 |= ((v > 0u) ? 1u : 0u) << k;
            v1 |= ((v > 1u) ? 1u : 0u) << k;
            v2 |= ((v > 2u) ? 1u : 0u) << k;
            v3 |= ((v > 3u) ? 1u : 0u) << k;
        }

        // 3) Store the four bit-planes for this feature/word (guard tail)
        if (i_out < ( (N + 31) >> 5 )) { // M = ceil_div(N,32)
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

// Host API: Compute directly on X without allocating a transposed copy.

torch::Tensor encode_cuts(torch::Tensor X /* [N,F] int8 */) {
    // We do NOT transpose X. We read it as-is.
    // X is expected to be [N, F]
    const int N = (int)X.size(0);
    const int F = (int)X.size(1);
    
    // Get strides to pass to kernel
    const int stride_n = (int)X.stride(0);
    const int stride_f = (int)X.stride(1);

    const int M = (N + 31) >> 5;  // words

    auto dXB = torch::empty({ (long long)4 * F, (long long)M },
                             X.options().dtype(torch::kUInt32));

    // Keep strides fixed at 64 (as requested)
    const int strides = 64;                   // grid.y
    // Murky's stride: ceil( N / (32^2 * strides) )
    int stride = (N + (32*32*strides) - 1) / (32*32*strides);
    if (stride < 1) stride = 1;

    dim3 grid(F, strides, 1);
    dim3 block(32, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    _encode_cuts<<<grid, block, 0, stream.stream()>>>(
        X.data_ptr<int8_t>(), 
        dXB.data_ptr<uint32_t>(), 
        F, N, stride, 
        stride_n, stride_f);

    return dXB;
}
