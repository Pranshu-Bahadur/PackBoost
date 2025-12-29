#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// H layout: [nfeatsets, nodes, 2, 32]
// Returns pointer to H[feat, node, chan, lane] as unsigned long long*
static inline __device__ unsigned long long int* Hptr(
    int64_t* H, int nodes, int feat, int node, int chan, int lane) {
  size_t idx = static_cast<size_t>(feat) * nodes * 2 * 32
               + static_cast<size_t>(node) * 2 * 32
               + static_cast<size_t>(chan) * 32
               + static_cast<size_t>(lane);
  return reinterpret_cast<unsigned long long int*>(H + idx);
}

// Pack sum (int32) and count (uint32) into 64-bit
__device__ __forceinline__ unsigned long long pack_sc(int32_t sum32, uint32_t cnt32) {
    return (static_cast<unsigned long long>(cnt32) << 32) |
           static_cast<unsigned long long>(static_cast<uint32_t>(sum32));
}

// Add two packed values
static __device__ __forceinline__ unsigned long long add_pack(unsigned long long a, unsigned long long b) {
    int32_t sa = static_cast<int32_t>(static_cast<uint32_t>(a));
    uint32_t ca = static_cast<uint32_t>(a >> 32);
    int32_t sb = static_cast<int32_t>(static_cast<uint32_t>(b));
    uint32_t cb = static_cast<uint32_t>(b >> 32);
    return pack_sc(sa + sb, ca + cb);
}

template <typename LF_T>
__global__ void _h_sm_2(
    const uint32_t* __restrict__ XS,
    const int16_t* __restrict__ Y,
    const LF_T* __restrict__ LF,
    int64_t* __restrict__ H,
    int nfeatsets,
    int cols_32M,
    int N,
    int max_depth,
    int warps_per_block,
    int stride,
    int nodes_total
) {
    const int feat_set = blockIdx.x;
    const int block_warp = threadIdx.x >> 5;  // warp index within block
    const int lane = threadIdx.x & 31;        // lane within warp
    const int gwarp = warps_per_block * blockIdx.y + block_warp;

    // Accumulators for depths 0, 1, 2 (nodes 0..6)
    int64_t hf0 = 0, hw0 = 0;
    int64_t hf10 = 0, hf11 = 0, hw10 = 0, hw11 = 0;
    int64_t hf20 = 0, hf21 = 0, hf22 = 0, hf23 = 0, hw20 = 0, hw21 = 0, hw22 = 0, hw23 = 0;

    // Shared memory for low-depth reduction: [warps_per_block, 7, 32] packed (ULL)
    extern __shared__ char smem_char[];
    // Align sh_low to 8 bytes
    size_t offset_low = 0; // no sh_high anymore
    unsigned long long* sh_low = reinterpret_cast<unsigned long long*>(smem_char + offset_low);

    // Zero this warp's section in sh_low (only lanes 0-31 matter per node)
    #pragma unroll
    for (int nd = 0; nd < 7; ++nd) {
        sh_low[(block_warp * 7 + nd) * 32 + lane] = 0ULL;
    }
    __syncthreads();

    // Process tiles
    for (int j = 0; j < stride; ++j) {
        const int base = 32 * (stride * gwarp + j);
        if (base >= N) continue;

        int32_t y_lane = 0;
        uint32_t l32 = 0;
        uint32_t xfd_local = 0u;

        const int jj = base + lane;
        if (jj < N) {
            y_lane = static_cast<int32_t>(Y[jj]);
            LF_T lval = LF[static_cast<size_t>(feat_set) * static_cast<size_t>(N) + jj];
            l32 = static_cast<uint32_t>(lval);
            xfd_local = XS[static_cast<size_t>(feat_set) * static_cast<size_t>(cols_32M) + jj];
        }

        // Mask tail
        const int rem = N - base;
        uint32_t valid_mask = (rem >= 32) ? 0xFFFFFFFFu : (rem > 0 ? (1u << rem) - 1u : 0u);
        xfd_local &= valid_mask;

        const unsigned mask = __activemask(); // or __ballot_sync(-1, true)

        for (int k = 0; k < 32; ++k) {
            const int v = static_cast<int>(xfd_local & 1u);
            xfd_local >>= 1;

            const int32_t yk = __shfl_sync(mask, y_lane, k);
            uint32_t lk = __shfl_sync(mask, l32, k);

            const int64_t add = static_cast<int64_t>(v) * static_cast<int64_t>(yk);

            // d = 0 → node 0
            hf0 += add; hw0 += v;

            // d = 1 → nodes 1 (0), 2 (1)
            unsigned tk = lk & 1u; lk >>= 1;
            if (tk == 0u) { hf10 += add; hw10 += v; }
            else          { hf11 += add; hw11 += v; }

            // d = 2 → nodes 3,4,5,6 (tk=0,1,2,3)
            tk = lk & 3u; lk >>= 2;
            if      (tk == 0u) { hf20 += add; hw20 += v; }
            else if (tk == 1u) { hf21 += add; hw21 += v; }
            else if (tk == 2u) { hf22 += add; hw22 += v; }
            else               { hf23 += add; hw23 += v; }

            // d >= 3 → direct global atomic
            for (int d = 3; d < max_depth; ++d) {
                const unsigned width = 1u << d;          // number of nodes at depth d
                const unsigned tkd = lk & (width - 1u);  // path index in [0, width)
                lk >>= d;
                const int node_id = (1 << d) - 1 + tkd;  // heap layout

                if (v && node_id < nodes_total) {
                    atomicAdd(Hptr(H, nodes_total, feat_set, node_id, 0, lane), static_cast<unsigned long long>(yk));
                    atomicAdd(Hptr(H, nodes_total, feat_set, node_id, 1, lane), 1ULL);
                }
            }
        }
    }

    // Write low-depth accumulators to shared (packed)
    sh_low[(block_warp * 7 + 0) * 32 + lane] = pack_sc(static_cast<int32_t>(hf0), static_cast<uint32_t>(hw0));
    sh_low[(block_warp * 7 + 1) * 32 + lane] = pack_sc(static_cast<int32_t>(hf10), static_cast<uint32_t>(hw10));
    sh_low[(block_warp * 7 + 2) * 32 + lane] = pack_sc(static_cast<int32_t>(hf11), static_cast<uint32_t>(hw11));
    sh_low[(block_warp * 7 + 3) * 32 + lane] = pack_sc(static_cast<int32_t>(hf20), static_cast<uint32_t>(hw20));
    sh_low[(block_warp * 7 + 4) * 32 + lane] = pack_sc(static_cast<int32_t>(hf21), static_cast<uint32_t>(hw21));
    sh_low[(block_warp * 7 + 5) * 32 + lane] = pack_sc(static_cast<int32_t>(hf22), static_cast<uint32_t>(hw22));
    sh_low[(block_warp * 7 + 6) * 32 + lane] = pack_sc(static_cast<int32_t>(hf23), static_cast<uint32_t>(hw23));

    __syncthreads();

    // Butterfly reduction across warps (for low-depth nodes)
    int log_wpb = 0;
    for (int tmp = warps_per_block; tmp > 1; tmp >>= 1) ++log_wpb;

    for (int s = 0; s < log_wpb; ++s) {
        __syncthreads();
        const int ofs = 1 << s;
        if ((block_warp & ofs) == 0 && (block_warp + ofs) < warps_per_block) {
            for (int nd = 0; nd < 7; ++nd) {
                const int idx = (block_warp * 7 + nd) * 32 + lane;
                const int idx_p = ((block_warp + ofs) * 7 + nd) * 32 + lane;
                sh_low[idx] = add_pack(sh_low[idx], sh_low[idx_p]);
            }
        }
    }

    __syncthreads();

    // Warp 0 writes reduced low-depth results to global
    if (block_warp == 0) {
        const int low_node_map[7] = {0, 1, 2, 3, 4, 5, 6};
        for (int nd = 0; nd < 7; ++nd) {
            unsigned long long pack = sh_low[nd * 32 + lane];
            int32_t fsum = static_cast<int32_t>(static_cast<uint32_t>(pack));
            uint32_t csum = static_cast<uint32_t>(pack >> 32);
            int node = low_node_map[nd];
            atomicAdd(Hptr(H, nodes_total, feat_set, node, 0, lane), static_cast<unsigned long long>(fsum));
            atomicAdd(Hptr(H, nodes_total, feat_set, node, 1, lane), static_cast<unsigned long long>(csum));
        }
    }
}

// Host helpers
static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

static inline int infer_hist_warps_per_block(int max_depth) {
    const int residual = 7 - max_depth; // 15 - 2 - 6 - max_depth
    TORCH_CHECK(residual >= 0, "max_depth too large (>", 7, ")");
    int lg2 = 4 - std::max(0, residual);
    if (lg2 < 0) lg2 = 0;
    return 1 << lg2; // 16,8,4,2,1
}

static inline void infer_grid_stride(
    int nfeatsets, int cols_32M, int warps_per_block,
    int& blocks_per_feat_out, int& stride_out) {
    const int A100_SCHED = 64 * 103;
    int blocks_per_feat = ceil_div_int(A100_SCHED, warps_per_block);
    blocks_per_feat = ceil_div_int(blocks_per_feat, nfeatsets);
    blocks_per_feat = std::max(blocks_per_feat, 1);
    const int total_warps = blocks_per_feat * warps_per_block;
    int stride = ceil_div_int(cols_32M, total_warps * 32);
    stride = std::max(stride, 1);
    blocks_per_feat_out = blocks_per_feat;
    stride_out = stride;
}

// No more choose_warps_that_fit based on sh_high (since it's gone)
static inline int choose_warps_that_fit_simple(int max_depth, size_t smem_cap) {
    // Only low-depth shared memory: warps_per_block * 7 * 32 * sizeof(ULL)
    int wpb = 16;
    while (wpb > 1) {
        size_t smem_needed = static_cast<size_t>(wpb) * 7 * 32 * sizeof(unsigned long long);
        if (smem_needed <= smem_cap) break;
        wpb >>= 1;
    }
    return std::max(wpb, 1);
}

torch::Tensor h_sm_2(
    torch::Tensor XS,
    torch::Tensor Y,
    torch::Tensor LF,
    int max_depth)
{
    TORCH_CHECK(max_depth >= 1 && max_depth <= 8, "max_depth must be 1..8");
    const int nfeatsets = static_cast<int>(XS.size(0));
    const int cols_32M = static_cast<int>(XS.size(1));
    const int N = static_cast<int>(Y.size(0));
    const int nodes_tot = (1 << max_depth) - 1;

    auto opts = XS.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous);
    auto H = torch::zeros({nfeatsets, nodes_tot, 2, 32}, opts);

    TORCH_CHECK(Y.scalar_type() == torch::kInt16, "Y must be int16");
    TORCH_CHECK(XS.scalar_type() == torch::kUInt32 || XS.scalar_type() == torch::kInt32,
                "XS must be uint32/int32");

    const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());

    auto* prop = at::cuda::getCurrentDeviceProperties();
    size_t smem_cap = prop->sharedMemPerBlockOptin ? prop->sharedMemPerBlockOptin
                                                   : prop->sharedMemPerBlock;

    // Only low-depth shared memory needed now
    int warps_per_block = choose_warps_that_fit_simple(max_depth, smem_cap);
    int blocks_per_feat = 0, stride = 0;
    infer_grid_stride(nfeatsets, cols_32M, warps_per_block, blocks_per_feat, stride);

    dim3 grid(nfeatsets, blocks_per_feat, 1);
    dim3 block(warps_per_block * 32, 1, 1);

    size_t smem_bytes = static_cast<size_t>(warps_per_block) * 7 * 32 * sizeof(unsigned long long);
    TORCH_CHECK(smem_bytes <= smem_cap,
                "Shared memory ", smem_bytes, " > limit ", smem_cap);

    auto stream = at::cuda::getCurrentCUDAStream();
    const auto lf_dt = LF.scalar_type();

    // Set dynamic shared memory limit
    auto set_attr = [&](auto func) {
        cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
    };

    if (lf_dt == torch::kUInt16) {
        set_attr(_h_sm_2<uint16_t>);
        _h_sm_2<uint16_t><<<grid, block, smem_bytes, stream>>>(
            XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint16_t>(),
            H.data_ptr<int64_t>(), nfeatsets, cols_32M, N, max_depth,
            warps_per_block, stride, nodes_tot);
    } else if (lf_dt == torch::kUInt32) {
        set_attr(_h_sm_2<uint32_t>);
        _h_sm_2<uint32_t><<<grid, block, smem_bytes, stream>>>(
            XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint32_t>(),
            H.data_ptr<int64_t>(), nfeatsets, cols_32M, N, max_depth,
            warps_per_block, stride, nodes_tot);
    } else if (lf_dt == torch::kUInt64) {
        set_attr(_h_sm_2<uint64_t>);
        _h_sm_2<uint64_t><<<grid, block, smem_bytes, stream>>>(
            XS_ptr, Y.data_ptr<int16_t>(), reinterpret_cast<const uint64_t*>(LF.data_ptr()),
            H.data_ptr<int64_t>(), nfeatsets, cols_32M, N, max_depth,
            warps_per_block, stride, nodes_tot);
    } else {
        TORCH_CHECK(false, "LF must be uint16/32/64");
    }

    return H;
}
