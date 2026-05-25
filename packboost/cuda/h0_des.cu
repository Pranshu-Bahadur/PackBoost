// h0_des_butterfly.cu (FIXED: correct H0 scatter indexing)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <algorithm>

constexpr int WARP_SIZE = 32;

// --- helpers, identical packing/transposes as your h0.cu ---
__device__ __forceinline__ int SH_idx(int node, int ch, int lane) {
    // shared layout: [nodes, 2, 32] packed as ((node*2 + ch)*32 + lane)
    return (node * 2 + ch) * 32 + lane;
}

static __device__ __forceinline__ std::uint64_t pack_sc(int sum32, int cnt32) {
    return ((std::uint64_t)(std::uint32_t)cnt32 << 32)
         |  (std::uint64_t)(std::uint32_t)sum32;
}

static __device__ __forceinline__ std::uint64_t add_pack(std::uint64_t a, std::uint64_t b) {
    int sa = (int)(std::uint32_t)a;                 // low 32 = sum
    int ca = (int)(std::uint32_t)(a >> 32);         // high 32 = count
    int sb = (int)(std::uint32_t)b;
    int cb = (int)(std::uint32_t)(b >> 32);
    return pack_sc(sa + sb, ca + cb);
}

// 64-bit atomic add via unsigned long long (two's complement safe)
__device__ __forceinline__ void atomicAdd_i64(std::int64_t* addr, std::int64_t val) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(val));
}

static_assert(sizeof(unsigned long long) == 8, "ULL must be 64-bit.");
static_assert(sizeof(std::int64_t)       == 8, "int64_t must be 64-bit.");

template <typename T>
__global__ void _h0_des_butterfly(
    const std::int16_t*  __restrict__ G,         // [N]
    const T*             __restrict__ LE,        // [nfolds, N] (signed/unsigned ok)
    const std::int32_t*  __restrict__ era_ends,  // [n_eras], increasing, last == N (X-space)
    std::int64_t*        __restrict__ H0,        // [nfolds, nodes, n_eras, 2] (int64)
    const int N,
    const int nfolds,
    const int n_eras,
    const int max_depth,
    const int stride_per_warp                    // columns per block / 32
){
    const int tree_set = blockIdx.x;      // 0..nfolds-1
    const int bi       = blockIdx.y;      // 0..strides-1
    const int lane     = threadIdx.x;     // 0..31

    if (tree_set >= nfolds || blockDim.x != 32 || lane >= 32) return;

    const int nodes      = 1 << max_depth;   // last row unused
    const int used_nodes = nodes - 1;

    extern __shared__ int s_hist[]; // [nodes, 2, 32] as int32
    const unsigned mask = __ballot_sync(__activemask(), true);

    // Each warp processes 'stride_per_warp' tiles of 32 consecutive samples.
    for (int j = 0; j < stride_per_warp; ++j) {
        const int first = 32 * (stride_per_warp * bi + j);
        if (first >= N) break;

        int left  = min(32, N - first);  // lanes remaining in this 32-wide chunk
        int start = first;               // current starting sample in the chunk

        // Find starting era for 'start'
        int era_id = 0;
        while (era_id < n_eras && era_ends[era_id] <= start) ++era_id;

        // Split this 32-wide chunk across era boundaries as needed
        while (left > 0 && era_id < n_eras) {
            const int e_cur = (int)era_ends[era_id];
            const int seg   = max(0, min(left, e_cur - start)); // lanes in current era
            if (seg == 0) { ++era_id; continue; }               // exactly at boundary

            // --- zero shared for this segment ---
            for (int n = 0; n < nodes; ++n) {
                s_hist[SH_idx(n, 0, lane)] = 0;  // sum
                s_hist[SH_idx(n, 1, lane)] = 0;  // count
            }
            __syncwarp();

            // --- accumulate only 'seg' lanes for indices [start + lane] ---
            using U = typename std::make_unsigned<T>::type;
            if (lane < seg) {
                const int jj = start + lane;
                U lk = static_cast<U>(LE[(std::size_t)tree_set * N + jj]);
                const int g = (int)G[jj];

                #pragma unroll
                for (int d = 0; d < 32; ++d) {
                    if (d >= max_depth) break;
                    const int to = (1 << d) - 1;
                    const int tk = (d == 0) ? 0 : (int)(lk & (U)to);
                    if (d > 0) lk >>= d;
                    const int node = to + tk;  // 0..used_nodes-1

                    s_hist[SH_idx(node, 0, lane)] += g;   // sum(G)
                    s_hist[SH_idx(node, 1, lane)] += 1;   // count
                }
            }
            __syncwarp();

            // --- butterfly reduce across lanes and scatter to the CORRECT location ---
            for (int k0 = 0; k0 < used_nodes; k0 += 32) {
                // Load this lane's column (32 rows) for BOTH channels
                std::uint64_t P[32];
                int idx = (k0 * 2 + 0) * 32 + lane;  // ch=0 base
                #pragma unroll
                for (int i = 0; i < 32; ++i, idx += 64) { // +64 == 2*32
                    const int node = k0 + i;
                    int ssum = 0, ccnt = 0;
                    if (node < used_nodes) {
                        ssum = s_hist[idx];       // ch=0
                        ccnt = s_hist[idx + 32];  // ch=1
                    }
                    P[i] = pack_sc(ssum, ccnt);
                }

                // 5-stage butterfly over 32 lanes
                #pragma unroll
                for (int stage = 0; stage < 5; ++stage) {
                    const int ofs = 1 << stage;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        const std::uint64_t partner = __shfl_xor_sync(mask, P[i], ofs, 32);
                        P[i] = add_pack(P[i], partner);
                    }
                }

                // Scatter: lane ℓ writes node_out = k0 + ℓ (both channels)
                const int node_out = k0 + lane;
                if (node_out < used_nodes) {
                    const std::uint64_t pack = P[lane];
                    const std::int64_t sum_i64 = (std::int64_t)(int)(std::uint32_t)pack;
                    const std::int64_t cnt_i64 = (std::int64_t)(int)(std::uint32_t)(pack >> 32);

                    if (sum_i64 | cnt_i64) {
                        // ---- FIXED INDEXING ----
                        // Layout is [nfolds, nodes, n_eras, 2] (contiguous)
                        // flat = (((tree_set * nodes + node_out) * n_eras) + era_id) * 2 + ch
                        const std::int64_t node_base = ((std::int64_t)tree_set * nodes + node_out)
                                                     * (std::int64_t)n_eras * 2;
                        const std::int64_t era_off   = ((std::int64_t)era_id) * 2;

                        if (sum_i64) {
                            atomicAdd_i64(H0 + node_base + era_off + 0, sum_i64);
                        }
                        if (cnt_i64) {
                            atomicAdd_i64(H0 + node_base + era_off + 1, cnt_i64);
                        }
                    }
                }
            }
            __syncwarp();

            // advance within this chunk/era
            start += seg;
            left  -= seg;
            if (start >= e_cur) ++era_id;
        }
    }
}

// ---- Host launcher (pick strides like h0_sm_butterfly; NO n_eras in the grid) ----
torch::Tensor h0_des_butterfly(
    torch::Tensor G,          // [N], int16, CUDA
    torch::Tensor LE,         // [nfolds, N], (u)int16/32/64 or signed, CUDA
    torch::Tensor era_ends,   // [n_eras], int32 (CUDA) in X-space
    int max_depth
){
    TORCH_CHECK(G.is_cuda() && LE.is_cuda() && era_ends.is_cuda(),
                "G, LE, era_ends must be CUDA tensors.");
    TORCH_CHECK(G.dim() == 1 && LE.dim() == 2, "G:[N], LE:[nfolds,N].");
    TORCH_CHECK(era_ends.dim() == 1, "era_ends must be 1D.");
    TORCH_CHECK(G.scalar_type() == c10::ScalarType::Short, "G must be int16.");
    TORCH_CHECK(era_ends.scalar_type() == c10::ScalarType::Int,
                "era_ends must be int32 on device.");
    TORCH_CHECK(max_depth > 0, "max_depth must be > 0.");

    const std::int64_t nfolds64 = LE.size(0);
    const std::int64_t N64      = LE.size(1);
    TORCH_CHECK(G.size(0) == N64, "LE.size(1) must equal G.size(0).");
    TORCH_CHECK(nfolds64 <= (std::int64_t)std::numeric_limits<int>::max());
    TORCH_CHECK(N64      <= (std::int64_t)std::numeric_limits<int>::max());
    const int nfolds = (int)nfolds64;
    const int N      = (int)N64;

    const std::int64_t n_eras64 = era_ends.size(0);
    TORCH_CHECK(n_eras64 > 0, "era_ends cannot be empty.");
    TORCH_CHECK(n_eras64 <= (std::int64_t)std::numeric_limits<int>::max());
    const int n_eras = (int)n_eras64;

    auto ee_host = era_ends.to(torch::kCPU);
    TORCH_CHECK(ee_host.data_ptr<std::int32_t>()[n_eras - 1] == N,
                "era_ends last value must equal N.");

    // Pick strides like h0_sm_butterfly
    static constexpr int lanes = 32;
    int SM = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    int target_blocks_per_SM = 32;
    int min_total_blocks     = SM * target_blocks_per_SM;
    int strides              = (min_total_blocks + nfolds - 1) / nfolds;

    static constexpr int min_workload_per_thread = 128;
    std::int64_t max_strides = (N64 + (lanes * min_workload_per_thread) - 1)
                             / (lanes * min_workload_per_thread);

    strides = (int)std::min<std::int64_t>(strides, max_strides);
    if (strides < 32) strides = 32;

    // Each warp/block processes 'stride_per_warp' 32-wide chunks
    const int stride_per_warp = (N + (lanes * strides) - 1) / (lanes * strides);
    const dim3 block(lanes, 1, 1);
    const dim3 grid (nfolds, strides, 1);

    const int nodes = 1 << max_depth;
    auto H0 = torch::zeros({nfolds64, (std::int64_t)nodes, n_eras64, 2},
                           LE.options().dtype(torch::kLong)  // int64_t
                                    .memory_format(c10::MemoryFormat::Contiguous));

    // Dynamic SMEM: [nodes, 2, 32] * sizeof(int)
    const std::size_t smem_bytes = (std::size_t)nodes * 2 * 32 * sizeof(int);
    auto* prop = at::cuda::getCurrentDeviceProperties();
    std::size_t smem_cap = prop->sharedMemPerBlockOptin
                           ? (std::size_t)prop->sharedMemPerBlockOptin
                           : (std::size_t)prop->sharedMemPerBlock;
    // removed for L4
    //TORCH_CHECK(smem_bytes <= smem_cap,
    //    "h0_des_butterfly requires ", smem_bytes, "B shared memory, device allows ", smem_cap, "B.");

    auto stream = at::cuda::getCurrentCUDAStream();
    const auto dt = LE.scalar_type();

    // Launch per LE dtype
    if (dt == c10::ScalarType::UInt16) {
        cudaFuncSetAttribute(_h0_des_butterfly<std::uint16_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_des_butterfly<std::uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<std::int16_t>(),
            LE.data_ptr<std::uint16_t>(),
            era_ends.data_ptr<std::int32_t>(),
            H0.data_ptr<std::int64_t>(),
            N, nfolds, n_eras, max_depth, stride_per_warp);
    } else if (dt == c10::ScalarType::UInt32) {
        cudaFuncSetAttribute(_h0_des_butterfly<std::uint32_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_des_butterfly<std::uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<std::int16_t>(),
            LE.data_ptr<std::uint32_t>(),
            era_ends.data_ptr<std::int32_t>(),
            H0.data_ptr<std::int64_t>(),
            N, nfolds, n_eras, max_depth, stride_per_warp);
    } else if (dt == c10::ScalarType::UInt64) {
        cudaFuncSetAttribute(_h0_des_butterfly<std::uint64_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_des_butterfly<std::uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<std::int16_t>(),
            LE.data_ptr<std::uint64_t>(),
            era_ends.data_ptr<std::int32_t>(),
            H0.data_ptr<std::int64_t>(),
            N, nfolds, n_eras, max_depth, stride_per_warp);
    } else if (dt == c10::ScalarType::Short) {
        cudaFuncSetAttribute(_h0_des_butterfly<std::int16_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_des_butterfly<std::int16_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<std::int16_t>(),
            LE.data_ptr<std::int16_t>(),
            era_ends.data_ptr<std::int32_t>(),
            H0.data_ptr<std::int64_t>(),
            N, nfolds, n_eras, max_depth, stride_per_warp);
    } else if (dt == c10::ScalarType::Int) {
        cudaFuncSetAttribute(_h0_des_butterfly<std::int32_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_des_butterfly<std::int32_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<std::int16_t>(),
            LE.data_ptr<std::int32_t>(),
            era_ends.data_ptr<std::int32_t>(),
            H0.data_ptr<std::int64_t>(),
            N, nfolds, n_eras, max_depth, stride_per_warp);
    } else if (dt == c10::ScalarType::Long) {
        cudaFuncSetAttribute(_h0_des_butterfly<std::int64_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_des_butterfly<std::int64_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<std::int16_t>(),
            LE.data_ptr<std::int64_t>(),
            era_ends.data_ptr<std::int32_t>(),
            H0.data_ptr<std::int64_t>(),
            N, nfolds, n_eras, max_depth, stride_per_warp);
    } else {
        TORCH_CHECK(false, "LE must be one of: (u)int16, (u)int32, (u)int64 (or signed variants).");
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "h0_des_butterfly launch failed: ",
                cudaGetErrorString(cudaGetLastError()));
    return H0;
}
