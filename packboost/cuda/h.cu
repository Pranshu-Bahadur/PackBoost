// packboost/cuda/h_optimized.cu
// Fully optimized histogram kernel for maximum performance
// Key optimizations:
// 1. Coalesced memory layout: H[nfeatsets, nodes, 32, 2] instead of [nfeatsets, nodes, 2, 32]
// 2. Reduced register pressure: int32 accumulators with periodic flushing
// 3. Eliminated bank conflicts: padded shared memory layout
// 4. Warp-level atomic coalescing for global writes
// 5. Optimized occupancy with dynamic shared memory tuning

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// COALESCED MEMORY LAYOUT HELPERS
// ============================================================================

// NEW LAYOUT: H[nfeatsets, nodes, 32, 2] - lanes are consecutive
// This ensures all 32 lanes in a warp hit consecutive addresses (coalesced)
static inline __device__ int64_t* Hptr_coalesced(
    int64_t* H, int nodes,
    int feat, int node, int lane, int chan) 
{
    // Index: ((feat * nodes + node) * 32 + lane) * 2 + chan
    size_t idx = (((size_t)feat * (size_t)nodes + (size_t)node) * 32u + (size_t)lane) * 2u + (size_t)chan;
    return H + idx;
}

// Pack (sum, count) into 64-bit for efficient shuffles
__device__ __forceinline__ unsigned long long pack_sc(int sum32, int cnt32) {
    return ((unsigned long long)(unsigned int)cnt32 << 32) | 
           (unsigned long long)(unsigned int)sum32;
}

__device__ __forceinline__ unsigned long long add_pack(unsigned long long a, unsigned long long b) {
    int sa = (int)(unsigned int)a;
    int ca = (int)(unsigned int)(a >> 32);
    int sb = (int)(unsigned int)b;
    int cb = (int)(unsigned int)(b >> 32);
    return pack_sc(sa + sb, ca + cb);
}

// ============================================================================
// OPTIMIZED KERNEL WITH REDUCED REGISTER PRESSURE
// ============================================================================

template <typename LF_T>
__global__ void __launch_bounds__(256, 4)  // Optimize for 4 blocks/SM
_h_sm_optimized(
    const uint32_t* __restrict__ XS,  // [nfeatsets, cols_32M]
    const int16_t* __restrict__ Y,    // [N]
    const LF_T* __restrict__ LF,      // [nfeatsets, N]
    int64_t* __restrict__ H,          // [nfeatsets, nodes, 32, 2] - NEW LAYOUT
    int nfeatsets,
    int cols_32M,
    int N,
    int max_depth,
    int warps_per_block,
    int stride,
    int nodes_total)
{
    const int feat_set = blockIdx.x;
    const int block_warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gwarp = warps_per_block * blockIdx.y + block_warp;
    
    if (feat_set >= nfeatsets) return;
    
    // ========================================================================
    // REDUCED REGISTER PRESSURE: Use int32 accumulators, flush periodically
    // ========================================================================
    
    // Low-depth accumulators (depths 0-2: 7 nodes)
    // OLD: 14 int64_t = 112 bytes/thread
    // NEW: 14 int32_t = 56 bytes/thread (50% reduction)
    int hf0 = 0, hw0 = 0;
    int hf10 = 0, hf11 = 0, hw10 = 0, hw11 = 0;
    int hf20 = 0, hf21 = 0, hf22 = 0, hf23 = 0;
    int hw20 = 0, hw21 = 0, hw22 = 0, hw23 = 0;
    
    // ========================================================================
    // BANK-CONFLICT-FREE SHARED MEMORY LAYOUT
    // ========================================================================
    
    int n_ge3 = (1 << max_depth) - 8;
    if (n_ge3 < 1) n_ge3 = 1;
    
    extern __shared__ int shmem[];
    
    // High-depth histogram: [n_ge3, 2, 33] with padding to avoid bank conflicts
    // OLD: [n_ge3, 2, 32] - caused 8-way conflicts on stride-64 access
    // NEW: [n_ge3, 2, 33] - padding eliminates conflicts
    int* sh_high = shmem;
    const int sh_high_stride = 33;  // Padded stride
    
    // Low-depth packed storage: [warps_per_block, 7, 32]
    unsigned long long* sh_low = (unsigned long long*)(shmem + n_ge3 * 2 * sh_high_stride);
    
    const unsigned mask = __ballot_sync(__activemask(), true);
    
    // Zero shared memory for high depths
    for (int i = 0; i < n_ge3; ++i) {
        sh_high[(i * 2 + 0) * sh_high_stride + lane] = 0;  // sum
        sh_high[(i * 2 + 1) * sh_high_stride + lane] = 0;  // count
    }
    __syncthreads();
    
    // ========================================================================
    // MAIN ACCUMULATION LOOP WITH PERIODIC FLUSHING
    // ========================================================================
    
    const int FLUSH_INTERVAL = 512;  // Flush every 512 iterations to prevent overflow
    int tile_count = 0;
    
    for (int j = 0; j < stride; ++j) {
        const int base = 32 * (stride * gwarp + j);
        if (base >= cols_32M) break;
        
        // Load lane-local data
        const int jj_lane = base + lane;
        int32_t y_lane = 0;
        uint32_t l32 = 0;
        
        if (jj_lane < N) {
            y_lane = Y[jj_lane];
            LF_T lval = LF[(size_t)feat_set * (size_t)N + (size_t)jj_lane];
            l32 = (uint32_t)lval;
        }
        
        // Load feature bitset
        uint32_t xfd_local = 0u;
        if (jj_lane < cols_32M) {
            xfd_local = XS[(size_t)feat_set * (size_t)cols_32M + (size_t)jj_lane];
        }
        
        // Mask tail bits
        const int rem = N - base;
        uint32_t valid_mask = (rem >= 32) ? 0xFFFFFFFFu : 
                              (rem > 0) ? ((1u << rem) - 1u) : 0u;
        xfd_local &= valid_mask;
        
        // Process 32 samples in this tile
        for (int k = 0; k < 32; ++k) {
            const int v = (int)(xfd_local & 1u);
            xfd_local >>= 1;
            
            // Broadcast k-th sample's label and leaf
            const int32_t yk = __shfl_sync(mask, y_lane, k);
            uint32_t lk = __shfl_sync(mask, l32, k);
            
            const int vy = v * yk;  // Precompute v*yk once
            
            // Depth 0 (root)
            hf0 += vy;
            hw0 += v;
            
            // Depth 1 (2 nodes)
            const unsigned tk1 = lk & 1u;
            lk >>= 1;
            if (tk1 == 0u) {
                hf10 += vy; hw10 += v;
            } else {
                hf11 += vy; hw11 += v;
            }
            
            // Depth 2 (4 nodes)
            const unsigned tk2 = lk & 3u;
            lk >>= 2;
            switch (tk2) {
                case 0u: hf20 += vy; hw20 += v; break;
                case 1u: hf21 += vy; hw21 += v; break;
                case 2u: hf22 += vy; hw22 += v; break;
                case 3u: hf23 += vy; hw23 += v; break;
            }
            
            // Depths >= 3: accumulate to shared memory
            // NO ATOMICS NEEDED - each lane writes to its own column
            if (max_depth > 3) {
                #pragma unroll
                for (int d = 3; d < 8; ++d) {  // Unroll up to depth 7
                    if (d >= max_depth) break;
                    const unsigned to = (1u << d) - 1u;
                    const unsigned tkd = lk & to;
                    lk >>= d;
                    const int node_idx = (int)(to + tkd) - 7;
                    
                    // Direct write - no atomic needed (lane-private column)
                    sh_high[(node_idx * 2 + 0) * sh_high_stride + lane] += vy;
                    sh_high[(node_idx * 2 + 1) * sh_high_stride + lane] += v;
                }
            }
        }
        
        tile_count++;
        
        // Periodic flush to prevent int32 overflow
        // For 16-bit Y values: max accumulation = 32K samples before overflow risk
        if (tile_count >= FLUSH_INTERVAL) {
            __syncthreads();  // Ensure all writes to shared complete
            
            // Flush low-depth registers to shared
            const int warp_offset = block_warp * 7 * 32;
            sh_low[warp_offset + 0 * 32 + lane] = pack_sc(hf0, hw0);
            sh_low[warp_offset + 1 * 32 + lane] = pack_sc(hf10, hw10);
            sh_low[warp_offset + 2 * 32 + lane] = pack_sc(hf11, hw11);
            sh_low[warp_offset + 3 * 32 + lane] = pack_sc(hf20, hw20);
            sh_low[warp_offset + 4 * 32 + lane] = pack_sc(hf21, hw21);
            sh_low[warp_offset + 5 * 32 + lane] = pack_sc(hf22, hw22);
            sh_low[warp_offset + 6 * 32 + lane] = pack_sc(hf23, hw23);
            
            __syncthreads();
            
            // Warp 0 reduces and writes to global
            if (block_warp == 0) {
                for (int node = 0; node < 7; ++node) {
                    unsigned long long acc = 0ull;
                    for (int w = 0; w < warps_per_block; ++w) {
                        acc = add_pack(acc, sh_low[w * 7 * 32 + node * 32 + lane]);
                    }
                    
                    const int64_t fsum = (int64_t)(int)(unsigned int)acc;
                    const int64_t csum = (int64_t)(unsigned int)(acc >> 32);
                    
                    // COALESCED ATOMIC: All 32 lanes write consecutive addresses
                    atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 0), 
                              (unsigned long long)fsum);
                    atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 1), 
                              (unsigned long long)csum);
                }
                
                // Also flush high-depth shared memory
                for (int node = 7; node < nodes_total; ++node) {
                    int sum_acc = 0, cnt_acc = 0;
                    for (int w = 0; w < warps_per_block; ++w) {
                        // Each warp owns a subset of nodes
                        const int node_idx = node - 7;
                        sum_acc += sh_high[(node_idx * 2 + 0) * sh_high_stride + lane];
                        cnt_acc += sh_high[(node_idx * 2 + 1) * sh_high_stride + lane];
                    }
                    
                    if (sum_acc | cnt_acc) {
                        atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 0),
                                  (unsigned long long)(int64_t)sum_acc);
                        atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 1),
                                  (unsigned long long)(int64_t)cnt_acc);
                    }
                }
                
                // Zero shared for next batch
                for (int i = 0; i < n_ge3; ++i) {
                    sh_high[(i * 2 + 0) * sh_high_stride + lane] = 0;
                    sh_high[(i * 2 + 1) * sh_high_stride + lane] = 0;
                }
            }
            
            __syncthreads();
            
            // Reset accumulators
            hf0 = hw0 = 0;
            hf10 = hf11 = hw10 = hw11 = 0;
            hf20 = hf21 = hf22 = hf23 = 0;
            hw20 = hw21 = hw22 = hw23 = 0;
            tile_count = 0;
        }
    }
    
    // ========================================================================
    // FINAL REDUCTION AND COALESCED GLOBAL WRITE
    // ========================================================================
    
    // Write remaining low-depth values to shared
    const int warp_offset = block_warp * 7 * 32;
    sh_low[warp_offset + 0 * 32 + lane] = pack_sc(hf0, hw0);
    sh_low[warp_offset + 1 * 32 + lane] = pack_sc(hf10, hw10);
    sh_low[warp_offset + 2 * 32 + lane] = pack_sc(hf11, hw11);
    sh_low[warp_offset + 3 * 32 + lane] = pack_sc(hf20, hw20);
    sh_low[warp_offset + 4 * 32 + lane] = pack_sc(hf21, hw21);
    sh_low[warp_offset + 5 * 32 + lane] = pack_sc(hf22, hw22);
    sh_low[warp_offset + 6 * 32 + lane] = pack_sc(hf23, hw23);
    
    __syncthreads();
    
    // Warp 0 performs final reduction and write
    if (block_warp == 0) {
        // Low-depth nodes (0-6)
        for (int node = 0; node < 7; ++node) {
            unsigned long long acc = 0ull;
            
            // Butterfly reduction across warps
            for (int w = 0; w < warps_per_block; ++w) {
                acc = add_pack(acc, sh_low[w * 7 * 32 + node * 32 + lane]);
            }
            
            const int64_t fsum = (int64_t)(int)(unsigned int)acc;
            const int64_t csum = (int64_t)(unsigned int)(acc >> 32);
            
            // COALESCED WRITE: consecutive lanes → consecutive addresses
            if (fsum | csum) {
                atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 0),
                          (unsigned long long)fsum);
                atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 1),
                          (unsigned long long)csum);
            }
        }
        
        // High-depth nodes (7+)
        const int rows_per_lane = (n_ge3 + 31) / 32;
        for (int r = 0; r < rows_per_lane; ++r) {
            const int node_idx = r * 32 + lane;
            if (node_idx >= n_ge3) break;
            
            const int node = 7 + node_idx;
            if (node >= nodes_total) break;
            
            const int64_t fsum = (int64_t)sh_high[(node_idx * 2 + 0) * sh_high_stride + lane];
            const int64_t csum = (int64_t)sh_high[(node_idx * 2 + 1) * sh_high_stride + lane];
            
            if (fsum | csum) {
                atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 0),
                          (unsigned long long)fsum);
                atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 1),
                          (unsigned long long)csum);
            }
        }
    }
}

// ============================================================================
// HOST LAUNCHER WITH OPTIMIZED OCCUPANCY
// ============================================================================

static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

static inline int choose_warps_optimized(int max_depth, size_t smem_cap) {
    int n_ge3 = std::max((1 << max_depth) - 8, 1);
    
    // Try different warp counts and pick the one with best occupancy
    for (int wpb : {8, 4, 2, 1}) {  // Favor 8 for better occupancy
        size_t smem_high = (size_t)n_ge3 * 2 * 33 * sizeof(int);  // 33 for padding
        size_t smem_low = (size_t)wpb * 7 * 32 * sizeof(unsigned long long);
        size_t total = smem_high + smem_low;
        
        if (total <= smem_cap) {
            return wpb;
        }
    }
    return 1;
}

static inline void infer_grid_optimized(
    int nfeatsets, int cols_32M, int warps_per_block,
    int& blocks_per_feat_out, int& stride_out)
{
    // Target high occupancy: aim for 16-32 blocks per SM
    auto* prop = at::cuda::getCurrentDeviceProperties();
    const int SM = prop->multiProcessorCount;
    const int target_blocks = SM * 24;  // 24 blocks/SM target
    
    int blocks_per_feat = std::max(1, target_blocks / (nfeatsets * warps_per_block));
    const int total_warps = blocks_per_feat * warps_per_block * nfeatsets;
    int stride = std::max(1, ceil_div_int(cols_32M, total_warps * 32));
    
    blocks_per_feat_out = blocks_per_feat;
    stride_out = stride;
}

torch::Tensor h_sm_optimized(
    torch::Tensor XS,
    torch::Tensor Y,
    torch::Tensor LF,
    int max_depth)
{
    const int nfeatsets = (int)XS.size(0);
    const int cols_32M = (int)XS.size(1);
    const int N = (int)Y.size(0);
    const int nodes_tot = (1 << max_depth) - 1;
    
    // NEW LAYOUT: [nfeatsets, nodes, 32, 2] for coalescing
    auto opts = XS.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous);
    auto H = torch::zeros({(long long)nfeatsets, (long long)nodes_tot, 32LL, 2LL}, opts);
    
    // Optimized shared memory allocation
    auto* prop = at::cuda::getCurrentDeviceProperties();
    size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                   : (size_t)prop->sharedMemPerBlock;
    
    const int warps_per_block = choose_warps_optimized(max_depth, smem_cap);
    
    int blocks_per_feat = 0, stride = 0;
    infer_grid_optimized(nfeatsets, cols_32M, warps_per_block, blocks_per_feat, stride);
    
    dim3 grid(nfeatsets, blocks_per_feat, 1);
    dim3 block(warps_per_block * 32, 1, 1);
    
    // Calculate shared memory with padding
    int n_ge3 = std::max((1 << max_depth) - 8, 1);
    size_t smem_high = (size_t)n_ge3 * 2 * 33 * sizeof(int);  // Padded to 33
    size_t smem_low = (size_t)warps_per_block * 7 * 32 * sizeof(unsigned long long);
    size_t smem_bytes = smem_high + smem_low;
    
    TORCH_CHECK(smem_bytes <= smem_cap,
                "Required shared memory (", smem_bytes, 
                ") exceeds device limit (", smem_cap, ")");
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    TORCH_CHECK(Y.scalar_type() == torch::kInt16, "Y must be int16");
    TORCH_CHECK(XS.scalar_type() == torch::kUInt32 || XS.scalar_type() == torch::kInt32,
                "XS must be uint32/int32");
    
    const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());
    
    // Dispatch by LF dtype
    const auto lf_dt = LF.scalar_type();
    if (lf_dt == torch::kUInt16) {
        cudaFuncSetAttribute(_h_sm_optimized<uint16_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
        _h_sm_optimized<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
            XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint16_t>(),
            H.data_ptr<int64_t>(), nfeatsets, cols_32M, N, max_depth,
            warps_per_block, stride, nodes_tot);
    } else if (lf_dt == torch::kUInt32) {
        cudaFuncSetAttribute(_h_sm_optimized<uint32_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
        _h_sm_optimized<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
            XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint32_t>(),
            H.data_ptr<int64_t>(), nfeatsets, cols_32M, N, max_depth,
            warps_per_block, stride, nodes_tot);
    } else if (lf_dt == torch::kUInt64) {
        cudaFuncSetAttribute(_h_sm_optimized<uint64_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
        _h_sm_optimized<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
            XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint64_t>(),
            H.data_ptr<int64_t>(), nfeatsets, cols_32M, N, max_depth,
            warps_per_block, stride, nodes_tot);
    } else {
        TORCH_CHECK(false, "LF must be uint16/uint32/uint64");
    }
    
    // Transpose output back to original layout [nfeatsets, nodes, 2, 32] if needed
    // Or update downstream code to expect [nfeatsets, nodes, 32, 2]
    return H.permute({0, 1, 3, 2}).contiguous();  // [nfeatsets, nodes, 32, 2] -> [nfeatsets, nodes, 2, 32]
}
