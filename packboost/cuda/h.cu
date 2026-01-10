// packboost/cuda/h.cu
// Fully optimized histogram kernel for maximum performance
// FIX: Restored missing shared memory zeroing after periodic flush
// FIX: Correct synchronization barriers for multi-warp reduction

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// COALESCED MEMORY LAYOUT HELPERS
// ============================================================================

static inline __device__ int64_t* Hptr_coalesced(
    int64_t* H, int nodes,
    int feat, int node, int lane, int chan) 
{
    // Index: ((feat * nodes + node) * 32 + lane) * 2 + chan
    size_t idx = (((size_t)feat * (size_t)nodes + (size_t)node) * 32u + (size_t)lane) * 2u + (size_t)chan;
    return H + idx;
}

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
// OPTIMIZED KERNEL
// ============================================================================

template <typename LF_T>
__global__ void __launch_bounds__(256, 4)
_h_sm(
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
    int nodes_total,
    int sh_high_stride)
{
    const int feat_set = blockIdx.x;
    const int block_warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gwarp = warps_per_block * blockIdx.y + block_warp;
    
    if (feat_set >= nfeatsets) return;
    
    // Use int64 accumulators for low depths to prevent overflow
    int64_t hf0 = 0, hw0 = 0;
    int64_t hf10 = 0, hf11 = 0, hw10 = 0, hw11 = 0;
    int64_t hf20 = 0, hf21 = 0, hf22 = 0, hf23 = 0;
    int64_t hw20 = 0, hw21 = 0, hw22 = 0, hw23 = 0;
    
    int n_ge3 = (1 << max_depth) - 8;
    if (n_ge3 < 1) n_ge3 = 1;
    
    extern __shared__ int shmem[];
    
    // Each warp gets its own region to avoid atomic contention
    int* sh_high_base = shmem;
    int* my_warp_region = sh_high_base + block_warp * n_ge3 * 2 * sh_high_stride;
    
    // Low-depth packed storage at end of all regions
    size_t high_total_elems = (size_t)warps_per_block * n_ge3 * 2 * sh_high_stride;
    if (high_total_elems % 2 != 0) high_total_elems++;
    unsigned long long* sh_low = (unsigned long long*)(sh_high_base + high_total_elems);
    
    const unsigned mask = __ballot_sync(__activemask(), true);
    
    // Zero my warp's region
    for (int i = 0; i < n_ge3; ++i) {
        my_warp_region[(i * 2 + 0) * sh_high_stride + lane] = 0;
        my_warp_region[(i * 2 + 1) * sh_high_stride + lane] = 0;
    }
    __syncthreads();
    
    const int FLUSH_INTERVAL = 512;
    int tile_count = 0;
    
    for (int j = 0; j < stride; ++j) {
        const int base = 32 * (stride * gwarp + j);
        if (base >= cols_32M) break;
        
        // Data Loading
        const int jj_lane = base + lane;
        int32_t y_lane = 0;
        uint32_t l32 = 0;
        
        if (jj_lane < N) {
            y_lane = Y[jj_lane];
            LF_T lval = LF[(size_t)feat_set * (size_t)N + (size_t)jj_lane];
            l32 = (uint32_t)lval;
        }
        
        uint32_t xfd_local = 0u;
        if (jj_lane < cols_32M) {
            xfd_local = XS[(size_t)feat_set * (size_t)cols_32M + (size_t)jj_lane];
        }
        
        const int rem = N - base;
        if (rem < 32) {
            uint32_t valid_mask = (rem > 0) ? ((1u << rem) - 1u) : 0u;
            xfd_local &= valid_mask;
        }
        
        // Process 32 bits
        for (int k = 0; k < 32; ++k) {
            const int v = (int)(xfd_local & 1u);
            xfd_local >>= 1;
            
            const int32_t yk = __shfl_sync(mask, y_lane, k);
            uint32_t lk = __shfl_sync(mask, l32, k);
            
            const int64_t vy = (int64_t)v * (int64_t)yk;
            
            hf0 += vy; hw0 += v;
            
            const unsigned tk1 = lk & 1u; lk >>= 1;
            if (tk1 == 0u) { hf10 += vy; hw10 += v; }
            else           { hf11 += vy; hw11 += v; }
            
            const unsigned tk2 = lk & 3u; lk >>= 2;
            switch (tk2) {
                case 0u: hf20 += vy; hw20 += v; break;
                case 1u: hf21 += vy; hw21 += v; break;
                case 2u: hf22 += vy; hw22 += v; break;
                case 3u: hf23 += vy; hw23 += v; break;
            }
            
            if (max_depth > 3) {
                #pragma unroll
                for (int d = 3; d < 8; ++d) {
                    if (d >= max_depth) break;
                    const unsigned to = (1u << d) - 1u;
                    const unsigned tkd = lk & to; lk >>= d;
                    const int node_idx = (int)(to + tkd) - 7;
                    
                    // Accumulate to thread's own lane in warp region
                    my_warp_region[(node_idx * 2 + 0) * sh_high_stride + lane] += (int)(v * yk);
                    my_warp_region[(node_idx * 2 + 1) * sh_high_stride + lane] += v;
                }
            }
        }
        
        tile_count++;
        
        if (tile_count >= FLUSH_INTERVAL) {
            // 1. Pack low-depth registers to shared
            const int warp_offset = block_warp * 7 * 32;
            sh_low[warp_offset + 0 * 32 + lane] = pack_sc((int)hf0, (int)hw0);
            sh_low[warp_offset + 1 * 32 + lane] = pack_sc((int)hf10, (int)hw10);
            sh_low[warp_offset + 2 * 32 + lane] = pack_sc((int)hf11, (int)hw11);
            sh_low[warp_offset + 3 * 32 + lane] = pack_sc((int)hf20, (int)hw20);
            sh_low[warp_offset + 4 * 32 + lane] = pack_sc((int)hf21, (int)hw21);
            sh_low[warp_offset + 5 * 32 + lane] = pack_sc((int)hf22, (int)hw22);
            sh_low[warp_offset + 6 * 32 + lane] = pack_sc((int)hf23, (int)hw23);
            
            __syncthreads(); // Wait for everyone to write to shared
            
            // 2. Warp 0 aggregates and flushes to global
            if (block_warp == 0) {
                // Low depths
                for (int node = 0; node < 7; ++node) {
                    unsigned long long acc = 0ull;
                    for (int w = 0; w < warps_per_block; ++w) {
                        acc = add_pack(acc, sh_low[w * 7 * 32 + node * 32 + lane]);
                    }
                    int64_t fsum = (int64_t)(int)(uint32_t)acc;
                    int64_t csum = (int64_t)(uint32_t)(acc >> 32);
                    
                    if (fsum | csum) {
                        atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 0), (unsigned long long)fsum);
                        atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 1), (unsigned long long)csum);
                    }
                }
                
                // High depths - sum across all warps' regions
                for (int node_idx = 0; node_idx < n_ge3; ++node_idx) {
                    int64_t sum_acc = 0, cnt_acc = 0;
                    for (int w = 0; w < warps_per_block; ++w) {
                        int* wr = sh_high_base + w * n_ge3 * 2 * sh_high_stride;
                        sum_acc += (int64_t)wr[(node_idx * 2 + 0) * sh_high_stride + lane];
                        cnt_acc += (int64_t)wr[(node_idx * 2 + 1) * sh_high_stride + lane];
                    }
                    if (sum_acc | cnt_acc) {
                        int node = 7 + node_idx;
                        atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 0), (unsigned long long)sum_acc);
                        atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 1), (unsigned long long)cnt_acc);
                    }
                }
            }
            
            __syncthreads(); // Wait for Warp 0 to finish reading
            
            // 3. Reset shared memory (All warps do this in parallel)
            for (int i = 0; i < n_ge3; ++i) {
                my_warp_region[(i * 2 + 0) * sh_high_stride + lane] = 0;
                my_warp_region[(i * 2 + 1) * sh_high_stride + lane] = 0;
            }
            
            __syncthreads(); // Wait for reset to finish
            
            // 4. Reset registers
            hf0 = hw0 = 0; hf10 = hf11 = hw10 = hw11 = 0;
            hf20 = hf21 = hf22 = hf23 = 0; hw20 = hw21 = hw22 = hw23 = 0;
            tile_count = 0;
        }
    }
    
    // Final Flush (no reset needed)
    const int warp_offset = block_warp * 7 * 32;
    sh_low[warp_offset + 0 * 32 + lane] = pack_sc((int)hf0, (int)hw0);
    sh_low[warp_offset + 1 * 32 + lane] = pack_sc((int)hf10, (int)hw10);
    sh_low[warp_offset + 2 * 32 + lane] = pack_sc((int)hf11, (int)hw11);
    sh_low[warp_offset + 3 * 32 + lane] = pack_sc((int)hf20, (int)hw20);
    sh_low[warp_offset + 4 * 32 + lane] = pack_sc((int)hf21, (int)hw21);
    sh_low[warp_offset + 5 * 32 + lane] = pack_sc((int)hf22, (int)hw22);
    sh_low[warp_offset + 6 * 32 + lane] = pack_sc((int)hf23, (int)hw23);
    
    __syncthreads();
    
    if (block_warp == 0) {
        for (int node = 0; node < 7; ++node) {
            unsigned long long acc = 0ull;
            for (int w = 0; w < warps_per_block; ++w) {
                acc = add_pack(acc, sh_low[w * 7 * 32 + node * 32 + lane]);
            }
            int64_t fsum = (int64_t)(int)(uint32_t)acc;
            int64_t csum = (int64_t)(uint32_t)(acc >> 32);
            if (fsum | csum) {
                atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 0), (unsigned long long)fsum);
                atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 1), (unsigned long long)csum);
            }
        }
        
        for (int node_idx = 0; node_idx < n_ge3; ++node_idx) {
            int64_t sum_acc = 0, cnt_acc = 0;
            for (int w = 0; w < warps_per_block; ++w) {
                int* wr = sh_high_base + w * n_ge3 * 2 * sh_high_stride;
                sum_acc += (int64_t)wr[(node_idx * 2 + 0) * sh_high_stride + lane];
                cnt_acc += (int64_t)wr[(node_idx * 2 + 1) * sh_high_stride + lane];
            }
            if (sum_acc | cnt_acc) {
                int node = 7 + node_idx;
                atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 0), (unsigned long long)sum_acc);
                atomicAdd((unsigned long long*)Hptr_coalesced(H, nodes_total, feat_set, node, lane, 1), (unsigned long long)cnt_acc);
            }
        }
    }
}

// ============================================================================
// HOST LAUNCHER
// ============================================================================

static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

torch::Tensor h_sm(
    torch::Tensor XS,
    torch::Tensor Y,
    torch::Tensor LF,
    int max_depth)
{
    const int nfeatsets = (int)XS.size(0);
    const int cols_32M = (int)XS.size(1);
    const int N = (int)Y.size(0);
    const int nodes_tot = (1 << max_depth) - 1;
    
    // Use coalesced layout for kernel: [nfeatsets, nodes, 32, 2]
    auto opts = XS.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous);
    auto H_opt = torch::zeros({(long long)nfeatsets, (long long)nodes_tot, 32LL, 2LL}, opts);
    
    auto* prop = at::cuda::getCurrentDeviceProperties();
    size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin : (size_t)prop->sharedMemPerBlock;
    
    int n_ge3 = std::max((1 << max_depth) - 8, 1);
    
    // Dynamic selection
    int sh_high_stride = 33;
    int warps_per_block = 1;
    
    bool fit = false;
    for (int stride_opt : {33, 32}) {
        for (int wpb : {8, 4, 2, 1}) {
            size_t smem_high = (size_t)wpb * n_ge3 * 2 * stride_opt * sizeof(int);
            if (smem_high % 8 != 0) smem_high += 4;
            size_t smem_low = (size_t)wpb * 7 * 32 * sizeof(unsigned long long);
            
            if (smem_high + smem_low <= smem_cap) {
                sh_high_stride = stride_opt;
                warps_per_block = wpb;
                fit = true;
                break;
            }
        }
        if (fit) break;
    }
    
    size_t smem_high = (size_t)warps_per_block * n_ge3 * 2 * sh_high_stride * sizeof(int);
    if (smem_high % 8 != 0) smem_high += 4;
    size_t smem_low = (size_t)warps_per_block * 7 * 32 * sizeof(unsigned long long);
    size_t smem_bytes = smem_high + smem_low;

    TORCH_CHECK(smem_bytes <= smem_cap, "Required shared memory (", smem_bytes, ") exceeds limit (", smem_cap, ")");

    const int SM = prop->multiProcessorCount;
    const int target_blocks = SM * 24;
    int blocks_per_feat = std::max(1, target_blocks / (nfeatsets * warps_per_block));
    const int total_warps = blocks_per_feat * warps_per_block * nfeatsets;
    int stride = std::max(1, ceil_div_int(cols_32M, total_warps * 32));
    
    dim3 grid(nfeatsets, blocks_per_feat, 1);
    dim3 block(warps_per_block * 32, 1, 1);
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    TORCH_CHECK(Y.scalar_type() == torch::kInt16, "Y must be int16");
    TORCH_CHECK(XS.scalar_type() == torch::kUInt32 || XS.scalar_type() == torch::kInt32, "XS must be uint32/int32");
    const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());
    
    const auto lf_dt = LF.scalar_type();
    
    #define LAUNCH_H(TYPE) \
        cudaFuncSetAttribute(_h_sm<TYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes); \
        _h_sm<TYPE><<<grid, block, smem_bytes, stream.stream()>>>( \
            XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<TYPE>(), \
            H_opt.data_ptr<int64_t>(), nfeatsets, cols_32M, N, max_depth, \
            warps_per_block, stride, nodes_tot, sh_high_stride);

    if (lf_dt == torch::kUInt16) { LAUNCH_H(uint16_t) }
    else if (lf_dt == torch::kUInt32) { LAUNCH_H(uint32_t) }
    else if (lf_dt == torch::kUInt64) { LAUNCH_H(uint64_t) }
    else { TORCH_CHECK(false, "LF must be uint16/32/64"); }
    
    // Transpose back to original layout [nfeatsets, nodes, 2, 32]
    return H_opt.permute({0, 1, 3, 2}).contiguous();
}
