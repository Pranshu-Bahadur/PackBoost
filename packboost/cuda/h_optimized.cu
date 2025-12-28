#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// H layout helper: [nfeatsets, nodes, 2, 32]
static inline __device__ unsigned long long int* Hptr(int64_t* H, int nodes,
                                         int feat, int node, int chan, int lane) {
  size_t idx = (((static_cast<size_t>(feat) * nodes + node) * 2 + chan) * 32u + lane);
  return (unsigned long long int*)(H + idx);
}

// pack (sum,count) -> 64-bit (low32 = sum (signed), high32 = count (unsigned))
__device__ __forceinline__ unsigned long long pack_sc(int sum32, int cnt32) {
    return ( (unsigned long long)(unsigned int)cnt32 << 32 ) |
             (unsigned long long)(unsigned int)sum32;
}

// add two packed values
static __device__ __forceinline__ unsigned long long add_pack(unsigned long long a, unsigned long long b) {
    int sa = (int)(unsigned int)a;
    int ca = (int)(unsigned int)(a >> 32);
    int sb = (int)(unsigned int)b;
    int cb = (int)(unsigned int)(b >> 32);
    return pack_sc(sa + sb, ca + cb);
}
 
// ============================================================================
// KERNEL WITH VECTORIZED INNER LOOP (Process 4 bits/iteration instead of 1)
// ============================================================================
template <typename LF_T>
__global__ void _h_sm_v1(
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
){
  const int feat_set = blockIdx.x;
  const int block_warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int gwarp = warps_per_block * blockIdx.y + block_warp;
  
  // Registers for depths 0..2
  int64_t hf0=0, hw0=0;
  int64_t hf10=0, hf11=0, hw10=0, hw11=0;
  int64_t hf20=0, hf21=0, hf22=0, hf23=0;
  int64_t hw20=0, hw21=0, hw22=0, hw23=0;
  
  // Shared histogram for depths >=3
  int n_ge3 = (1 << max_depth) - 8;
  if (n_ge3 < 1) n_ge3 = 1;
  extern __shared__ int shmem[];
  int* sh_high = shmem;
  unsigned long long* sh_low = (unsigned long long*)(shmem + n_ge3 * 2 * 32);
  
  const unsigned mask = __ballot_sync(__activemask(), true);
  
  #pragma unroll
  for (int i = 0; i < n_ge3; ++i) {
    sh_high[(i * 2 + 0) * 32 + lane] = 0;
    sh_high[(i * 2 + 1) * 32 + lane] = 0;
  }
  __syncthreads();
  
  // ========================================================================
  // MAIN LOOP - Same as original
  // ========================================================================
  for (int j = 0; j < stride; ++j) {
    const int base = 32 * (stride * gwarp + j);
    if (base < cols_32M) {
      const int jj_lane = base + lane;
      int32_t y_lane = 0;
      uint32_t l32 = 0;
      if (jj_lane < N) {
        y_lane = Y[jj_lane];
        LF_T lval = LF[static_cast<size_t>(feat_set) * static_cast<size_t>(N) + jj_lane];
        l32 = static_cast<uint32_t>(lval);
      }

      uint32_t xfd_local = 0u;
      if (base + lane < cols_32M) {
        xfd_local = XS[static_cast<size_t>(feat_set) * static_cast<size_t>(cols_32M)
            + static_cast<size_t>(base + lane)];
      }

      const int rem = N - base;
      uint32_t valid_mask;
      if (rem >= 32)      valid_mask = 0xFFFFFFFFu;
      else if (rem > 0)   valid_mask = (1u << rem) - 1u;
      else                valid_mask = 0u;
      xfd_local &= valid_mask;

      // ====================================================================
      // OPTIMIZATION: Vectorized inner loop (4 bits per iteration)
      // ====================================================================
      #pragma unroll 8
      for (int k = 0; k < 32; k += 4) {
        // Shuffle 4 values at once (reduced from 32 to 8 iterations)
        const int32_t y0 = __shfl_sync(mask, y_lane, k + 0);
        const int32_t y1 = __shfl_sync(mask, y_lane, k + 1);
        const int32_t y2 = __shfl_sync(mask, y_lane, k + 2);
        const int32_t y3 = __shfl_sync(mask, y_lane, k + 3);
        
        const uint32_t l0 = __shfl_sync(mask, l32, k + 0);
        const uint32_t l1 = __shfl_sync(mask, l32, k + 1);
        const uint32_t l2 = __shfl_sync(mask, l32, k + 2);
        const uint32_t l3 = __shfl_sync(mask, l32, k + 3);
        
        // Extract 4 bits at once
        const int v0 = (xfd_local >> (k + 0)) & 1;
        const int v1 = (xfd_local >> (k + 1)) & 1;
        const int v2 = (xfd_local >> (k + 2)) & 1;
        const int v3 = (xfd_local >> (k + 3)) & 1;
        
        const int64_t add0 = static_cast<int64_t>(v0) * y0;
        const int64_t add1 = static_cast<int64_t>(v1) * y1;
        const int64_t add2 = static_cast<int64_t>(v2) * y2;
        const int64_t add3 = static_cast<int64_t>(v3) * y3;
        
        // Depth 0 - accumulate all 4
        hf0 += add0 + add1 + add2 + add3;
        hw0 += v0 + v1 + v2 + v3;
        
        // Depth 1 - process all 4
        unsigned tk0 = l0 & 1u;
        unsigned tk1 = l1 & 1u;
        unsigned tk2 = l2 & 1u;
        unsigned tk3 = l3 & 1u;
        
        if (tk0 == 0u) { hf10 += add0; hw10 += v0; } else { hf11 += add0; hw11 += v0; }
        if (tk1 == 0u) { hf10 += add1; hw10 += v1; } else { hf11 += add1; hw11 += v1; }
        if (tk2 == 0u) { hf10 += add2; hw10 += v2; } else { hf11 += add2; hw11 += v2; }
        if (tk3 == 0u) { hf10 += add3; hw10 += v3; } else { hf11 += add3; hw11 += v3; }
        
        // Depth 2 - process all 4
        tk0 = (l0 >> 1) & 3u;
        tk1 = (l1 >> 1) & 3u;
        tk2 = (l2 >> 1) & 3u;
        tk3 = (l3 >> 1) & 3u;
        
        if      (tk0 == 0u) { hf20 += add0; hw20 += v0; }
        else if (tk0 == 1u) { hf21 += add0; hw21 += v0; }
        else if (tk0 == 2u) { hf22 += add0; hw22 += v0; }
        else                { hf23 += add0; hw23 += v0; }
        
        if      (tk1 == 0u) { hf20 += add1; hw20 += v1; }
        else if (tk1 == 1u) { hf21 += add1; hw21 += v1; }
        else if (tk1 == 2u) { hf22 += add1; hw22 += v1; }
        else                { hf23 += add1; hw23 += v1; }
        
        if      (tk2 == 0u) { hf20 += add2; hw20 += v2; }
        else if (tk2 == 1u) { hf21 += add2; hw21 += v2; }
        else if (tk2 == 2u) { hf22 += add2; hw22 += v2; }
        else                { hf23 += add2; hw23 += v2; }
        
        if      (tk3 == 0u) { hf20 += add3; hw20 += v3; }
        else if (tk3 == 1u) { hf21 += add3; hw21 += v3; }
        else if (tk3 == 2u) { hf22 += add3; hw22 += v3; }
        else                { hf23 += add3; hw23 += v3; }
        
        // Depths >= 3 - process all 4
        uint32_t lk0 = l0 >> 3;
        uint32_t lk1 = l1 >> 3;
        uint32_t lk2 = l2 >> 3;
        uint32_t lk3 = l3 >> 3;
        
        #pragma unroll
        for (int d = 3; d < max_depth; ++d) {
          const unsigned to = (1u << d) - 1u;
          const unsigned tkd0 = lk0 & to; lk0 >>= d;
          const unsigned tkd1 = lk1 & to; lk1 >>= d;
          const unsigned tkd2 = lk2 & to; lk2 >>= d;
          const unsigned tkd3 = lk3 & to; lk3 >>= d;
          
          const int idx0 = static_cast<int>(to + tkd0) - 7;
          const int idx1 = static_cast<int>(to + tkd1) - 7;
          const int idx2 = static_cast<int>(to + tkd2) - 7;
          const int idx3 = static_cast<int>(to + tkd3) - 7;
          
          atomicAdd(&sh_high[(idx0 * 2 + 0) * 32 + lane], v0 * y0);
          atomicAdd(&sh_high[(idx0 * 2 + 1) * 32 + lane], v0);
          atomicAdd(&sh_high[(idx1 * 2 + 0) * 32 + lane], v1 * y1);
          atomicAdd(&sh_high[(idx1 * 2 + 1) * 32 + lane], v1);
          atomicAdd(&sh_high[(idx2 * 2 + 0) * 32 + lane], v2 * y2);
          atomicAdd(&sh_high[(idx2 * 2 + 1) * 32 + lane], v2);
          atomicAdd(&sh_high[(idx3 * 2 + 0) * 32 + lane], v3 * y3);
          atomicAdd(&sh_high[(idx3 * 2 + 1) * 32 + lane], v3);
        }
      }
      // END VECTORIZED INNER LOOP
      // ====================================================================
    }
  }
  
  // ========================================================================
  // REST OF KERNEL - UNCHANGED (reduction & writeback)
  // ========================================================================
  const int low_nodes = 7;
  int nd = 0;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf0), static_cast<int>(hw0));
  nd = 1;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf10), static_cast<int>(hw10));
  nd = 2;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf11), static_cast<int>(hw11));
  nd = 3;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf20), static_cast<int>(hw20));
  nd = 4;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf21), static_cast<int>(hw21));
  nd = 5;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf22), static_cast<int>(hw22));
  nd = 6;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf23), static_cast<int>(hw23));
  
  int log_wpb = 0;
  for (int tmp = warps_per_block; tmp > 1; tmp >>= 1) ++log_wpb;
  
  for (int s = 0; s < log_wpb; ++s) {
    __syncthreads();
    const int ofs = 1 << s;
    if ((block_warp & ofs) == 0 && (block_warp + ofs) < warps_per_block) {
      for (int ndi = 0; ndi < low_nodes; ++ndi) {
        const int idx = (block_warp * low_nodes + ndi) * 32 + lane;
        const int idx_p = ((block_warp + ofs) * low_nodes + ndi) * 32 + lane;
        sh_low[idx] = add_pack(sh_low[idx], sh_low[idx_p]);
      }
    }
  }
  
  __syncthreads();
  if (block_warp == 0) {
    const int low_node_map[7] = {0, 1, 2, 3, 4, 5, 6};
    for (int ndi = 0; ndi < low_nodes; ++ndi) {
      const unsigned long long pack = sh_low[(0 * low_nodes + ndi) * 32 + lane];
      const int64_t fsum = static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(pack)));
      const int64_t csum = static_cast<int64_t>(static_cast<uint32_t>(pack >> 32));
      const int node = low_node_map[ndi];
      atomicAdd(Hptr(H, nodes_total, feat_set, node, 0, lane), static_cast<unsigned long long>(fsum));
      atomicAdd(Hptr(H, nodes_total, feat_set, node, 1, lane), static_cast<unsigned long long>(csum));
    }
  }
  __syncthreads();
  
  const int rows_per_warp = (n_ge3 + warps_per_block - 1) / warps_per_block;
  for (int k = 0; k < rows_per_warp; ++k) {
    const int node = 7 + rows_per_warp * block_warp + k;
    if (node < nodes_total) {
      const int base = ((node - 7) * 2) * 32 + lane;
      const int64_t fsum = static_cast<int64_t>(sh_high[base + 0]);
      const int64_t csum = static_cast<int64_t>(sh_high[base + 32]);
      atomicAdd(Hptr(H, nodes_total, feat_set, node, 0, lane), (unsigned long long int)fsum);
      atomicAdd(Hptr(H, nodes_total, feat_set, node, 1, lane), (unsigned long long int)csum);
    }
  }
}

// ============================================================================
// HOST WRAPPER - Same helper functions as original
// ============================================================================
static inline int ceil_div_int(int a, int b){ return (a + b - 1) / b; }

static inline void infer_grid_stride(
    int nfeatsets, int cols_32M, int warps_per_block,
    int& blocks_per_feat_out, int& stride_out)
{
  const int A100_SCHED = 64 * 103;
  int blocks_per_feat = ceil_div_int(A100_SCHED, warps_per_block);
  blocks_per_feat = ceil_div_int(blocks_per_feat, nfeatsets);
  if (blocks_per_feat < 1) blocks_per_feat = 1;
  const int total_warps = blocks_per_feat * warps_per_block;
  int stride = ceil_div_int(cols_32M, total_warps * 32);
  if (stride < 1) stride = 1;
  blocks_per_feat_out = blocks_per_feat;
  stride_out = stride;
}

static inline int choose_warps_that_fit(size_t smem_high, size_t smem_cap) {
  int wpb = 16;
  while (wpb > 1) {
    size_t smem_low = (size_t)wpb * 7 * 32 * sizeof(unsigned long long);
    if (smem_high + smem_low <= smem_cap) break;
    wpb >>= 1;
  }
  return (wpb < 1) ? 1 : wpb;
}

torch::Tensor h_sm_optimized(
    torch::Tensor XS,
    torch::Tensor Y,
    torch::Tensor LF,
    int max_depth)
{
  const int nfeatsets = static_cast<int>(XS.size(0));
  const int cols_32M = static_cast<int>(XS.size(1));
  const int N = static_cast<int>(Y.size(0));
  const int nodes_tot = (1 << max_depth) - 1;
  
  auto opts = XS.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous);
  auto H = torch::zeros({XS.size(0), nodes_tot, 2, 32}, opts);
  
  int n_ge3 = std::max((1 << max_depth) - 8, 1);
  size_t smem_high = (size_t)n_ge3 * 2 * 32 * sizeof(int);
  
  auto* prop = at::cuda::getCurrentDeviceProperties();
  size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                 : (size_t)prop->sharedMemPerBlock;
  
  const int warps_per_block = choose_warps_that_fit(smem_high, smem_cap);
  int blocks_per_feat = 0, stride = 0;
  infer_grid_stride(nfeatsets, cols_32M, warps_per_block, blocks_per_feat, stride);
  
  dim3 grid(nfeatsets, blocks_per_feat, 1);
  dim3 block(warps_per_block * 32, 1, 1);
  
  size_t smem_low = static_cast<size_t>(warps_per_block) * 7 * 32 * sizeof(unsigned long long);
  size_t smem_bytes = smem_high + smem_low;
  
  TORCH_CHECK(smem_bytes <= smem_cap,
              "Required dynamic shared memory exceeds device limit");
  
  auto stream = at::cuda::getCurrentCUDAStream();
  
  TORCH_CHECK(Y.scalar_type() == torch::kInt16, "Y must be int16");
  TORCH_CHECK(
    XS.scalar_type() == torch::kUInt32 || XS.scalar_type() == torch::kInt32,
    "XS must be uint32/int32"
  );
  
  const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());
  
  const auto lf_dt = LF.scalar_type();
  if (lf_dt == torch::kUInt16) {
    cudaFuncSetAttribute(_h_sm_v1<uint16_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    _h_sm_v1<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint16_t>(),
      H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth,
      warps_per_block, stride, nodes_tot
    );
  } else if (lf_dt == torch::kUInt32) {
    cudaFuncSetAttribute(_h_sm_v1<uint32_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    _h_sm_v1<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint32_t>(),
      H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth,
      warps_per_block, stride, nodes_tot
    );
  } else if (lf_dt == torch::kUInt64) {
    cudaFuncSetAttribute(_h_sm_v1<uint64_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    _h_sm_v1<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), static_cast<uint64_t*>(LF.data_ptr()),
      H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth,
      warps_per_block, stride, nodes_tot
    );
  } else {
    TORCH_CHECK(false, "LF must be one of: uint16, uint32, uint64");
  }
  
  return H;
}
