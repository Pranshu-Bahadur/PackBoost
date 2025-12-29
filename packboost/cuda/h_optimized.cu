// Vectorized + Branchless + EXPLICIT Register Histograms
// v1: Vectorized inner loop (4 bits/iteration)
// v2: Branchless accumulation (depths 1-2)
// v3_explicit: Explicit scalar registers for depths 0-3 (NO ARRAYS!)
//              Avoids stack frame allocation (192 bytes in v3_lite)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

static inline __device__ unsigned long long int* Hptr(int64_t* H, int nodes,
                                         int feat, int node, int chan, int lane) {
  size_t idx = (((static_cast<size_t>(feat) * nodes + node) * 2 + chan) * 32u + lane);
  return (unsigned long long int*)(H + idx);
}

__device__ __forceinline__ unsigned long long pack_sc(int sum32, int cnt32) {
    return ( (unsigned long long)(unsigned int)cnt32 << 32 ) |
             (unsigned long long)(unsigned int)sum32;
}

static __device__ __forceinline__ unsigned long long add_pack(unsigned long long a, unsigned long long b) {
    int sa = (int)(unsigned int)a;
    int ca = (int)(unsigned int)(a >> 32);
    int sb = (int)(unsigned int)b;
    int cb = (int)(unsigned int)(b >> 32);
    return pack_sc(sa + sb, ca + cb);
}
 
// ============================================================================
// KERNEL: Vectorized + Branchless + EXPLICIT Register Histograms (Depths 0-3)
// ============================================================================
template <typename LF_T>
__global__ void _h_sm_v3_explicit(
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
  
  // ========================================================================
  // EXPLICIT scalar registers for depths 0-3 (30 int64 registers, 0 bytes stack!)
  // ========================================================================
  // Depth 0: 1 node (2 registers)
  int64_t hf0 = 0, hw0 = 0;
  
  // Depth 1: 2 nodes (4 registers)
  int64_t hf1_0 = 0, hw1_0 = 0;
  int64_t hf1_1 = 0, hw1_1 = 0;
  
  // Depth 2: 4 nodes (8 registers)
  int64_t hf2_0 = 0, hw2_0 = 0;
  int64_t hf2_1 = 0, hw2_1 = 0;
  int64_t hf2_2 = 0, hw2_2 = 0;
  int64_t hf2_3 = 0, hw2_3 = 0;
  
  // Depth 3: 8 nodes (16 registers)
  int64_t hf3_0 = 0, hw3_0 = 0;
  int64_t hf3_1 = 0, hw3_1 = 0;
  int64_t hf3_2 = 0, hw3_2 = 0;
  int64_t hf3_3 = 0, hw3_3 = 0;
  int64_t hf3_4 = 0, hw3_4 = 0;
  int64_t hf3_5 = 0, hw3_5 = 0;
  int64_t hf3_6 = 0, hw3_6 = 0;
  int64_t hf3_7 = 0, hw3_7 = 0;
  
  // Total: 30 int64 registers (same as v3_lite but NO STACK!)
  
  // Shared histogram for depths >= 4
  int n_ge4 = (max_depth >= 4) ? std::max((1 << max_depth) - 15, 1) : 1;
  extern __shared__ int shmem[];
  int* sh_high = shmem;
  unsigned long long* sh_low = (unsigned long long*)(shmem + n_ge4 * 2 * 32);
  
  const unsigned mask = __ballot_sync(__activemask(), true);
  
  // Initialize shared memory for depths >= 4
  #pragma unroll
  for (int i = 0; i < n_ge4; ++i) {
    sh_high[(i * 2 + 0) * 32 + lane] = 0;
    sh_high[(i * 2 + 1) * 32 + lane] = 0;
  }
  __syncthreads();
  
  // ========================================================================
  // MAIN LOOP: Process data with EXPLICIT register histograms
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

      // Vectorized + branchless inner loop
      #pragma unroll 8
      for (int k = 0; k < 32; k += 4) {
        // Shuffle 4 values at once
        const int32_t y0 = __shfl_sync(mask, y_lane, k + 0);
        const int32_t y1 = __shfl_sync(mask, y_lane, k + 1);
        const int32_t y2 = __shfl_sync(mask, y_lane, k + 2);
        const int32_t y3 = __shfl_sync(mask, y_lane, k + 3);
        
        const uint32_t l0 = __shfl_sync(mask, l32, k + 0);
        const uint32_t l1 = __shfl_sync(mask, l32, k + 1);
        const uint32_t l2 = __shfl_sync(mask, l32, k + 2);
        const uint32_t l3 = __shfl_sync(mask, l32, k + 3);
        
        // Extract 4 bits
        const int v0 = (xfd_local >> (k + 0)) & 1;
        const int v1 = (xfd_local >> (k + 1)) & 1;
        const int v2 = (xfd_local >> (k + 2)) & 1;
        const int v3 = (xfd_local >> (k + 3)) & 1;
        
        const int64_t add0 = static_cast<int64_t>(v0) * y0;
        const int64_t add1 = static_cast<int64_t>(v1) * y1;
        const int64_t add2 = static_cast<int64_t>(v2) * y2;
        const int64_t add3 = static_cast<int64_t>(v3) * y3;
        
        // Depth 0
        hf0 += add0 + add1 + add2 + add3;
        hw0 += v0 + v1 + v2 + v3;
        
        // Depth 1 - branchless
        const int b1_0 = l0 & 1;
        const int b1_1 = l1 & 1;
        const int b1_2 = l2 & 1;
        const int b1_3 = l3 & 1;
        
        hf1_0 += add0 * (1 - b1_0) + add1 * (1 - b1_1) + add2 * (1 - b1_2) + add3 * (1 - b1_3);
        hw1_0 += v0 * (1 - b1_0) + v1 * (1 - b1_1) + v2 * (1 - b1_2) + v3 * (1 - b1_3);
        hf1_1 += add0 * b1_0 + add1 * b1_1 + add2 * b1_2 + add3 * b1_3;
        hw1_1 += v0 * b1_0 + v1 * b1_1 + v2 * b1_2 + v3 * b1_3;
        
        // Depth 2 - branchless (4 nodes)
        const int idx2_0 = (l0 >> 1) & 3;
        const int idx2_1 = (l1 >> 1) & 3;
        const int idx2_2 = (l2 >> 1) & 3;
        const int idx2_3 = (l3 >> 1) & 3;
        
        // Branchless updates for depth 2
        const int b2_0_0 = (idx2_0 == 0);
        const int b2_0_1 = (idx2_0 == 1);
        const int b2_0_2 = (idx2_0 == 2);
        const int b2_0_3 = (idx2_0 == 3);
        
        hf2_0 += add0 * b2_0_0; hw2_0 += v0 * b2_0_0;
        hf2_1 += add0 * b2_0_1; hw2_1 += v0 * b2_0_1;
        hf2_2 += add0 * b2_0_2; hw2_2 += v0 * b2_0_2;
        hf2_3 += add0 * b2_0_3; hw2_3 += v0 * b2_0_3;
        
        const int b2_1_0 = (idx2_1 == 0);
        const int b2_1_1 = (idx2_1 == 1);
        const int b2_1_2 = (idx2_1 == 2);
        const int b2_1_3 = (idx2_1 == 3);
        
        hf2_0 += add1 * b2_1_0; hw2_0 += v1 * b2_1_0;
        hf2_1 += add1 * b2_1_1; hw2_1 += v1 * b2_1_1;
        hf2_2 += add1 * b2_1_2; hw2_2 += v1 * b2_1_2;
        hf2_3 += add1 * b2_1_3; hw2_3 += v1 * b2_1_3;
        
        const int b2_2_0 = (idx2_2 == 0);
        const int b2_2_1 = (idx2_2 == 1);
        const int b2_2_2 = (idx2_2 == 2);
        const int b2_2_3 = (idx2_2 == 3);
        
        hf2_0 += add2 * b2_2_0; hw2_0 += v2 * b2_2_0;
        hf2_1 += add2 * b2_2_1; hw2_1 += v2 * b2_2_1;
        hf2_2 += add2 * b2_2_2; hw2_2 += v2 * b2_2_2;
        hf2_3 += add2 * b2_2_3; hw2_3 += v2 * b2_2_3;
        
        const int b2_3_0 = (idx2_3 == 0);
        const int b2_3_1 = (idx2_3 == 1);
        const int b2_3_2 = (idx2_3 == 2);
        const int b2_3_3 = (idx2_3 == 3);
        
        hf2_0 += add3 * b2_3_0; hw2_0 += v3 * b2_3_0;
        hf2_1 += add3 * b2_3_1; hw2_1 += v3 * b2_3_1;
        hf2_2 += add3 * b2_3_2; hw2_2 += v3 * b2_3_2;
        hf2_3 += add3 * b2_3_3; hw2_3 += v3 * b2_3_3;
        
        // ================================================================
        // Depth 3 - EXPLICIT branchless (8 nodes, NO ARRAY!)
        // ================================================================
        if (max_depth >= 4) {
          const int idx3_0 = (l0 >> 3) & 7;
          const int idx3_1 = (l1 >> 3) & 7;
          const int idx3_2 = (l2 >> 3) & 7;
          const int idx3_3 = (l3 >> 3) & 7;
          
          // Process sample 0
          const int b3_0_0 = (idx3_0 == 0);
          const int b3_0_1 = (idx3_0 == 1);
          const int b3_0_2 = (idx3_0 == 2);
          const int b3_0_3 = (idx3_0 == 3);
          const int b3_0_4 = (idx3_0 == 4);
          const int b3_0_5 = (idx3_0 == 5);
          const int b3_0_6 = (idx3_0 == 6);
          const int b3_0_7 = (idx3_0 == 7);
          
          hf3_0 += add0 * b3_0_0; hw3_0 += v0 * b3_0_0;
          hf3_1 += add0 * b3_0_1; hw3_1 += v0 * b3_0_1;
          hf3_2 += add0 * b3_0_2; hw3_2 += v0 * b3_0_2;
          hf3_3 += add0 * b3_0_3; hw3_3 += v0 * b3_0_3;
          hf3_4 += add0 * b3_0_4; hw3_4 += v0 * b3_0_4;
          hf3_5 += add0 * b3_0_5; hw3_5 += v0 * b3_0_5;
          hf3_6 += add0 * b3_0_6; hw3_6 += v0 * b3_0_6;
          hf3_7 += add0 * b3_0_7; hw3_7 += v0 * b3_0_7;
          
          // Process sample 1
          const int b3_1_0 = (idx3_1 == 0);
          const int b3_1_1 = (idx3_1 == 1);
          const int b3_1_2 = (idx3_1 == 2);
          const int b3_1_3 = (idx3_1 == 3);
          const int b3_1_4 = (idx3_1 == 4);
          const int b3_1_5 = (idx3_1 == 5);
          const int b3_1_6 = (idx3_1 == 6);
          const int b3_1_7 = (idx3_1 == 7);
          
          hf3_0 += add1 * b3_1_0; hw3_0 += v1 * b3_1_0;
          hf3_1 += add1 * b3_1_1; hw3_1 += v1 * b3_1_1;
          hf3_2 += add1 * b3_1_2; hw3_2 += v1 * b3_1_2;
          hf3_3 += add1 * b3_1_3; hw3_3 += v1 * b3_1_3;
          hf3_4 += add1 * b3_1_4; hw3_4 += v1 * b3_1_4;
          hf3_5 += add1 * b3_1_5; hw3_5 += v1 * b3_1_5;
          hf3_6 += add1 * b3_1_6; hw3_6 += v1 * b3_1_6;
          hf3_7 += add1 * b3_1_7; hw3_7 += v1 * b3_1_7;
          
          // Process sample 2
          const int b3_2_0 = (idx3_2 == 0);
          const int b3_2_1 = (idx3_2 == 1);
          const int b3_2_2 = (idx3_2 == 2);
          const int b3_2_3 = (idx3_2 == 3);
          const int b3_2_4 = (idx3_2 == 4);
          const int b3_2_5 = (idx3_2 == 5);
          const int b3_2_6 = (idx3_2 == 6);
          const int b3_2_7 = (idx3_2 == 7);
          
          hf3_0 += add2 * b3_2_0; hw3_0 += v2 * b3_2_0;
          hf3_1 += add2 * b3_2_1; hw3_1 += v2 * b3_2_1;
          hf3_2 += add2 * b3_2_2; hw3_2 += v2 * b3_2_2;
          hf3_3 += add2 * b3_2_3; hw3_3 += v2 * b3_2_3;
          hf3_4 += add2 * b3_2_4; hw3_4 += v2 * b3_2_4;
          hf3_5 += add2 * b3_2_5; hw3_5 += v2 * b3_2_5;
          hf3_6 += add2 * b3_2_6; hw3_6 += v2 * b3_2_6;
          hf3_7 += add2 * b3_2_7; hw3_7 += v2 * b3_2_7;
          
          // Process sample 3
          const int b3_3_0 = (idx3_3 == 0);
          const int b3_3_1 = (idx3_3 == 1);
          const int b3_3_2 = (idx3_3 == 2);
          const int b3_3_3 = (idx3_3 == 3);
          const int b3_3_4 = (idx3_3 == 4);
          const int b3_3_5 = (idx3_3 == 5);
          const int b3_3_6 = (idx3_3 == 6);
          const int b3_3_7 = (idx3_3 == 7);
          
          hf3_0 += add3 * b3_3_0; hw3_0 += v3 * b3_3_0;
          hf3_1 += add3 * b3_3_1; hw3_1 += v3 * b3_3_1;
          hf3_2 += add3 * b3_3_2; hw3_2 += v3 * b3_3_2;
          hf3_3 += add3 * b3_3_3; hw3_3 += v3 * b3_3_3;
          hf3_4 += add3 * b3_3_4; hw3_4 += v3 * b3_3_4;
          hf3_5 += add3 * b3_3_5; hw3_5 += v3 * b3_3_5;
          hf3_6 += add3 * b3_3_6; hw3_6 += v3 * b3_3_6;
          hf3_7 += add3 * b3_3_7; hw3_7 += v3 * b3_3_7;
        }
        
        // Depths >= 4: use shared memory atomics
        if (max_depth >= 5) {
          uint32_t lk0 = l0 >> 6;
          uint32_t lk1 = l1 >> 6;
          uint32_t lk2 = l2 >> 6;
          uint32_t lk3 = l3 >> 6;
          
          #pragma unroll
          for (int d = 4; d < max_depth; ++d) {
            const unsigned to = (1u << d) - 1u;
            const unsigned tkd0 = lk0 & to; lk0 >>= d;
            const unsigned tkd1 = lk1 & to; lk1 >>= d;
            const unsigned tkd2 = lk2 & to; lk2 >>= d;
            const unsigned tkd3 = lk3 & to; lk3 >>= d;
            
            const int idx0_d = static_cast<int>(to + tkd0) - 15;
            const int idx1_d = static_cast<int>(to + tkd1) - 15;
            const int idx2_d = static_cast<int>(to + tkd2) - 15;
            const int idx3_d = static_cast<int>(to + tkd3) - 15;
            
            atomicAdd(&sh_high[(idx0_d * 2 + 0) * 32 + lane], v0 * y0);
            atomicAdd(&sh_high[(idx0_d * 2 + 1) * 32 + lane], v0);
            atomicAdd(&sh_high[(idx1_d * 2 + 0) * 32 + lane], v1 * y1);
            atomicAdd(&sh_high[(idx1_d * 2 + 1) * 32 + lane], v1);
            atomicAdd(&sh_high[(idx2_d * 2 + 0) * 32 + lane], v2 * y2);
            atomicAdd(&sh_high[(idx2_d * 2 + 1) * 32 + lane], v2);
            atomicAdd(&sh_high[(idx3_d * 2 + 0) * 32 + lane], v3 * y3);
            atomicAdd(&sh_high[(idx3_d * 2 + 1) * 32 + lane], v3);
          }
        }
      }
    }
  }
  
  // ========================================================================
  // Write EXPLICIT register histograms to shared memory for reduction
  // ========================================================================
  const int low_nodes = 15;  // Depths 0-3
  
  // Pack and write to shared low area
  sh_low[(block_warp * low_nodes + 0) * 32 + lane] = pack_sc(static_cast<int>(hf0), static_cast<int>(hw0));
  sh_low[(block_warp * low_nodes + 1) * 32 + lane] = pack_sc(static_cast<int>(hf1_0), static_cast<int>(hw1_0));
  sh_low[(block_warp * low_nodes + 2) * 32 + lane] = pack_sc(static_cast<int>(hf1_1), static_cast<int>(hw1_1));
  sh_low[(block_warp * low_nodes + 3) * 32 + lane] = pack_sc(static_cast<int>(hf2_0), static_cast<int>(hw2_0));
  sh_low[(block_warp * low_nodes + 4) * 32 + lane] = pack_sc(static_cast<int>(hf2_1), static_cast<int>(hw2_1));
  sh_low[(block_warp * low_nodes + 5) * 32 + lane] = pack_sc(static_cast<int>(hf2_2), static_cast<int>(hw2_2));
  sh_low[(block_warp * low_nodes + 6) * 32 + lane] = pack_sc(static_cast<int>(hf2_3), static_cast<int>(hw2_3));
  
  if (max_depth >= 4) {
    sh_low[(block_warp * low_nodes + 7) * 32 + lane] = pack_sc(static_cast<int>(hf3_0), static_cast<int>(hw3_0));
    sh_low[(block_warp * low_nodes + 8) * 32 + lane] = pack_sc(static_cast<int>(hf3_1), static_cast<int>(hw3_1));
    sh_low[(block_warp * low_nodes + 9) * 32 + lane] = pack_sc(static_cast<int>(hf3_2), static_cast<int>(hw3_2));
    sh_low[(block_warp * low_nodes + 10) * 32 + lane] = pack_sc(static_cast<int>(hf3_3), static_cast<int>(hw3_3));
    sh_low[(block_warp * low_nodes + 11) * 32 + lane] = pack_sc(static_cast<int>(hf3_4), static_cast<int>(hw3_4));
    sh_low[(block_warp * low_nodes + 12) * 32 + lane] = pack_sc(static_cast<int>(hf3_5), static_cast<int>(hw3_5));
    sh_low[(block_warp * low_nodes + 13) * 32 + lane] = pack_sc(static_cast<int>(hf3_6), static_cast<int>(hw3_6));
    sh_low[(block_warp * low_nodes + 14) * 32 + lane] = pack_sc(static_cast<int>(hf3_7), static_cast<int>(hw3_7));
  }
  
  // Butterfly reduction for register depths (0-3)
  int log_wpb = 0;
  for (int tmp = warps_per_block; tmp > 1; tmp >>= 1) ++log_wpb;
  
  const int reg_nodes = (max_depth >= 4) ? 15 : 7;
  
  for (int s = 0; s < log_wpb; ++s) {
    __syncthreads();
    const int ofs = 1 << s;
    if ((block_warp & ofs) == 0 && (block_warp + ofs) < warps_per_block) {
      for (int ndi = 0; ndi < reg_nodes; ++ndi) {
        const int idx = (block_warp * low_nodes + ndi) * 32 + lane;
        const int idx_p = ((block_warp + ofs) * low_nodes + ndi) * 32 + lane;
        sh_low[idx] = add_pack(sh_low[idx], sh_low[idx_p]);
      }
    }
  }
  
  // Write reduced register values to global
  __syncthreads();
  if (block_warp == 0) {
    for (int ndi = 0; ndi < reg_nodes; ++ndi) {
      const unsigned long long pack = sh_low[ndi * 32 + lane];
      const int64_t fsum = static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(pack)));
      const int64_t csum = static_cast<int64_t>(static_cast<uint32_t>(pack >> 32));
      atomicAdd(Hptr(H, nodes_total, feat_set, ndi, 0, lane), static_cast<unsigned long long>(fsum));
      atomicAdd(Hptr(H, nodes_total, feat_set, ndi, 1, lane), static_cast<unsigned long long>(csum));
    }
  }
  __syncthreads();
  
  // Drain shared histogram (depths >= 4)
  if (max_depth >= 5) {
    const int rows_per_warp = (n_ge4 + warps_per_block - 1) / warps_per_block;
    for (int k = 0; k < rows_per_warp; ++k) {
      const int node = 15 + rows_per_warp * block_warp + k;
      if (node < nodes_total) {
        const int base = ((node - 15) * 2) * 32 + lane;
        const int64_t fsum = static_cast<int64_t>(sh_high[base + 0]);
        const int64_t csum = static_cast<int64_t>(sh_high[base + 32]);
        atomicAdd(Hptr(H, nodes_total, feat_set, node, 0, lane), (unsigned long long int)fsum);
        atomicAdd(Hptr(H, nodes_total, feat_set, node, 1, lane), (unsigned long long int)csum);
      }
    }
  }
}

// ============================================================================
// HOST WRAPPER
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
    size_t smem_low = (size_t)wpb * 15 * 32 * sizeof(unsigned long long);
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
  
  // Shared memory for depths >= 4
  int n_ge4 = (max_depth >= 4) ? std::max((1 << max_depth) - 15, 1) : 1;
  size_t smem_high = (size_t)n_ge4 * 2 * 32 * sizeof(int);
  
  auto* prop = at::cuda::getCurrentDeviceProperties();
  size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                 : (size_t)prop->sharedMemPerBlock;
  
  const int warps_per_block = choose_warps_that_fit(smem_high, smem_cap);
  int blocks_per_feat = 0, stride = 0;
  infer_grid_stride(nfeatsets, cols_32M, warps_per_block, blocks_per_feat, stride);
  
  dim3 grid(nfeatsets, blocks_per_feat, 1);
  dim3 block(warps_per_block * 32, 1, 1);
  
  // Shared memory: high (depths >= 4) + low (depths 0-3 for reduction)
  size_t smem_low = static_cast<size_t>(warps_per_block) * 15 * 32 * sizeof(unsigned long long);
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
    cudaFuncSetAttribute(_h_sm_v3_explicit<uint16_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    _h_sm_v3_explicit<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint16_t>(),
      H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth,
      warps_per_block, stride, nodes_tot
    );
  } else if (lf_dt == torch::kUInt32) {
    cudaFuncSetAttribute(_h_sm_v3_explicit<uint32_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    _h_sm_v3_explicit<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint32_t>(),
      H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth,
      warps_per_block, stride, nodes_tot
    );
  } else if (lf_dt == torch::kUInt64) {
    cudaFuncSetAttribute(_h_sm_v3_explicit<uint64_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    _h_sm_v3_explicit<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
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
