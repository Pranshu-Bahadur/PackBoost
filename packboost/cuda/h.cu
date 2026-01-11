#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

// ---------------- Device Helpers ----------------

// H layout: [nfeatsets, nodes, 2, 32]
static inline __device__ unsigned long long int* Hptr(int64_t* H, int nodes,
                                         int feat, int node, int chan, int lane) {
  size_t idx = (((static_cast<size_t>(feat) * nodes + node) * 2 + chan) * 32u + lane);
  return (unsigned long long int*)(H + idx);
}

// pack (sum, count) -> 64-bit (low32=sum, high32=count)
// Note: We deliberately truncate 64-bit sums to 32-bit here, consistent with the original design.
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
 
template <typename LF_T>
__global__ void _h_sm(
    const uint32_t* __restrict__ XS, // [nfeatsets, cols_32M]
    const int16_t* __restrict__ Y,   // [N]
    const LF_T* __restrict__ LF,     // [nfeatsets, N]
    int64_t* __restrict__ H,         // [nfeatsets, nodes, 2, 32]
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

  // FIX 4: Use int32 accumulators. 
  // This reduces register pressure by 50% vs int64, preventing spills on T4.
  // The final reduction packs to 32-bit anyway, so precision loss is identical to baseline.
  int32_t hf0=0, hw0=0;
  int32_t hf10=0, hf11=0, hw10=0, hw11=0;
  int32_t hf20=0, hf21=0, hf22=0, hf23=0;
  int32_t hw20=0, hw21=0, hw22=0, hw23=0;

  // Shared histogram for depths >=3
  // If max_depth=7 (127 nodes), n_ge3 = 120 nodes (indices 7..126).
  int n_ge3 = (1 << max_depth) - 8;
  if (n_ge3 < 1) n_ge3 = 1;
  
  extern __shared__ int shmem[];
  int* sh_high = shmem; // [n_ge3 * 2 * 32] ints
  unsigned long long* sh_low = (unsigned long long*)(shmem + n_ge3 * 2 * 32);

  // FIX 1: Efficient block-wide zeroing of shared memory
  const int sh_high_elems = n_ge3 * 2 * 32;
  for (int i = threadIdx.x; i < sh_high_elems; i += blockDim.x) {
    sh_high[i] = 0;
  }
  
  const unsigned mask = __ballot_sync(__activemask(), true);
  __syncthreads();

  // Grid-stride loop
  for (int j = 0; j < stride; ++j) {
    const int base = 32 * (stride * gwarp + j);
    if (base < cols_32M) {
      const int jj_lane = base + lane;
      int32_t y_lane = 0;
      uint32_t l32 = 0;
      
      // Load Y and Leaf ID
      if (jj_lane < N) {
        y_lane = Y[jj_lane]; // int16 -> int32
        l32 = static_cast<uint32_t>(LF[static_cast<size_t>(feat_set) * N + jj_lane]);
      }

      // Load 32 packed features
      uint32_t xfd_local = 0u;
      if (base + lane < cols_32M) {
          xfd_local = XS[static_cast<size_t>(feat_set) * static_cast<size_t>(cols_32M)
              + static_cast<size_t>(base + lane)];
      }

      // Mask out padding bits if N is not multiple of 32
      const int rem = N - base;
      uint32_t valid_mask = (rem >= 32) ? 0xFFFFFFFFu : ((rem > 0) ? (1u << rem) - 1u : 0u);
      xfd_local &= valid_mask;

      // Process 32 bits (features)
      for (int k = 0; k < 32; ++k) {
          const int v = static_cast<int>(xfd_local & 1u);
          xfd_local >>= 1;

          const int32_t yk = __shfl_sync(mask, y_lane, k);
          uint32_t      lk = __shfl_sync(mask, l32,    k);

          // d = 0
          hf0 += v * yk;
          hw0 += v;

          // d = 1
          unsigned tk = lk & 1u; lk >>= 1;
          const int32_t add = v * yk;
          if (tk == 0u) { hf10 += add; hw10 += v; }
          else          { hf11 += add; hw11 += v; }

          // d = 2
          tk = lk & 3u; lk >>= 2;
          if      (tk == 0u) { hf20 += add; hw20 += v; }
          else if (tk == 1u) { hf21 += add; hw21 += v; }
          else if (tk == 2u) { hf22 += add; hw22 += v; }
          else               { hf23 += add; hw23 += v; }

          // d >= 3 (Stored in Shared Memory)
          // FIX 2: Predicated atomicAdd avoids contention when v=0
          #pragma unroll
          for (int d = 3; d < max_depth; ++d) {
            const unsigned to  = (1u << d) - 1u;
            const unsigned tkd = lk & to; lk >>= d;
            const int idx = static_cast<int>(to + tkd) - 7; // Map nodes 7..127 to 0..120
            
            if (v) {
              atomicAdd(&sh_high[(idx * 2 + 0) * 32 + lane], yk);
              atomicAdd(&sh_high[(idx * 2 + 1) * 32 + lane], 1);
            }
          }
        }
    }
  }

  // Pack registers (d=0..2) to shared memory for reduction
  const int low_nodes = 7;
  int nd = 0;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(hf0, hw0);
  nd = 1;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(hf10, hw10);
  nd = 2;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(hf11, hw11);
  nd = 3;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(hf20, hw20);
  nd = 4;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(hf21, hw21);
  nd = 5;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(hf22, hw22);
  nd = 6;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(hf23, hw23);

  // Butterfly reduction for d=0..2 across warps
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

  // Drain d=0..2 to global memory (from warp 0)
  if (block_warp == 0) {
    for (int ndi = 0; ndi < low_nodes; ++ndi) {
      const int node = ndi;
      if (node < nodes_total) {
        const unsigned long long pack = sh_low[(0 * low_nodes + ndi) * 32 + lane];
        const int64_t fsum = static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(pack)));
        const int64_t csum = static_cast<int64_t>(static_cast<uint32_t>(pack >> 32));
        atomicAdd(Hptr(H, nodes_total, feat_set, node, 0, lane), static_cast<unsigned long long>(fsum));
        atomicAdd(Hptr(H, nodes_total, feat_set, node, 1, lane), static_cast<unsigned long long>(csum));
      }
    }
  }
  __syncthreads();
  
  // Drain d>=3 (Shared High) to global memory (parallel across warps)
  const int rows_per_warp = (n_ge3 + warps_per_block - 1) / warps_per_block;
  for (int k = 0; k < rows_per_warp; ++k) {
    const int node = 7 + rows_per_warp * block_warp + k; 
    if (node < nodes_total) {
      const int base = ((node - 7) * 2) * 32 + lane;
      // sh_high is int32, H is int64. Cast to int64 before atomicAdd.
      const int64_t fsum = static_cast<int64_t>(sh_high[base + 0]);
      const int64_t csum = static_cast<int64_t>(sh_high[base + 32]);
      atomicAdd(Hptr(H, nodes_total, feat_set, node, 0, lane), (unsigned long long int)fsum);
      atomicAdd(Hptr(H, nodes_total, feat_set, node, 1, lane), (unsigned long long int)csum);
    }
  }
}

// ---------------- Host Launcher ----------------

static inline int ceil_div_int(int a, int b){ return (a + b - 1) / b; }

static inline int infer_hist_warps_per_block(int max_depth){
  const int residual = 15 - 2 - 6 - max_depth; 
  int lg2 = 4 - std::max(0, residual); 
  return 1 << std::max(0, lg2); 
}

static inline void infer_grid_stride(
    int nfeatsets, int cols_32M,
    int warps_per_block,
    int& blocks_per_feat,
    int& stride
){
  const int A100_SCHED = 64 * 103; // Heuristic
  blocks_per_feat = ceil_div_int(A100_SCHED, warps_per_block);
  // Cap for small problems
  int total_warps = blocks_per_feat * warps_per_block;
  if (total_warps * 32 > cols_32M) {
      total_warps = ceil_div_int(cols_32M, 32);
      blocks_per_feat = ceil_div_int(total_warps, warps_per_block);
      if(blocks_per_feat < 1) blocks_per_feat = 1;
      total_warps = blocks_per_feat * warps_per_block;
  }
  blocks_per_feat = std::min(blocks_per_feat, 32); // Hard cap
  stride = ceil_div_int(cols_32M, total_warps * 32);
  if(stride < 1) stride = 1;
}

static inline int choose_warps_that_fit(size_t smem_high, size_t smem_cap) {
    int wpb = 16;
    while(wpb > 1) {
        // low_nodes = 7
        size_t smem_low = (size_t)wpb * 7 * 32 * sizeof(unsigned long long);
        if (smem_high + smem_low <= smem_cap) return wpb;
        wpb >>= 1;
    }
    return 1;
}

torch::Tensor h_sm(
    torch::Tensor XS, 
    torch::Tensor Y, 
    torch::Tensor LF, 
    int max_depth
){
  TORCH_CHECK(XS.is_cuda() && Y.is_cuda() && LF.is_cuda(), "All inputs must be CUDA");
  TORCH_CHECK(XS.is_contiguous() && Y.is_contiguous() && LF.is_contiguous(), "Inputs must be contiguous");
  TORCH_CHECK(Y.scalar_type() == torch::kInt16, "Y must be int16");
  TORCH_CHECK(XS.scalar_type() == torch::kUInt32 || XS.scalar_type() == torch::kInt32, "XS must be uint32/int32");

  const int nfeatsets = static_cast<int>(XS.size(0));
  const int cols_32M = static_cast<int>(XS.size(1)); 
  const int N = static_cast<int>(LF.size(1));
  
  // FIX 6: Correct nodes calculation to match CPU reference
  // nodes_total = 2^max_depth - 1. (e.g., depth 3 has 7 nodes)
  const int nodes_total = (1 << max_depth) - 1;
  
  auto opts = torch::TensorOptions().dtype(torch::kLong).device(XS.device());
  auto H = torch::zeros({nfeatsets, nodes_total, 2, 32}, opts);

  // Shared mem calculation
  int n_ge3 = std::max((1 << max_depth) - 8, 1);
  size_t smem_high = (size_t)n_ge3 * 2 * 32 * sizeof(int);
  
  auto* prop = at::cuda::getCurrentDeviceProperties();
  size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                 : (size_t)prop->sharedMemPerBlock;
                                                 
  int warps_per_block = infer_hist_warps_per_block(max_depth);
  
  // FIX 5: Cap warps and blocks for T4 (Turing) to improve atomic throughput
  if (prop->major < 8) {
      if (warps_per_block > 8) warps_per_block = 8;
  }
  
  // Downscale if shared mem limited
  warps_per_block = std::min(warps_per_block, choose_warps_that_fit(smem_high, smem_cap));
  
  int blocks_per_feat, stride;
  infer_grid_stride(nfeatsets, cols_32M, warps_per_block, blocks_per_feat, stride);

  // Apply T4 block cap after inference
  if (prop->major < 8 && blocks_per_feat > 8) {
     blocks_per_feat = 8;
     int total_warps = blocks_per_feat * warps_per_block;
     stride = ceil_div_int(cols_32M, total_warps * 32);
     if (stride < 1) stride = 1;
  }

  size_t smem_low = (size_t)warps_per_block * 7 * 32 * sizeof(unsigned long long);
  size_t smem_bytes = smem_high + smem_low;
  
  TORCH_CHECK(smem_bytes <= smem_cap, "Shared memory required exceeds capacity");

  dim3 grid(nfeatsets, blocks_per_feat, 1);
  dim3 block(warps_per_block * 32, 1, 1);
  auto stream = at::cuda::getCurrentCUDAStream();

  // Increase shared mem limit if needed
  if (smem_bytes > (size_t)prop->sharedMemPerBlock) {
      cudaFuncSetAttribute((void*)_h_sm<uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
      cudaFuncSetAttribute((void*)_h_sm<uint32_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
      cudaFuncSetAttribute((void*)_h_sm<uint64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
  }

  const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());
  const int16_t* Y_ptr = Y.data_ptr<int16_t>();
  int64_t* H_ptr = H.data_ptr<int64_t>();

  #define CALL_K(TYPE) \
    _h_sm<TYPE><<<grid, block, smem_bytes, stream>>>( \
      XS_ptr, Y_ptr, LF.data_ptr<TYPE>(), H_ptr, \
      nfeatsets, cols_32M, N, max_depth, \
      warps_per_block, stride, nodes_total);

  if (LF.scalar_type() == torch::kUInt16 || LF.scalar_type() == torch::kInt16) {
      CALL_K(uint16_t);
  } else if (LF.scalar_type() == torch::kUInt32 || LF.scalar_type() == torch::kInt32) {
      CALL_K(uint32_t);
  } else if (LF.scalar_type() == torch::kUInt64 || LF.scalar_type() == torch::kLong) {
      CALL_K(uint64_t);
  } else {
      TORCH_CHECK(false, "Unsupported LF type");
  }

  return H;
}
