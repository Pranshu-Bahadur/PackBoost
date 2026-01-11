// packboost/cuda/h.cu
// FIX: Proper template instantiation and dispatch

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                << cudaGetErrorString(err) << std::endl; \
      throw std::runtime_error(cudaGetErrorString(err)); \
    } \
  } while(0)

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

template <typename LF_T, int MAX_LOCAL_NODES>
__global__ void __launch_bounds__(512, 2) _h_sm_optimized(
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
    int sh_stride)
{
  const int feat_set = blockIdx.x;
  const int block_warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int gwarp = warps_per_block * blockIdx.y + block_warp;
  
  int64_t hf0=0, hw0=0;
  int64_t hf10=0, hf11=0, hw10=0, hw11=0;
  int64_t hf20=0, hf21=0, hf22=0, hf23=0;
  int64_t hw20=0, hw21=0, hw22=0, hw23=0;
  
  int n_ge3 = (1 << max_depth) - 8;
  if (n_ge3 < 1) n_ge3 = 1;
  
  int local_sum[MAX_LOCAL_NODES];
  int local_cnt[MAX_LOCAL_NODES];
  
  #pragma unroll
  for (int i = 0; i < MAX_LOCAL_NODES; ++i) {
    local_sum[i] = 0;
    local_cnt[i] = 0;
  }
  
  extern __shared__ int shmem[];
  int* sh_high = shmem;
  unsigned long long* sh_low = (unsigned long long*)(shmem + n_ge3 * 2 * sh_stride);
  
  const unsigned mask = __ballot_sync(__activemask(), true);
  
  #pragma unroll 4
  for (int i = 0; i < n_ge3; ++i) {
    sh_high[(i * 2 + 0) * sh_stride + lane] = 0;
    sh_high[(i * 2 + 1) * sh_stride + lane] = 0;
  }
  __syncthreads();
  
  for (int j = 0; j < stride; ++j) {
    const int base = 32 * (stride * gwarp + j);
    if (base >= cols_32M) break;
    
    const int jj_lane = base + lane;
    int32_t y_lane = 0;
    uint32_t l32 = 0;
    
    if (jj_lane < N) {
      y_lane = Y[jj_lane];
      LF_T lval = LF[static_cast<size_t>(feat_set) * static_cast<size_t>(N) + jj_lane];
      l32 = static_cast<uint32_t>(lval);
    }

    uint32_t xfd_local = 0u;
    if (jj_lane < cols_32M) {
      xfd_local = XS[static_cast<size_t>(feat_set) * static_cast<size_t>(cols_32M) + jj_lane];
    }

    const int rem = N - base;
    if (rem < 32) {
      if (rem > 0) xfd_local &= (1u << rem) - 1u;
      else xfd_local = 0u;
    }

    #pragma unroll
    for (int k = 0; k < 32; ++k) {
      const int v = static_cast<int>(xfd_local & 1u);
      xfd_local >>= 1;

      const int32_t yk = __shfl_sync(mask, y_lane, k);
      uint32_t lk = __shfl_sync(mask, l32, k);

      const int vyk = v * yk;
      const int64_t vy = static_cast<int64_t>(vyk);

      hf0 += vy; 
      hw0 += v;

      const unsigned t1 = lk & 1u; 
      lk >>= 1;
      hf10 += (1 - t1) * vy;
      hf11 += t1 * vy;
      hw10 += (1 - t1) * v;
      hw11 += t1 * v;

      const unsigned t2 = lk & 3u; 
      lk >>= 2;
      hf20 += (t2 == 0) * vy;
      hf21 += (t2 == 1) * vy;
      hf22 += (t2 == 2) * vy;
      hf23 += (t2 == 3) * vy;
      hw20 += (t2 == 0) * v;
      hw21 += (t2 == 1) * v;
      hw22 += (t2 == 2) * v;
      hw23 += (t2 == 3) * v;

      if (max_depth > 3) {
        #pragma unroll
        for (int d = 3; d < 8 && d < max_depth; ++d) {
          const unsigned to  = (1u << d) - 1u;
          const unsigned tkd = lk & to; 
          lk >>= d;
          const int node_idx = static_cast<int>(to + tkd) - 7;
          
          if (node_idx < MAX_LOCAL_NODES) {
            local_sum[node_idx] += vyk;
            local_cnt[node_idx] += v;
          } else {
            atomicAdd(&sh_high[(node_idx * 2 + 0) * sh_stride + lane], vyk);
            atomicAdd(&sh_high[(node_idx * 2 + 1) * sh_stride + lane], v);
          }
        }
      }
    }
  }
  
  #pragma unroll
  for (int i = 0; i < MAX_LOCAL_NODES && i < n_ge3; ++i) {
    if (local_sum[i] | local_cnt[i]) {
      atomicAdd(&sh_high[(i * 2 + 0) * sh_stride + lane], local_sum[i]);
      atomicAdd(&sh_high[(i * 2 + 1) * sh_stride + lane], local_cnt[i]);
    }
  }
  
  const int low_nodes = 7;
  sh_low[(block_warp * low_nodes + 0) * 32 + lane] = pack_sc(static_cast<int>(hf0), static_cast<int>(hw0));
  sh_low[(block_warp * low_nodes + 1) * 32 + lane] = pack_sc(static_cast<int>(hf10), static_cast<int>(hw10));
  sh_low[(block_warp * low_nodes + 2) * 32 + lane] = pack_sc(static_cast<int>(hf11), static_cast<int>(hw11));
  sh_low[(block_warp * low_nodes + 3) * 32 + lane] = pack_sc(static_cast<int>(hf20), static_cast<int>(hw20));
  sh_low[(block_warp * low_nodes + 4) * 32 + lane] = pack_sc(static_cast<int>(hf21), static_cast<int>(hw21));
  sh_low[(block_warp * low_nodes + 5) * 32 + lane] = pack_sc(static_cast<int>(hf22), static_cast<int>(hw22));
  sh_low[(block_warp * low_nodes + 6) * 32 + lane] = pack_sc(static_cast<int>(hf23), static_cast<int>(hw23));
  
  int log_wpb = 0;
  for (int tmp = warps_per_block; tmp > 1; tmp >>= 1) ++log_wpb;
  
  for (int s = 0; s < log_wpb; ++s) {
    __syncthreads();
    const int ofs = 1 << s;
    if ((block_warp & ofs) == 0 && (block_warp + ofs) < warps_per_block) {
      #pragma unroll
      for (int ndi = 0; ndi < low_nodes; ++ndi) {
        const int idx = (block_warp * low_nodes + ndi) * 32 + lane;
        const int idx_p = ((block_warp + ofs) * low_nodes + ndi) * 32 + lane;
        sh_low[idx] = add_pack(sh_low[idx], sh_low[idx_p]);
      }
    }
  }
  
  __syncthreads();
  
  if (block_warp == 0) {
    #pragma unroll
    for (int ndi = 0; ndi < low_nodes; ++ndi) {
      const unsigned long long pack = sh_low[ndi * 32 + lane];
      const int64_t fsum = static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(pack)));
      const int64_t csum = static_cast<int64_t>(static_cast<uint32_t>(pack >> 32));
      if (fsum | csum) {
        atomicAdd(Hptr(H, nodes_total, feat_set, ndi, 0, lane), static_cast<unsigned long long>(fsum));
        atomicAdd(Hptr(H, nodes_total, feat_set, ndi, 1, lane), static_cast<unsigned long long>(csum));
      }
    }
  }
  
  __syncthreads();
  
  const int rows_per_warp = (n_ge3 + warps_per_block - 1) / warps_per_block;
  for (int k = 0; k < rows_per_warp; ++k) {
    const int node = 7 + rows_per_warp * block_warp + k;
    if (node < nodes_total) {
      const int base = ((node - 7) * 2) * sh_stride + lane;
      const int64_t fsum = static_cast<int64_t>(sh_high[base + 0]);
      const int64_t csum = static_cast<int64_t>(sh_high[base + sh_stride]);
      if (fsum | csum) {
        atomicAdd(Hptr(H, nodes_total, feat_set, node, 0, lane), (unsigned long long int)fsum);
        atomicAdd(Hptr(H, nodes_total, feat_set, node, 1, lane), (unsigned long long int)csum);
      }
    }
  }
}

static inline int ceil_div_int(int a, int b){ return (a + b - 1) / b; }

static inline int choose_warps_and_stride(int max_depth, size_t smem_cap, int& stride_out) {
  int n_ge3 = std::max((1 << max_depth) - 8, 1);
  
  for (int s : {33, 32}) {
    for (int wpb : {16, 8, 4, 2, 1}) {
      size_t smem_high = (size_t)n_ge3 * 2 * s * sizeof(int);
      size_t smem_low = (size_t)wpb * 7 * 32 * sizeof(unsigned long long);
      if (smem_high + smem_low <= smem_cap) {
        stride_out = s;
        return wpb;
      }
    }
  }
  stride_out = 32;
  return 1;
}

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

// Helper macro to launch kernel with proper template parameters
#define LAUNCH_KERNEL(LF_TYPE, LOCAL_SIZE) \
  do { \
    CUDA_CHECK(cudaFuncSetAttribute(_h_sm_optimized<LF_TYPE, LOCAL_SIZE>, \
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes))); \
    cudaFuncSetCacheConfig(_h_sm_optimized<LF_TYPE, LOCAL_SIZE>, cudaFuncCachePreferShared); \
    _h_sm_optimized<LF_TYPE, LOCAL_SIZE><<<grid, block, smem_bytes, stream.stream()>>>( \
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<LF_TYPE>(), H.data_ptr<int64_t>(), \
      nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot, sh_stride); \
    CUDA_CHECK(cudaGetLastError()); \
  } while(0)

torch::Tensor h_sm(
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
  
  auto* prop = at::cuda::getCurrentDeviceProperties();
  
  size_t smem_cap_default = (size_t)prop->sharedMemPerBlock;
  size_t smem_cap_optin = (size_t)prop->sharedMemPerBlockOptin;
  size_t smem_cap = (smem_cap_optin > smem_cap_default) ? smem_cap_optin : smem_cap_default;
  
  int sh_stride = 32;
  const int warps_per_block = choose_warps_and_stride(max_depth, smem_cap, sh_stride);
  
  int blocks_per_feat = 0, stride = 0;
  infer_grid_stride(nfeatsets, cols_32M, warps_per_block, blocks_per_feat, stride);
  
  dim3 grid(nfeatsets, blocks_per_feat, 1);
  dim3 block(warps_per_block * 32, 1, 1);
  
  int n_ge3 = std::max((1 << max_depth) - 8, 1);
  size_t smem_high = static_cast<size_t>(n_ge3) * 2 * sh_stride * sizeof(int);
  size_t smem_low = static_cast<size_t>(warps_per_block) * 7 * 32 * sizeof(unsigned long long);
  size_t smem_bytes = smem_high + smem_low;
  
  TORCH_CHECK(smem_bytes <= smem_cap,
              "Required dynamic shared memory (", smem_bytes,
              ") exceeds device limit (", smem_cap, ")");
  
  auto stream = at::cuda::getCurrentCUDAStream();
  
  TORCH_CHECK(Y.scalar_type() == torch::kInt16, "Y must be int16");
  TORCH_CHECK(XS.scalar_type() == torch::kUInt32 || XS.scalar_type() == torch::kInt32, "XS must be uint32/int32");
  const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());
  
  const auto lf_dt = LF.scalar_type();
  
  // Dispatch based on depth and LF type
  if (max_depth <= 6) {
    // D<=6: 56 high-depth nodes
    if (lf_dt == torch::kUInt16) {
      LAUNCH_KERNEL(uint16_t, 56);
    } else if (lf_dt == torch::kUInt32) {
      LAUNCH_KERNEL(uint32_t, 56);
    } else if (lf_dt == torch::kUInt64) {
      LAUNCH_KERNEL(uint64_t, 56);
    } else {
      TORCH_CHECK(false, "LF must be uint16/32/64");
    }
  } else if (max_depth == 7) {
    // D=7: 120 high-depth nodes
    if (lf_dt == torch::kUInt16) {
      LAUNCH_KERNEL(uint16_t, 120);
    } else if (lf_dt == torch::kUInt32) {
      LAUNCH_KERNEL(uint32_t, 120);
    } else if (lf_dt == torch::kUInt64) {
      LAUNCH_KERNEL(uint64_t, 120);
    } else {
      TORCH_CHECK(false, "LF must be uint16/32/64");
    }
  } else {
    // D=8: 248 nodes, use smaller buffer
    if (lf_dt == torch::kUInt16) {
      LAUNCH_KERNEL(uint16_t, 64);
    } else if (lf_dt == torch::kUInt32) {
      LAUNCH_KERNEL(uint32_t, 64);
    } else if (lf_dt == torch::kUInt64) {
      LAUNCH_KERNEL(uint64_t, 64);
    } else {
      TORCH_CHECK(false, "LF must be uint16/32/64");
    }
  }
  
  return H;
}

#undef LAUNCH_KERNEL
