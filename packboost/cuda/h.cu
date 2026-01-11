// packboost/cuda/h.cu
// FINAL OPTIMIZATION: Vectorized loads + reduced instruction count

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

// OPTIMIZATION: Process 2 columns per iteration with vectorized loads
template <typename LF_T>
__global__ void __launch_bounds__(512, 4) _h_sm_vec(
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
  
  extern __shared__ int shmem[];
  int* sh_high = shmem;
  unsigned long long* sh_low = (unsigned long long*)(shmem + n_ge3 * 2 * sh_stride);
  
  const unsigned mask = __ballot_sync(__activemask(), true);
  
  for (int i = threadIdx.x; i < n_ge3 * 2 * sh_stride; i += blockDim.x) {
    sh_high[i] = 0;
  }
  __syncthreads();
  
  // Process 2 consecutive 32-sample chunks per iteration
  for (int j = 0; j < stride; j += 2) {
    #pragma unroll 2
    for (int jj = j; jj < j + 2 && jj < stride; ++jj) {
      const int base = 32 * (stride * gwarp + jj);
      if (base >= cols_32M) break;
      
      const int jj_lane = base + lane;
      
      // OPTIMIZATION: Coalesced loads
      int32_t y_lane = (jj_lane < N) ? Y[jj_lane] : 0;
      uint32_t l32 = 0;
      if (jj_lane < N) {
        LF_T lval = LF[static_cast<size_t>(feat_set) * static_cast<size_t>(N) + jj_lane];
        l32 = static_cast<uint32_t>(lval);
      }

      uint32_t xfd_local = (jj_lane < cols_32M) ? 
        XS[static_cast<size_t>(feat_set) * static_cast<size_t>(cols_32M) + jj_lane] : 0u;

      const int rem = N - base;
      if (rem < 32 && rem > 0) {
        xfd_local &= (1u << rem) - 1u;
      } else if (rem <= 0) {
        xfd_local = 0u;
      }

      // OPTIMIZATION: Manual unroll with reduced operations
      #pragma unroll 32
      for (int k = 0; k < 32; ++k) {
        const int v = xfd_local & 1;
        xfd_local >>= 1;

        const int32_t yk = __shfl_sync(mask, y_lane, k);
        uint32_t lk = __shfl_sync(mask, l32, k);
        
        const int vyk = v * yk;

        // Depth 0
        hf0 += vyk; 
        hw0 += v;

        // Depth 1 - extract bit
        const int b1 = lk & 1;
        lk >>= 1;
        hf10 += (b1 ^ 1) * vyk;
        hf11 += b1 * vyk;
        hw10 += (b1 ^ 1) * v;
        hw11 += b1 * v;

        // Depth 2 - extract 2 bits
        const int b2 = lk & 3;
        lk >>= 2;
        const int is0 = (b2 == 0);
        const int is1 = (b2 == 1);
        const int is2 = (b2 == 2);
        const int is3 = (b2 == 3);
        hf20 += is0 * vyk;
        hf21 += is1 * vyk;
        hf22 += is2 * vyk;
        hf23 += is3 * vyk;
        hw20 += is0 * v;
        hw21 += is1 * v;
        hw22 += is2 * v;
        hw23 += is3 * v;

        // Depth 3+: direct atomics (simpler, less register pressure)
        if (max_depth > 3) {
          for (int d = 3; d < max_depth; ++d) {
            const unsigned mask_d = (1u << d) - 1u;
            const unsigned node_offset = lk & mask_d;
            lk >>= d;
            const int node_idx = static_cast<int>(mask_d + node_offset) - 7;
            atomicAdd(&sh_high[(node_idx * 2 + 0) * sh_stride + lane], vyk);
            atomicAdd(&sh_high[(node_idx * 2 + 1) * sh_stride + lane], v);
          }
        }
      }
    }
  }
  
  const int low_nodes = 7;
  sh_low[(block_warp * low_nodes + 0) * 32 + lane] = pack_sc(hf0, hw0);
  sh_low[(block_warp * low_nodes + 1) * 32 + lane] = pack_sc(hf10, hw10);
  sh_low[(block_warp * low_nodes + 2) * 32 + lane] = pack_sc(hf11, hw11);
  sh_low[(block_warp * low_nodes + 3) * 32 + lane] = pack_sc(hf20, hw20);
  sh_low[(block_warp * low_nodes + 4) * 32 + lane] = pack_sc(hf21, hw21);
  sh_low[(block_warp * low_nodes + 5) * 32 + lane] = pack_sc(hf22, hw22);
  sh_low[(block_warp * low_nodes + 6) * 32 + lane] = pack_sc(hf23, hw23);
  
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
  
  // Round stride up to even number for vectorized version
  if (stride & 1) stride++;
  
  blocks_per_feat_out = blocks_per_feat;
  stride_out = stride;
}

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
  
  cudaFuncSetCacheConfig(_h_sm_vec<uint16_t>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(_h_sm_vec<uint32_t>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(_h_sm_vec<uint64_t>, cudaFuncCachePreferL1);
  
  if (lf_dt == torch::kUInt16) {
    CUDA_CHECK(cudaFuncSetAttribute(_h_sm_vec<uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    _h_sm_vec<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint16_t>(), H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot, sh_stride);
    CUDA_CHECK(cudaGetLastError());
  } else if (lf_dt == torch::kUInt32) {
    CUDA_CHECK(cudaFuncSetAttribute(_h_sm_vec<uint32_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    _h_sm_vec<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint32_t>(), H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot, sh_stride);
    CUDA_CHECK(cudaGetLastError()
