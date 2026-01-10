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
 
// ---------------- Kernel (templated on LF dtype) ----------------
template <typename LF_T>
__global__ void _h_sm(
    const uint32_t* __restrict__ XS, // [nfeatsets, cols_32M] where cols_32M = ceil(N/32)*32
    const int16_t* __restrict__ Y,   // [N]
    const LF_T* __restrict__ LF,     // [nfeatsets, N] (u16/u32/u64)
    int64_t* __restrict__ H,         // [nfeatsets, nodes, 2, 32] (int64)
    int nfeatsets,
    int cols_32M,      // XS.shape[1] (padded to 32-sample blocks)
    int N,             // Y.shape[0] (actual sample count)
    int max_depth,
    int warps_per_block,
    int stride,        // tiles per warp along columns
    int nodes_total    // (1<<max_depth)-1
){
  const int feat_set = blockIdx.x;
  const int block_warp = threadIdx.x >> 5; // 0..(warps_per_block-1)
  const int lane = threadIdx.x & 31;       // 0..31
  const int gwarp = warps_per_block * blockIdx.y + block_warp;
  
  // Registers for depths 0..2
  int64_t hf0=0, hw0=0;
  int64_t hf10=0, hf11=0, hw10=0, hw11=0;
  int64_t hf20=0, hf21=0, hf22=0, hf23=0;
  int64_t hw20=0, hw21=0, hw22=0, hw23=0;
  
  // Shared histogram for depths >=3 : shape [(2^D - 8), 2, 32] int32
  int n_ge3 = (1 << max_depth) - 8;
  if (n_ge3 < 1) n_ge3 = 1;
  
  extern __shared__ int shmem[];
  int* sh_high = shmem;
  unsigned long long* sh_low = (unsigned long long*)(shmem + n_ge3 * 2 * 32);
  
  // FIX #1: Zero ALL shared memory elements (not just lane-indexed)
  // Each thread zeros multiple elements in strided fashion
  const int total_high_elems = n_ge3 * 2 * 32;
  for (int i = threadIdx.x; i < total_high_elems; i += blockDim.x) {
    sh_high[i] = 0;
  }
  __syncthreads();
  
  const unsigned mask = __ballot_sync(__activemask(), true);
  
  // Each warp processes 'stride' tiles of 32 columns
  for (int j = 0; j < stride; ++j) {
    const int base = 32 * (stride * gwarp + j); // column start
    if (base < cols_32M) {
      // Load lane's locals
      const int jj_lane = base + lane;
      int32_t y_lane = 0;
      uint32_t l32 = 0;
      if (jj_lane < N) {
        y_lane = Y[jj_lane];
        // LF indexed [nfeatsets, N]
        LF_T lval = LF[static_cast<size_t>(feat_set) * static_cast<size_t>(N) + jj_lane];
        // Cast to 32-bit for warp shuffle (max_depth<=7 => safe)
        l32 = static_cast<uint32_t>(lval);
      }

      // Load this lane's 32-bit tile
      uint32_t xfd_local = 0u;
      if (base + lane < cols_32M) {
        xfd_local = XS[static_cast<size_t>(feat_set) * static_cast<size_t>(cols_32M)
            + static_cast<size_t>(base + lane)];
      }

      // Mask off bits beyond N for the tail tile (uniform)
      const int rem = N - base;
      uint32_t valid_mask;
      if (rem >= 32)      valid_mask = 0xFFFFFFFFu;
      else if (rem > 0)   valid_mask = (1u << rem) - 1u;
      else                valid_mask = 0u;
      xfd_local &= valid_mask;

      for (int k = 0; k < 32; ++k) {
        // consume one bit per iter (per-lane)
        const int v = static_cast<int>(xfd_local & 1u);
        xfd_local >>= 1;

        // all lanes participate in the shuffles each iter
        const int32_t yk = __shfl_sync(mask, y_lane, k);
        uint32_t      lk = __shfl_sync(mask, l32,    k);

        // d = 0
        hf0 += static_cast<int64_t>(v) * (int64_t)yk;
        hw0 += v;

        // d = 1
        unsigned tk = lk & 1u; lk >>= 1;
        const int64_t add = static_cast<int64_t>(v) * (int64_t)yk;
        if (tk == 0u) { hf10 += add; hw10 += v; }
        else          { hf11 += add; hw11 += v; }

        // d = 2
        tk = lk & 3u; lk >>= 2;
        if      (tk == 0u) { hf20 += add; hw20 += v; }
        else if (tk == 1u) { hf21 += add; hw21 += v; }
        else if (tk == 2u) { hf22 += add; hw22 += v; }
        else               { hf23 += add; hw23 += v; }

        // d >= 3 (FIX #7: skip atomics when v == 0)
        if (v != 0) {
          #pragma unroll
          for (int d = 3; d < max_depth; ++d) {
            const unsigned to  = (1u << d) - 1u;
            const unsigned tkd = lk & to; lk >>= d;
            const int idx = static_cast<int>(to + tkd) - 7;
            atomicAdd(&sh_high[(idx * 2 + 0) * 32 + lane], v * yk);
            atomicAdd(&sh_high[(idx * 2 + 1) * 32 + lane], v);
          }
        } else {
          // Still need to consume lk bits for alignment
          #pragma unroll
          for (int d = 3; d < max_depth; ++d) {
            const unsigned to = (1u << d) - 1u;
            lk >>= d;
          }
        }
      }
    }
  }
  
  // FIX #3: Ensure all atomic operations complete before reading sh_high
  __syncthreads();
  
  // Write low-depth registers to shared (packed, per warp, per node, per lane)
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
  
  // Compute log_wpb (assuming warps_per_block is power of 2)
  int log_wpb = 0;
  for (int tmp = warps_per_block; tmp > 1; tmp >>= 1) ++log_wpb;
  
  // Butterfly reduction for low depths
  //for (int s = 0; s < log_wpb; ++s) {
   // __syncthreads();
   // const int ofs = 1 << s;
   // if ((block_warp & ofs) == 0 && (block_warp + ofs) < warps_per_block) {
   //   for (int ndi = 0; ndi < low_nodes; ++ndi) {
   //     const int idx = (block_warp * low_nodes + ndi) * 32 + lane;
  //      const int idx_p = ((block_warp + ofs) * low_nodes + ndi) * 32 + lane;
 //       sh_low[idx] = add_pack(sh_low[idx], sh_low[idx_p]);
 //     }
 //   }
 // }

// Only let warp 0 perform the entire reduction sequentially
if (block_warp == 0) {
    for (int src_warp = 1; src_warp < warps_per_block; ++src_warp) {
        for (int ndi = 0; ndi < low_nodes; ++ndi) {
            const int idx_dst = (0 * low_nodes + ndi) * 32 + lane;
            const int idx_src = (src_warp * low_nodes + ndi) * 32 + lane;
            sh_low[idx_dst] = add_pack(sh_low[idx_dst], sh_low[idx_src]);
        }
    }
}
  
  // Write reduced low-depth values to global (from warp 0 only, unpacked)
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
  
  // Drain shared histogram (d >= 3)
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

// ---------------- Host launcher that RETURNS H (Murky-style) ----------------
static inline int ceil_div_int(int a, int b){ return (a + b - 1) / b; }

// blocks_per_feat & stride per Murky
static inline void infer_grid_stride(
    int nfeatsets, int cols_32M,
    int warps_per_block,
    int& blocks_per_feat_out, int& stride_out)
{
  // 64*103 from Murky scheduling (A100 warp/slot heuristic)
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
  int wpb = 16;                        // try aggressive
  while (wpb > 1) {
    size_t smem_low = (size_t)wpb * 7 * 32 * sizeof(unsigned long long);
    if (smem_high + smem_low <= smem_cap) break;
    wpb >>= 1;                          // 16→8→4→2→1
  }
  return (wpb < 1) ? 1 : wpb;
}

// API: returns H tensor; creates it inside like your h0_sm
// XS: [nfeatsets, cols_32M] (torch.uint32) where cols_32M = ceil(N/32)*32
// Y : [N] (torch.int16)
// LF: [nfeatsets, N] (torch.uint16/32/64)
// max_depth: <= 8 (this SMEM variant)
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
  
  // Output H: [nfeatsets, (1<<D)-1, 2, 32] int64
  auto opts = XS.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous);
  auto H = torch::zeros({XS.size(0), nodes_tot, 2, 32}, opts);
  
  // Infer launch params (Murky defaults)
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
  
  // Dynamic shared memory calculation
  size_t smem_low = static_cast<size_t>(warps_per_block) * 7 * 32 * sizeof(unsigned long long);
  size_t smem_bytes = smem_high + smem_low;
  
  TORCH_CHECK(smem_bytes <= smem_cap,
              "Required dynamic shared memory (", smem_bytes,
              ") exceeds device limit (", smem_cap, ")");
  
  auto stream = at::cuda::getCurrentCUDAStream();
  
  // Dtypes: XS must be (u)int32; Y must be int16; LF in {uint16,uint32,uint64}
  TORCH_CHECK(Y.scalar_type() == torch::kInt16,
              "Y must be int16 (got ", Y.scalar_type(), ")");
  
  TORCH_CHECK(
    XS.scalar_type() == torch::kUInt32 || XS.scalar_type() == torch::kInt32,
    "XS must be uint32/int32 (got ", XS.scalar_type(), ")"
  );
  
  const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());
  
  // Dispatch LF dtype
  const auto lf_dt = LF.scalar_type();
  if (lf_dt == torch::kUInt16) {
    cudaFuncSetAttribute(_h_sm<uint16_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    _h_sm<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr,
      Y.data_ptr<int16_t>(),
      LF.data_ptr<uint16_t>(),
      H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth,
      warps_per_block, stride, nodes_tot
    );
  } else if (lf_dt == torch::kUInt32) {
    cudaFuncSetAttribute(_h_sm<uint32_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    _h_sm<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr,
      Y.data_ptr<int16_t>(),
      LF.data_ptr<uint32_t>(),
      H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth,
      warps_per_block, stride, nodes_tot
    );
  } else if (lf_dt == torch::kUInt64) {
    cudaFuncSetAttribute(_h_sm<uint64_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    _h_sm<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr,
      Y.data_ptr<int16_t>(),
      static_cast<uint64_t*>(LF.data_ptr()),
      H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth,
      warps_per_block, stride, nodes_tot
    );
  } else {
    TORCH_CHECK(false, "LF must be one of: uint16, uint32, uint64 (got ", lf_dt, ")");
  }
  
  return H;
}
