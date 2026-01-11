#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

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
    const uint32_t* __restrict__ XS, // [nfeatsets, N]
    const int16_t* __restrict__ Y, // [N]
    const LF_T* __restrict__ LF, // [nfeatsets, N] (u16/u32/u64)
    int64_t* __restrict__ H, // [nfeatsets, nodes, 2, 32] (int64)
    int nfeatsets,
    int cols_32M, // XS.shape[1] == N
    int N, // Y.shape[0]
    int max_depth,
    int warps_per_block,
    int stride, // tiles per warp along columns
    int nodes_total // (1<<max_depth)-1
){
  const int feat_set = blockIdx.x;
  const int block_warp = threadIdx.x >> 5; // 0..(warps_per_block-1)
  const int lane = threadIdx.x & 31; // 0..31
  const int gwarp = warps_per_block * blockIdx.y + block_warp;

  // Registers for depths 0..3 (nodes 0..14)
  int64_t hf0=0, hw0=0;
  int64_t hf10=0, hf11=0, hw10=0, hw11=0;
  int64_t hf20=0, hf21=0, hf22=0, hf23=0;
  int64_t hw20=0, hw21=0, hw22=0, hw23=0;
  // FIX 3: Add depth 3 to registers
  int64_t hf30=0, hf31=0, hf32=0, hf33=0, hf34=0, hf35=0, hf36=0, hf37=0;
  int64_t hw30=0, hw31=0, hw32=0, hw33=0, hw34=0, hw35=0, hw36=0, hw37=0;

  // Shared histogram for depths >=4 : shape [(2^D - 16), 2, 32] int32
  int n_ge4 = (1 << max_depth) - 16;
  if (n_ge4 < 1) n_ge4 = 1;
  
  extern __shared__ int shmem[];
  int* sh_high = shmem;
  unsigned long long* sh_low = (unsigned long long*)(shmem + n_ge4 * 2 * 32);

  // FIX 1: Efficient block-wide zeroing of high-depth shared memory
  const int sh_high_elems = n_ge4 * 2 * 32;
  for (int i = threadIdx.x; i < sh_high_elems; i += blockDim.x) {
    sh_high[i] = 0;
  }
  
  const unsigned mask = __ballot_sync(__activemask(), true);
  __syncthreads();

  // Each warp processes 'stride' tiles of 32 columns
  for (int j = 0; j < stride; ++j) {
    const int base = 32 * (stride * gwarp + j); // column start
    if (base < cols_32M) {
      // Load lane’s locals
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

          // FIX 3: d = 3 (8 nodes) moved to registers
          tk = lk & 7u; lk >>= 3;
          if      (tk == 0u) { hf30 += add; hw30 += v; }
          else if (tk == 1u) { hf31 += add; hw31 += v; }
          else if (tk == 2u) { hf32 += add; hw32 += v; }
          else if (tk == 3u) { hf33 += add; hw33 += v; }
          else if (tk == 4u) { hf34 += add; hw34 += v; }
          else if (tk == 5u) { hf35 += add; hw35 += v; }
          else if (tk == 6u) { hf36 += add; hw36 += v; }
          else               { hf37 += add; hw37 += v; }

          // FIX 2 & 3: d >= 4 (predicated on v)
          #pragma unroll
          for (int d = 4; d < max_depth; ++d) {
            const unsigned to  = (1u << d) - 1u;
            const unsigned tkd = lk & to; lk >>= d;
            const int idx = static_cast<int>(to + tkd) - 15; // Node 15 is start of d=4
            
            // Predication significantly reduces atomic traffic for sparse bits
            if (v) {
              atomicAdd(&sh_high[(idx * 2 + 0) * 32 + lane], yk);
              atomicAdd(&sh_high[(idx * 2 + 1) * 32 + lane], 1);
            }
          }
        }
    }
  }

  // Write low-depth registers to shared (packed, per warp, per node, per lane)
  const int low_nodes = 15;
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
  // Pack d=3 registers
  nd = 7;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf30), static_cast<int>(hw30));
  nd = 8;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf31), static_cast<int>(hw31));
  nd = 9;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf32), static_cast<int>(hw32));
  nd = 10;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf33), static_cast<int>(hw33));
  nd = 11;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf34), static_cast<int>(hw34));
  nd = 12;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf35), static_cast<int>(hw35));
  nd = 13;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf36), static_cast<int>(hw36));
  nd = 14;
  sh_low[(block_warp * low_nodes + nd) * 32 + lane] = pack_sc(static_cast<int>(hf37), static_cast<int>(hw37));

  // Compute log_wpb (assuming warps_per_block is power of 2)
  int log_wpb = 0;
  for (int tmp = warps_per_block; tmp > 1; tmp >>= 1) ++log_wpb;
  // Butterfly reduction for low depths
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
  // Write reduced low-depth values to global (from warp 0 only, unpacked)
  __syncthreads();
  if (block_warp == 0) {
    for (int ndi = 0; ndi < low_nodes; ++ndi) {
      const int node = ndi; // 0..14
      // CHECK ADDED: Prevent out-of-bounds writes for small max_depth
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
  // Drain shared histogram (d >= 4)
  const int rows_per_warp = (n_ge4 + warps_per_block - 1) / warps_per_block;
  for (int k = 0; k < rows_per_warp; ++k) {
    const int node = 15 + rows_per_warp * block_warp + k; // Start at node 15
    if (node < nodes_total) {
      const int base = ((node - 15) * 2) * 32 + lane;
      const int64_t fsum = static_cast<int64_t>(sh_high[base + 0]);
      const int64_t csum = static_cast<int64_t>(sh_high[base + 32]);
      atomicAdd(Hptr(H, nodes_total, feat_set, node, 0, lane), (unsigned long long int)fsum);
      atomicAdd(Hptr(H, nodes_total, feat_set, node, 1, lane), (unsigned long long int)csum);
    }
  }
}
// ---------------- Host launcher that RETURNS H (Murky-style) ----------------
static inline int ceil_div_int(int a, int b){ return (a + b - 1) / b; }
// Murky A100 heuristic for warps-per-block
static inline int infer_hist_warps_per_block(int max_depth){
  // residual_sm_a100 = 15 - 2 - 6 - max_depth
  const int residual = 15 - 2 - 6 - max_depth; // = 7 - max_depth
  TORCH_CHECK(residual >= 0, "residual_sm_a100 < 0; max_depth too large for this variant");
  int lg2 = 4 - max(0, residual); // 4,3,2,1,0 -> 16,8,4,2,1
  if (lg2 < 0) lg2 = 0;
  return 1 << lg2; // warps_per_block in {1,2,4,8,16}
}
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
    // FIX 3: Multiplier increased to 15 (nodes 0..14 in registers)
    size_t smem_low = (size_t)wpb * 15 * 32 * sizeof(unsigned long long);
    if (smem_high + smem_low <= smem_cap) break;
    wpb >>= 1;                          // 16→8→4→2→1
  }
  return (wpb < 1) ? 1 : wpb;
}

// API: returns H tensor; creates it inside like your h0_sm
// XS: [nfeatsets, N] (torch.uint32)
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
  
  // FIX 3: high part now starts at node 15 (depth 4)
  int n_ge4 = std::max((1 << max_depth) - 16, 1);
  size_t smem_high = (size_t)n_ge4 * 2 * 32 * sizeof(int);
  
  auto* prop = at::cuda::getCurrentDeviceProperties();
  size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                 : (size_t)prop->sharedMemPerBlock;
  const int warps_per_block = choose_warps_that_fit(smem_high, smem_cap);
  int blocks_per_feat = 0, stride = 0;
  infer_grid_stride(nfeatsets, cols_32M, warps_per_block, blocks_per_feat, stride);
  dim3 grid(nfeatsets, blocks_per_feat, 1);
  dim3 block(warps_per_block * 32, 1, 1);
  
  // FIX 3: shared memory sizing updates
  // High part: (2^D - 16) * 2 * 32 * 4 bytes
  // Low part:  warps_per_block * 15 * 32 * 8 bytes
  size_t smem_low = static_cast<size_t>(warps_per_block) * 15 * 32 * sizeof(unsigned long long);
  size_t smem_bytes = smem_high + smem_low;
  
  TORCH_CHECK(smem_bytes <= smem_cap,
              "Required dynamic shared memory (", smem_bytes,
              ") exceeds device limit (", smem_cap, ")");
  auto stream = at::cuda::getCurrentCUDAStream();
  // Dtypes: XS must be (u)int32; Y must be int16; LF in {uint16,uint32,uint64}
  TORCH_CHECK(Y.scalar_type() == torch::kInt16,
              "Y must be int16 (got ", Y.scalar_type(), ")");
  // We accept XS either as uint32 or int32 (bit-identical)
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
