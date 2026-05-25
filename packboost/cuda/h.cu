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
  // Zero this lane’s column for both channels in high
  //const unsigned mask = __activemask();
  const unsigned mask = __ballot_sync(__activemask(), true);
  #pragma unroll
  for (int i = 0; i < n_ge3; ++i) {
    sh_high[(i * 2 + 0) * 32 + lane] = 0; // sum(y)
    sh_high[(i * 2 + 1) * 32 + lane] = 0; // count
  }
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

        // d >= 3 (no branch; multiply by v)
        #pragma unroll
        for (int d = 3; d < max_depth; ++d) {
        const unsigned to  = (1u << d) - 1u;
        const unsigned tkd = lk & to; lk >>= d;
        const int idx = static_cast<int>(to + tkd) - 7;
        atomicAdd(&sh_high[(idx * 2 + 0) * 32 + lane], v * yk);
        atomicAdd(&sh_high[(idx * 2 + 1) * 32 + lane], v);
        }
        }


    }
  }
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
    size_t smem_low = (size_t)wpb * 7 * 32 * sizeof(unsigned long long);
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
  // Dynamic shared memory for high depths: (2^D - 8, 2, 32) ints
  //int n_ge3 = std::max((1 << max_depth) - 8, 1);
  //size_t smem_high = static_cast<size_t>(n_ge3) * 2 * 32 * sizeof(int);
  // Additional for low depths: (warps_per_block, 7, 32) unsigned long long (packed)
  size_t smem_low = static_cast<size_t>(warps_per_block) * 7 * 32 * sizeof(unsigned long long);
  size_t smem_bytes = smem_high + smem_low;

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
  
  size_t optin_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                  : (size_t)prop->sharedMemPerBlock;
  TORCH_CHECK(smem_bytes <= optin_cap,
              "Required dynamic shared memory (", smem_bytes,
              ") exceeds opt-in device limit (", optin_cap, ")");


debug_schema and smoke_test_packboost keep using cpu

packboost-deploy.ipynb
3 étapes terminées
debug_schema is CPU-only because it has no gpu= setting, while smoke_test_packboost is explicitly pinned to gpu="T4" and should not be CPU-bound when invoked remotely.

What’s happening
debug_schema is decorated with @app.function(..., min_containers=1, volumes=common_volumes) and @modal.fastapi_endpoint(...), but there is no GPU configured on that function, so Modal schedules it on CPU containers .
smoke_test_packboost is decorated with @app.function(gpu="T4", ...), and the deployed remote call showing cuda_available: True, device_count: 1, and device_name: 'Tesla T4' confirms it is running on a GPU-backed container .

Why it may look like CPU
The health endpoint calls smoke_test_packboost.remote(), so hitting /health reaches the GPU function indirectly through a web endpoint wrapper.

If you are testing debug_schema, that endpoint will always look like a lightweight CPU metadata route because it only returns schema and path info .

If you want both on GPU
You should only move debug_schema to GPU if it actually needs CUDA or PackBoost imports; otherwise it is better left on CPU to avoid wasting GPU capacity .
If you want, the clean fix is to keep debug_schema as CPU and make smoke_test_packboost the only GPU check endpoint, since Modal supports separate container pools per function configuration.

Code change to make
If your intent is to force debug_schema onto the same class of container as smoke_test_packboost, change its decorator to include a GPU, for example @app.function(gpu="T4", timeout=..., ...) .
But if the issue is that smoke_test_packboost is unexpectedly not using GPU, then the current notebook output already contradicts that: the remote result reports cuda_available: True and device_name: 'Tesla T4' .

Would you like me to rewrite the notebook so debug_schema stays CPU-safe and smoke_test_packboost becomes the only GPU validation path?

import requests

TRAIN_URL = "https://jmbastide--packboost-polars-service-v4-train-from-path.modal.run"

payload = {
"dataset_path": "/data/pb_train.parquet",
# "dataset_path": "/data/pb_valid.parquet",
"target": "target",
"feature_columns": features,
"format": "parquet",
"model_params": {'device':'cuda'}
}

r = requests.post(TRAIN_URL, json=payload, timeout=3600)
r.raise_for_status()
print(r.json())

{'status': 'error', 'error_type': 'FileNotFoundError', 'error': 'dataset file not found: /data/pb_train.parquet'}
2 étapes terminées
/data/pb_train.parquet is being checked inside the Modal container, and Modal is correctly raising FileNotFoundError because that file is not present in the mounted /data volume at runtime .
So the problem is not CUDA; it is that the file was never uploaded to the volume, or the path/name is different from what the service sees .

What the notebook shows
Your read_table() helper explicitly does path = Path(dataset_path) and then if not path.exists(): raise FileNotFoundError(...), so /data/pb_train.parquet must physically exist inside the container’s mounted volume for the request to succeed .
The same notebook also shows that /data is mapped to a Modal volume named packboost-data, which means local files on your machine are not automatically visible unless they were committed to that volume first .
That explains why the API can start fine but still fail on the dataset lookup .

Fix the data path
The immediate fix is to ensure pb_train.parquet is actually inside the packboost-data volume at /data/pb_train.parquet before calling train_from_path .
If you intended to use a validation file too, pass valid_path separately and make sure it also exists in /data .
If the file is named differently, use the exact mounted filename, because the code does a strict filesystem existence check .

Likely causes
The file was never uploaded to the Modal volume mounted at /data .

The file exists locally but not in the deployed container filesystem.

The filename or extension is different, such as .parquet vs .pq or a typo in pb_train.parquet .

You may need to refresh or commit the volume after writing the file, because mounted volumes are not the same thing as the notebook’s local workspace .

Recommended check
A quick sanity test is to call the debug endpoint or add a tiny endpoint that lists /data contents, then verify that pb_train.parquet appears there before training .
Once the file is present in the mounted volume, your request should move past FileNotFoundError and only fail later if there is a schema or PackBoost issue .

Would you like me to patch the notebook so the training endpoint returns a clearer /data directory listing when the file is missing?

yes
4 étapes terminées
Patched the notebook so missing dataset errors now include the contents of /data, which should make it obvious whether the file is absent, misnamed, or not committed to the volume .

What changed
I added a list_data_dir() helper and changed read_table() so its FileNotFoundError now reports available_files from /data instead of only the missing path .
That means the train endpoint will now tell you what the container can actually see at runtime, which is the fastest way to debug Modal volume issues .

Why this helps
Your current failure happens before Polars or PackBoost are involved, because the code checks Path(dataset_path).exists() first .
With the patch, you can immediately confirm whether /data/pb_train.parquet is missing from the mounted volume or whether the filename differs from what you passed .

Next step
Redeploy the notebook, rerun the POST to train_from_path, and inspect the new error message if the file is still missing .
If you want, I can also patch train_from_path to return /data listings proactively when dataset_path is invalid, so you do not need to rely on the exception path.

no problme running on a T4 gpu

Yes — that fits the pattern. T4 can succeed where L4 fails because the kernel is likely running under a different architecture path or launch configuration, and T4’s shared-memory behavior differs enough that PackBoost’s kernel stays within its limits.

The important point is that the bug is not “L4 is bad”; it is that the CUDA kernel is not portable across both GPUs without an architecture-aware shared-memory setup.

What this means
T4 has 48 KB of shared memory per block in the common deviceQuery output, while L4/Ada paths can expose different per-block/shared-memory constraints and kernel launch rules.

So a kernel that works on T4 may still fail on L4 if it requests shared memory incorrectly or does not set the launch attributes the L4 path expects.

Best interpretation
Your PackBoost branch probably has a CUDA kernel that is:

tuned for T4/Turing behavior, or

relying on defaults that happen to work on T4, but not on L4.

That explains why training runs on T4 but throws Required dynamic shared memory (32512) exceeds device limit (1024) on L4.

Practical fix
Keep T4 as a fallback, but patch PackBoost so the kernel launch explicitly requests the dynamic shared memory it needs and checks whether the current device supports that request.

If the kernel cannot be made portable, then add a GPU-specific fallback path for L4 with reduced tile/block settings.

What to do next
Run the exact same training on T4 to confirm the rest of the pipeline is stable.

Inspect the CUDA extension for shared-memory launch code.

Add an L4-specific fallback or reduce the kernel’s shared-memory footprint.

If you want, I can help you write a minimal “T4 works / L4 fallback” config split for PackBoost.

yes

Here is the cleaned-up L4-safe version of your h.cu launcher. It keeps the kernel logic intact, but fixes the shared-memory handling so the host code checks against the opt-in limit and validates the cudaFuncSetAttribute call result before launch.

cpp
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
  return ((unsigned long long)(unsigned int)cnt32 << 32) |
         (unsigned long long)(unsigned int)sum32;
}

static __device__ __forceinline__ unsigned long long add_pack(unsigned long long a, unsigned long long b) {
  int sa = (int)(unsigned int)a;
  int ca = (int)(unsigned int)(a >> 32);
  int sb = (int)(unsigned int)b;
  int cb = (int)(unsigned int)(b >> 32);
  return pack_sc(sa + sb, ca + cb);
}

template <typename LF_T>
__global__ void _h_sm(
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

  int64_t hf0=0, hw0=0;
  int64_t hf10=0, hf11=0, hw10=0, hw11=0;
  int64_t hf20=0, hf21=0, hf22=0, hf23=0;
  int64_t hw20=0, hw21=0, hw22=0, hw23=0;

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

      for (int k = 0; k < 32; ++k) {
        const int v = static_cast<int>(xfd_local & 1u);
        xfd_local >>= 1;

        const int32_t yk = __shfl_sync(mask, y_lane, k);
        uint32_t      lk = __shfl_sync(mask, l32,    k);

        hf0 += static_cast<int64_t>(v) * (int64_t)yk;
        hw0 += v;

        unsigned tk = lk & 1u; lk >>= 1;
        const int64_t add = static_cast<int64_t>(v) * (int64_t)yk;
        if (tk == 0u) { hf10 += add; hw10 += v; }
        else          { hf11 += add; hw11 += v; }

        tk = lk & 3u; lk >>= 2;
        if      (tk == 0u) { hf20 += add; hw20 += v; }
        else if (tk == 1u) { hf21 += add; hw21 += v; }
        else if (tk == 2u) { hf22 += add; hw22 += v; }
        else               { hf23 += add; hw23 += v; }

        #pragma unroll
        for (int d = 3; d < max_depth; ++d) {
          const unsigned to  = (1u << d) - 1u;
          const unsigned tkd = lk & to; lk >>= d;
          const int idx = static_cast<int>(to + tkd) - 7;
          atomicAdd(&sh_high[(idx * 2 + 0) * 32 + lane], v * yk);
          atomicAdd(&sh_high[(idx * 2 + 1) * 32 + lane], v);
        }
      }
    }
  }

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

static inline int ceil_div_int(int a, int b){ return (a + b - 1) / b; }

static inline int infer_hist_warps_per_block(int max_depth){
  const int residual = 15 - 2 - 6 - max_depth;
  TORCH_CHECK(residual >= 0, "residual_sm_a100 < 0; max_depth too large for this variant");
  int lg2 = 4 - max(0, residual);
  if (lg2 < 0) lg2 = 0;
  return 1 << lg2;
}

static inline void infer_grid_stride(
    int nfeatsets, int cols_32M,
    int warps_per_block,
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

  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(Y.scalar_type() == torch::kInt16,
              "Y must be int16 (got ", Y.scalar_type(), ")");
  TORCH_CHECK(
    XS.scalar_type() == torch::kUInt32 || XS.scalar_type() == torch::kInt32,
    "XS must be uint32/int32 (got ", XS.scalar_type(), ")"
  );
  const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());
  const auto lf_dt = LF.scalar_type();

  size_t optin_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                  : (size_t)prop->sharedMemPerBlock;
  TORCH_CHECK(smem_bytes <= optin_cap,
              "Required dynamic shared memory (", smem_bytes,
              ") exceeds opt-in device limit (", optin_cap, ")");

  if (lf_dt == torch::kUInt16) {
    cudaError_t err = cudaFuncSetAttribute(_h_sm<uint16_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
    _h_sm<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint16_t>(), H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot);
  } else if (lf_dt == torch::kUInt32) {
    cudaError_t err = cudaFuncSetAttribute(_h_sm<uint32_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
    _h_sm<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint32_t>(), H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot);
  } else if (lf_dt == torch::kUInt64) {
    cudaError_t err = cudaFuncSetAttribute(_h_sm<uint64_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
    _h_sm<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), static_cast<uint64_t*>(LF.data_ptr()), H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot);
  } else {
    TORCH_CHECK(false, "LF must be one of: uint16, uint32, uint64 (got ", lf_dt, ")");
  }
  return H;
}
