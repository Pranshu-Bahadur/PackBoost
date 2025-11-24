// packboost/cuda/cut_des.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>

constexpr int WARP_SIZE      = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

#ifndef CUTCUDA_MAX_FOLDS
#define CUTCUDA_MAX_FOLDS 32
#endif

__device__ __forceinline__ int depth_from_leaf(int leaf) {
  // node 0 -> depth 0, nodes 1..2 -> depth 1, etc.
  return 31 - __clz((unsigned)(leaf + 1));
}

// Strict lexicographic compare for reduction: (dir, sbits)
__device__ __forceinline__ bool better_pair(float dir_a, int sbits_a,
                                            float dir_b, int sbits_b) {
  if (dir_a > dir_b) return true;
  if (dir_a < dir_b) return false;
  return sbits_a > sbits_b;
}

extern "C" __global__ void cut_des_cuda_kernel(
    // inputs
    const uint16_t* __restrict__ F,      // [treesets, 32*K1] (uint16)
    const uint8_t*  __restrict__ FST,    // [treesets, K1, D] (uint8)
    const int64_t*  __restrict__ H,      // [K1, E, nodes, 2, 32] (int64)
    const int64_t*  __restrict__ H0,     // [K0, nodes, E, 2]     (int64)
    // outputs
    int32_t*        __restrict__ V,      // [treesets, K0, 2*nodes]
    uint16_t*       __restrict__ I,      // [treesets, K0, nodes]
    // dims
    int treesets, int K0, int K1, int E, int nodes, int D,
    int tree_set,
    // hyperparams
    float L2, float lr, int qgrad_bits, int max_depth,
    float min_child_weight, float min_split_gain)
{
  #pragma fp_contract(off)

  const int leaf = blockIdx.x;
  const int lane = threadIdx.x;

  if (blockDim.x != WARP_SIZE || lane >= WARP_SIZE) return;
  if (leaf >= nodes) return;
  if ((unsigned)tree_set >= (unsigned)treesets) return;
  if ((unsigned)K0 > (unsigned)CUTCUDA_MAX_FOLDS) return;

  const int depth = depth_from_leaf(leaf);
  if ((unsigned)depth >= (unsigned)D) return;

  // scalars (fp32, match CPU pow)
  const float L2_eff = L2 * powf(2.0f, 5.0f - (float)depth);
  const float qscale = lr * (float)(1u << (31 - qgrad_bits))
                       * powf(2.0f, -(float)(max_depth - depth));

  const bool use_min_child = (min_child_weight > 0.0f);
  const bool use_min_gain  = (min_split_gain  > 0.0f);

  // Per-lane per-fold scratch (track best candidate)
  float    best_dir [CUTCUDA_MAX_FOLDS];
  int      best_sbit[CUTCUDA_MAX_FOLDS];
  int      best_vl  [CUTCUDA_MAX_FOLDS];
  int      best_vr  [CUTCUDA_MAX_FOLDS];
  uint16_t best_k   [CUTCUDA_MAX_FOLDS];

  #pragma unroll
  for (int f = 0; f < CUTCUDA_MAX_FOLDS; ++f) {
    if (f >= K0) break;
    best_dir [f] = -INFINITY;        // sentinel
    best_sbit[f] = INT_MIN;
    best_vl  [f] = 0;
    best_vr  [f] = 0;
    best_k   [f] = 0;
  }

  // Iterate all candidates k
  for (int k = 0; k < K1; ++k) {
    // FST[tree_set, k, depth] -> fold id
    const size_t fst_idx =
        (((size_t)tree_set * (size_t)K1) + (size_t)k) * (size_t)D
        + (size_t)depth;
    const int tree_fold = (int)FST[fst_idx];
    if ((unsigned)tree_fold >= (unsigned)K0) continue;

    // ---------- ERA LOOP: accumulate lane-local stats ----------
    float sum_dir = 0.0f;
    int   eras_used = 0;

    float G0_tot_lane = 0.0f;  // sum over eras for THIS lane
    float N0_tot_lane = 0.0f;
    float GT_tot      = 0.0f;  // parent stats (same across lanes)
    float HT_tot      = 0.0f;

    for (int e = 0; e < E; ++e) {
      // H index: [K1, E, nodes, 2, 32]
      const size_t h_base =
          (((((size_t)k * (size_t)E) + (size_t)e) * (size_t)nodes + (size_t)leaf) * 2u) * 32u
          + (size_t)lane;
      const float G0 = __ll2float_rn(H[h_base + 0 * 32u]);
      const float N0 = __ll2float_rn(H[h_base + 1 * 32u]);

      // H0 index: [K0, nodes, E, 2]
      const size_t h0_base =
          (((((size_t)tree_fold * (size_t)nodes) + (size_t)leaf) * (size_t)E) + (size_t)e) * 2u;
      const float GT = __ll2float_rn(H0[h0_base + 0]);
      const float HT = __ll2float_rn(H0[h0_base + 1]);

      // Right child by difference to parent (per era, lane-local left)
      const float G1 = GT - G0;
      const float N1 = HT - N0;

      if (N0 > 0.f && N1 > 0.f) {
        const float V0 = G0 / (N0 + L2_eff);
        const float V1 = G1 / (N1 + L2_eff);
        const float sgn = (V1 > V0) ? 1.f : (V1 < V0 ? -1.f : 0.f);
        sum_dir   += sgn;
        ++eras_used;
      }

      G0_tot_lane += G0;
      N0_tot_lane += N0;
      GT_tot      += GT;   // same across lanes; still harmless to accumulate per lane
      HT_tot      += HT;
    }

    // ---------- NODE-LEVEL AGGREGATION (across lanes & eras) ----------
    // Lane-local partials across eras: G0_tot_lane, N0_tot_lane
    float G0_acc = G0_tot_lane;
    float N0_acc = N0_tot_lane;

    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
      G0_acc += __shfl_down_sync(FULL_MASK, G0_acc, offset, WARP_SIZE);
      N0_acc += __shfl_down_sync(FULL_MASK, N0_acc, offset, WARP_SIZE);
    }

    // Broadcast full sums (lane 0) to all lanes
    const float G0_node = __shfl_sync(FULL_MASK, G0_acc, 0, WARP_SIZE);
    const float N0_node = __shfl_sync(FULL_MASK, N0_acc, 0, WARP_SIZE);

    // Parent totals are already node-level in H0; GT_tot/HT_tot are the same on all lanes
    const float G01_node = GT_tot;
    const float N01_node = HT_tot;
    const float G1_node  = G01_node - G0_node;
    const float N1_node  = N01_node - N0_node;

    // ---------- NODE-LEVEL GATING (min_child_weight / min_split_gain) ----------
    int keep = 1;

    if (use_min_child || use_min_gain) {
      // Basic sanity
      if (N0_node <= 0.0f || N1_node <= 0.0f) {
        keep = 0;
      }

      if (keep && use_min_child) {
        if (N0_node < min_child_weight || N1_node < min_child_weight) {
          keep = 0;
        }
      }

      if (keep && use_min_gain) {
        const float V0_node = G0_node / (N0_node + L2_eff);
        const float V1_node = G1_node / (N1_node + L2_eff);
        const float S0_node = G0_node * V0_node;
        const float S1_node = G1_node * V1_node;
        const float gain_node = S0_node + S1_node;
        if (gain_node < min_split_gain) {
          keep = 0;
        }
      }
    }

    // Broadcast keep decision from lane 0 to all lanes
    keep = __shfl_sync(FULL_MASK, keep, 0, WARP_SIZE);
    if (!keep) {
      continue;  // whole (k, leaf, fold) candidate is discarded
    }

    // ---------- LANE-LEVEL DES DIRECTION + GAIN (unchanged semantics) ----------
    const float dir_score =
        (eras_used > 0) ? fabsf(sum_dir) / (float)eras_used : 0.f;

    const float G1_tot = GT_tot - G0_tot_lane;
    const float N1_tot = HT_tot - N0_tot_lane;

    const float V0f = (N0_tot_lane > 0.f)
                        ? (G0_tot_lane / (N0_tot_lane + L2_eff))
                        : 0.f;
    const float V1f = (N1_tot > 0.f)
                        ? (G1_tot / (N1_tot + L2_eff))
                        : 0.f;

    const float S0 = G0_tot_lane * V0f;
    const float S1 = G1_tot * V1f;
    const float gain_lane = S0 + S1;

    const int S_bits = __float_as_int(gain_lane);

    if (better_pair(dir_score, S_bits, best_dir[tree_fold], best_sbit[tree_fold])) {
      best_dir [tree_fold] = dir_score;
      best_sbit[tree_fold] = S_bits;
      best_vl  [tree_fold] = __float2int_rz(qscale * V0f);
      best_vr  [tree_fold] = __float2int_rz(qscale * V1f);
      best_k   [tree_fold] = (uint16_t)k;
    }
  }

  __syncwarp(FULL_MASK);

  // ---------- Warp-reduce per fold on (dir, sbits); winner = highest lane among ties ----------
  for (int fold = 0; fold < K0; ++fold) {
    float dir_w = best_dir[fold];
    int   sb_w  = best_sbit[fold];

    // if nobody had a candidate for this fold, keep -inf / INT_MIN
    #pragma unroll
    for (int p = 0; p < 5; ++p) {
      const int ofs   = 1 << p;
      const float dir_p = __shfl_xor_sync(FULL_MASK, dir_w, ofs, WARP_SIZE);
      const int   sb_p  = __shfl_xor_sync(FULL_MASK, sb_w,  ofs, WARP_SIZE);
      if (better_pair(dir_p, sb_p, dir_w, sb_w)) {
        dir_w = dir_p;
        sb_w  = sb_p;
      }
    }

    // Empty fold? skip
    if (!isfinite(dir_w) || sb_w == INT_MIN)
      continue;

    const unsigned m_dir = __ballot_sync(FULL_MASK, best_dir[fold] == dir_w);
    const unsigned m_s   = __ballot_sync(FULL_MASK, best_sbit[fold] == sb_w);
    const unsigned msk   = m_dir & m_s;
    if (msk == 0u) continue;  // nothing to write

    const int winner_lane = 31 - __clz(msk);
    if (lane == winner_lane) {
      const size_t v_base =
          (((size_t)tree_set * (size_t)K0) + (size_t)fold)
          * (size_t)(2 * nodes) + (size_t)(2 * leaf);
      V[v_base + 0] = best_vl[fold];
      V[v_base + 1] = best_vr[fold];

      const size_t f_idx =
          (size_t)tree_set * (size_t)(32 * K1)
          + (size_t)best_k[fold] * 32u + (size_t)winner_lane;
      const size_t i_idx =
          (((size_t)tree_set * (size_t)K0) + (size_t)fold)
          * (size_t)nodes + (size_t)leaf;
      I[i_idx] = F[f_idx];
    }
  }
}

// -------- C++ launcher (shape checks for DES layouts) --------
void cut_des_cuda_launcher(
    torch::Tensor F,   // [rounds, 32*K1] uint16
    torch::Tensor FST, // [rounds, K1, D] uint8
    torch::Tensor H,   // [K1, E, nodes, 2, 32] int64
    torch::Tensor H0,  // [K0, nodes, E, 2]     int64
    torch::Tensor V,   // [rounds, K0, 2*nodes] int32
    torch::Tensor I,   // [rounds, K0, nodes]   uint16
    int tree_set,
    double L2,
    double lr,
    int qgrad_bits,
    int max_depth,
    double min_child_weight,
    double min_split_gain)
{
  TORCH_CHECK(F.dim()==2 && FST.dim()==3 && H.dim()==5 && H0.dim()==4,
              "Shapes must be: F[rounds,32*K1], FST[rounds,K1,D], H[K1,E,nodes,2,32], H0[K0,nodes,E,2]");
  const int rounds   = (int)F.size(0);
  const int K1       = (int)FST.size(1);
  const int D        = (int)FST.size(2);

  TORCH_CHECK((int)H.size(0) == K1, "H.size(0) must equal K1");
  const int E        = (int)H.size(1);
  const int nodes    = (int)H.size(2);
  TORCH_CHECK(H.size(3)==2 && H.size(4)==32, "H must be [..., 2, 32]");

  const int K0       = (int)H0.size(0);
  TORCH_CHECK(H0.size(1)==nodes && H0.size(2)==E && H0.size(3)==2, "H0 must be [K0,nodes,E,2]");

  TORCH_CHECK((int)V.size(0)==rounds && (int)V.size(1)==K0 && (int)V.size(2)==2*nodes, "V bad shape");
  TORCH_CHECK((int)I.size(0)==rounds && (int)I.size(1)==K0 && (int)I.size(2)==nodes,   "I bad shape");

  TORCH_CHECK((int)F.size(1) == 32 * K1, "F second dim must be 32*K1");

  const dim3 grid(nodes, 1, 1);   // nodes = (1 << D) - 1 internally
  const dim3 block(WARP_SIZE, 1, 1);

  auto stream = at::cuda::getCurrentCUDAStream();

  cut_des_cuda_kernel<<<grid, block, 0, stream.stream()>>>(
      reinterpret_cast<const uint16_t*>(F.data_ptr<uint16_t>()),
      FST.data_ptr<uint8_t>(),
      H.data_ptr<int64_t>(),
      H0.data_ptr<int64_t>(),
      V.data_ptr<int32_t>(),
      reinterpret_cast<uint16_t*>(I.data_ptr<uint16_t>()),
      rounds, K0, K1, E, nodes, D, (int)tree_set,
      (float)L2, (float)lr, qgrad_bits, max_depth,
      (float)min_child_weight,
      (float)min_split_gain);

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "cut_des_cuda_kernel launch failed: ",
              cudaGetErrorString(err));
}
