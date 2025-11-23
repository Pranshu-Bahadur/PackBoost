#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

#ifndef CUTCUDA_MAX_FOLDS
#define CUTCUDA_MAX_FOLDS 32
#endif

__device__ __forceinline__ int depth_from_leaf(int leaf) {
  return 31 - __clz(static_cast<unsigned>(leaf + 1));
}

extern "C" __global__ void cut_cuda_kernel(
    const uint16_t* __restrict__ F,      // [treesets, 32*K1]      (uint16)
    const uint8_t*  __restrict__ FST,    // [treesets, K1, D]      (uint8)
    const int64_t*  __restrict__ H,      // [K1, nodes, 2, 32]     (int64)
    const int64_t*  __restrict__ H0,     // [K0, nodes, 2]         (int64)
    int32_t*        __restrict__ V,      // [treesets, K0, 2*nodes](int32)
    uint16_t*       __restrict__ I,      // [treesets, K0, nodes]  (uint16)
    // dims
    int treesets, int K0, int K1, int nodes, int D,
    int tree_set,
    // hyperparams
    float L2, float lr, int qgrad_bits, int max_depth,
    float min_child_weight, float min_split_gain)
{
  #pragma fp_contract(off)

  const int leaf = blockIdx.x;
  const int wi   = threadIdx.x;
  if (blockDim.x != WARP_SIZE || wi >= WARP_SIZE) return;
  if (leaf >= nodes) return;
  if ((unsigned)tree_set >= (unsigned)treesets) return;
  if (K0 > CUTCUDA_MAX_FOLDS) return;

  const int depth = depth_from_leaf(leaf);
  if ((unsigned)depth >= (unsigned)D) return;

  // ---- per-fold scratch
  int       mxs[CUTCUDA_MAX_FOLDS];
  int       vls[CUTCUDA_MAX_FOLDS];
  int       vrs[CUTCUDA_MAX_FOLDS];
  uint16_t  fs [CUTCUDA_MAX_FOLDS];
  float     g01s[CUTCUDA_MAX_FOLDS];
  float     n01s[CUTCUDA_MAX_FOLDS];

  // sentinel init + fold totals
  for (int f = 0; f < K0; ++f) {
    mxs[f] = -100000000;              // negative -> smaller than any valid S_bits
    vls[f] = 0;
    vrs[f] = 0;
    fs[f]  = 0;
    const size_t base = ((size_t)f * (size_t)nodes + (size_t)leaf) * 2u;
    g01s[f] = (float)H0[base + 0];    // parent grad
    n01s[f] = (float)H0[base + 1];    // parent count
  }

  // scalars (match CPU pow; stay in fp32)
  const float L2_eff = L2 * powf(2.0f, 5.0f - (float)depth);
  const float qscale = lr * (float)(1u << (31 - qgrad_bits))
                       * powf(2.0f, -(float)(max_depth - depth));

  const bool use_min_child = (min_child_weight > 0.0f);
  const bool use_min_gain  = (min_split_gain  > 0.0f);

  // ---- per-candidate search (each lane keeps its own best per fold)
  for (int k = 0; k < K1; ++k) {
    const size_t fst_idx =
        (((size_t)tree_set * (size_t)K1) + (size_t)k) * (size_t)D
        + (size_t)depth;
    const int tree_fold = (int)FST[fst_idx];
    if ((unsigned)tree_fold >= (unsigned)K0) continue;

    const size_t h_base =
        ((((size_t)k * (size_t)nodes) + (size_t)leaf) * 2u) * 32u
        + (size_t)wi;

    const float G0 = __ll2float_rn(H[h_base + 0 * 32u]);  // left grad (lane)
    const float N0 = __ll2float_rn(H[h_base + 1 * 32u]);  // left count (lane)
    const float G1 = __fsub_rn(g01s[tree_fold], G0);      // right grad (lane)
    const float N1 = __fsub_rn(n01s[tree_fold], N0);      // right count (lane)

    // --- min_child_weight constraint (per-lane count proxy) ---
    if (use_min_child) {
      if (N0 < min_child_weight || N1 < min_child_weight) {
        continue;
      }
    }

    const float V0f = __fdiv_rn(G0, __fadd_rn(N0, L2_eff));
    const float V1f = __fdiv_rn(G1, __fadd_rn(N1, L2_eff));

    const float S0 = __fmul_rn(G0, V0f);
    const float S1 = __fmul_rn(G1, V1f);
    const float gain = __fadd_rn(S0, S1);

    // --- min_split_gain constraint ---
    if (use_min_gain && gain < min_split_gain) {
      continue;
    }

    const int S_bits = __float_as_int(gain);

    if (mxs[tree_fold] < S_bits) {
      mxs[tree_fold] = S_bits;
      vls[tree_fold] = __float2int_rz(__fmul_rn(qscale, V0f));
      vrs[tree_fold] = __float2int_rz(__fmul_rn(qscale, V1f));
      fs [tree_fold] = (uint16_t)k;
    }
  }

  __syncwarp(FULL_MASK);

  // ---- warp-reduce per fold + write
  for (int fold = 0; fold < K0; ++fold) {
    int mxw = mxs[fold];

    #pragma unroll
    for (int p = 0; p < 5; ++p) {
      const int partner = __shfl_xor_sync(FULL_MASK, mxw, 1 << p, WARP_SIZE);
      mxw = (partner > mxw) ? partner : mxw;
    }

    // empty fold? skip (matches CPU continue)
    if (mxw == -100000000) continue;

    const unsigned msk = __ballot_sync(FULL_MASK, mxs[fold] == mxw);
    if (msk == 0) continue; // paranoia

    const int winner_lane = 31 - __clz(msk);
    if (wi == winner_lane) {
      const size_t v_base =
          (((size_t)tree_set * (size_t)K0) + (size_t)fold)
          * (size_t)(2 * nodes) + (size_t)(2 * leaf);
      V[v_base + 0] = vls[fold];
      V[v_base + 1] = vrs[fold];

      const size_t f_idx =
          (size_t)tree_set * (size_t)(32 * K1)
          + (size_t)fs[fold] * 32u + (size_t)winner_lane;
      const size_t i_idx =
          (((size_t)tree_set * (size_t)K0) + (size_t)fold)
          * (size_t)nodes + (size_t)leaf;
      I[i_idx] = F[f_idx];
    }
  }
}

// -------- C++ launcher --------
void cut_cuda_launcher(
    torch::Tensor F,      // uint16 (stores uint16)
    torch::Tensor FST,    // uint8
    torch::Tensor H,      // int64
    torch::Tensor H0,     // int64
    torch::Tensor V,      // int32
    torch::Tensor I,      // int16 (stores uint16)
    int tree_set,
    double L2,
    double lr,
    int qgrad_bits,
    int max_depth,
    double min_child_weight,
    double min_split_gain)
{
  const int treesets = F.size(0);
  const int K1       = H.size(0);
  const int nodes    = H.size(1);
  const int K0       = H0.size(0);
  const int D        = FST.size(2);

  const dim3 grid((1 << max_depth) - 1, 1, 1);
  const dim3 block(WARP_SIZE, 1, 1);

  auto stream = at::cuda::getCurrentCUDAStream();

  cut_cuda_kernel<<<grid, block, 0, stream.stream()>>>(
      reinterpret_cast<const uint16_t*>(F.data_ptr<uint16_t>()),
      FST.data_ptr<uint8_t>(),
      H.data_ptr<int64_t>(),
      H0.data_ptr<int64_t>(),
      V.data_ptr<int32_t>(),
      reinterpret_cast<uint16_t*>(I.data_ptr<uint16_t>()),
      treesets, K0, K1, nodes, D, tree_set,
      static_cast<float>(L2),
      static_cast<float>(lr),
      qgrad_bits,
      max_depth,
      static_cast<float>(min_child_weight),
      static_cast<float>(min_split_gain));
}
