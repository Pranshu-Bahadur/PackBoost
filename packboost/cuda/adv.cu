#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>          // <- add this

constexpr int WARP_SIZE = 32;

template <typename LeafT>
__global__ void advance_and_predict_kernel(
    int32_t* __restrict__ P,            // [N]
    const uint32_t* __restrict__ X,     // [R, M]
    const LeafT*  __restrict__ L_old,   // [K0, Dm, N]
    LeafT*        __restrict__ L_new,   // [K0, Dm, N]
    const int32_t* __restrict__ V,      // [rounds, K0, 2*nodes]
    const uint16_t* __restrict__ I,     // [rounds, K0, nodes]
    // dims/params
    int N, int R, int M, int K0, int Dm, int nodes,
    int rounds, int tree_set,
    int stride)
{
  const int tree_fold = blockIdx.x;   // 0..K0-1
  const int depth     = blockIdx.y;   // 0..(depths-1)
  const int iblk      = blockIdx.z;
  const int wi        = threadIdx.x;  // 0..31
  if (wi >= WARP_SIZE || tree_fold >= K0) return;

  const size_t Vbase = ((size_t)tree_set * (size_t)K0 + (size_t)tree_fold) * (size_t)(2 * nodes);
  const size_t Ibase = ((size_t)tree_set * (size_t)K0 + (size_t)tree_fold) * (size_t)nodes;

  for (int j = 0; j < stride; ++j) {
    const int k = 32 * (stride * iblk + j) + wi;
    if (k >= N) continue;

    uint16_t leaf_prev = 0;
    if (depth > 0) {
      const size_t off_old = (((size_t)tree_fold * (size_t)Dm) + (size_t)(depth - 1)) * (size_t)N + (size_t)k;
      leaf_prev = (uint16_t)L_old[off_old];
    }

    const int lo = (int)leaf_prev + ((1 << depth) - 1);
    const uint16_t li = I[Ibase + (size_t)lo];

    const uint32_t word = X[(size_t)li * (size_t)M + (size_t)(k >> 5)];
    const uint32_t bit  = (word >> (k & 31)) & 1u;

    const uint16_t leaf_new = (uint16_t)((leaf_prev << 1) | (uint16_t)bit);
    if (depth < Dm) {
      const size_t off_new = (((size_t)tree_fold * (size_t)Dm) + (size_t)depth) * (size_t)N + (size_t)k;
      L_new[off_new] = (LeafT)leaf_new;
    }

    const int add = V[Vbase + (size_t)(2 * lo + (int)bit)];
    atomicAdd(&P[k], add);
  }
}

static void launch_advpred(
    torch::Tensor P,  // int32 [N]
    torch::Tensor X,  // int32 (bitwise uint32) [R, M]
    torch::Tensor L_old,
    torch::Tensor L_new,
    torch::Tensor V,  // int32 [rounds, K0, 2*nodes]
    torch::Tensor I,  // int16 [rounds, K0, nodes] (uint16 payload)
    int tree_set)
{

  const int N      = (int)P.size(0);
  const int R      = (int)X.size(0);
  const int M      = (int)X.size(1);

  const int rounds = (int)V.size(0);
  const int K0     = (int)V.size(1);
  const int nodes2 = (int)V.size(2);
  const int nodes  = nodes2 / 2;
  const int Dm = (int)L_old.size(1);

  // infer depths (grid.y) and stride/grid.z like original Numba call
  const int depths = std::min(tree_set + 1, Dm + 1);

  const int zblocks = 512;                              // same as original workflow
  int stride = (N + (zblocks * 32) - 1) / (zblocks * 32);
  if (stride < 1) stride = 1;
  const int gz = std::max(1, zblocks);

  const dim3 grid((unsigned)K0, (unsigned)depths, (unsigned)gz);
  const dim3 block(WARP_SIZE);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (L_old.scalar_type() == at::kByte && L_new.scalar_type() == at::kByte) {
    advance_and_predict_kernel<uint8_t><<<grid, block, 0, stream.stream()>>>(
        P.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(X.data_ptr()),
        L_old.data_ptr<uint8_t>(),
        L_new.data_ptr<uint8_t>(),
        V.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(I.data_ptr<uint16_t>()),
        N, R, M, K0, Dm, nodes, rounds, tree_set, stride);
  } else if (L_old.scalar_type() == at::kShort && L_new.scalar_type() == at::kShort) {
    advance_and_predict_kernel<uint16_t><<<grid, block, 0, stream.stream()>>>(
        P.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(X.data_ptr()),
        L_old.data_ptr<uint16_t>(),
        L_new.data_ptr<uint16_t>(),
        V.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(I.data_ptr<uint16_t>()),
        N, R, M, K0, Dm, nodes, rounds, tree_set, stride);
  } else {
    TORCH_CHECK(false, "L_old/L_new must both be uint8 or both be int16 (uint16)");
  }
}

void advance_and_predict_launcher(
    torch::Tensor P, torch::Tensor X, torch::Tensor L_old, torch::Tensor L_new,
    torch::Tensor V, torch::Tensor I, int tree_set)
{
  launch_advpred(P.contiguous(), X.contiguous(), L_old.contiguous(), L_new.contiguous(),
                 V.contiguous(), I.contiguous(), tree_set);
}